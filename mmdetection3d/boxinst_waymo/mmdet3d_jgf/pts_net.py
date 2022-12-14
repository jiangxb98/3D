from mmcv.runner import BaseModule
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d_jgf.point_sa import AttentionPointEncoder

import torch
from torch import nn
import torch.nn.functional as F

from .modality_mapper import img2pc
from .point_ops import build_image_location_map_single
@MIDDLE_ENCODERS.register_module()
class GenPoints(BaseModule):
    '''
    定义attention的一些参数
    '''
    def __init__(self,
                mask_and_jitter=True,
                sparse_query_rate=4,
                cpts=512,
                cimg=512,
                point_attention_cfg=dict(
                    use_cls_token=True,
                    fore_attn=False,
                    num_layers=4,
                    pos_embedding='SIN',  # [MLP, SIN, NO]
                    fuse_method='CAT',   # [ADD, CAT, GATE]
                    input_img_channel=512,
                    input_pts_channel=512,
                    position_embedding_channel=512,
                    hidden_size=768,
                    num_heads=12,
                    dropout_rate=0.2,
                    intermediate_size=1024,
                )):

        self.mask_and_jitter=mask_and_jitter
        self.sparse_query_rate=sparse_query_rate
        
        # MAttn Transformer
        self.attention_layers = AttentionPointEncoder(point_attention_cfg)

        self.xyzd_embedding = nn.Sequential(
            nn.Linear(3, cpts),
            nn.LayerNorm(cpts),
            nn.ReLU(inplace=True),
            nn.Linear(cpts, cpts)
        )

        self.unknown_f3d = nn.Parameter(torch.zeros(cpts))
        self.unknown_f3d = nn.init.normal_(self.unknown_f3d)

        hidden_size = point_attention_cfg.hidden_size

        self.foreground_head = nn.Sequential(
            nn.Linear(hidden_size+cimg+3, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 2)
        )
        self.xyz_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.4),
            nn.Linear(512, 3)
        )


    def forward(self, points, img, pts_feat, img_feat, img_metas):
        '''
        点云特征和图片特征直接从backbone里拿就好了
        '''
        # 获取投影到相机的点云
        points, points_mask, ori_points_image, ori_points_image_mask = get_points_img(points, img_metas)
        gt_coords_3d = points
        # 对图片进行降采样
        downsampled_images = self.mean_pool(ori_points_image, ori_points_image_mask, kernel_size=4*2, stride=4*2)
        downsampled_image_masks = F.max_pool2d(ori_points_image_mask.float(), kernel_size=4*2, stride=4*2, padding=0)

        # 对点云进行处理 1-realpoints,0-padding,2-mask,3-jitter
        # 个人认为不需要pad点云到固定大小

        # 确定点云的中心，然后统一到局部坐标下

        device = points.device  # cuda

        impact_points_mask = points_mask==1 # (B, N), points with known 3D coords, unmasked, unjittered
        unmaksed_known_points = (impact_points_mask) + (points_mask==3) # (B, N), points has 3D coords, no masked, no padding
        nonpadding_points = (unmaksed_known_points) + (points_mask==2)  # (B, N), points known, no padding

        B, _, H, W = img.size()
        image_features = image_features  # (B,C,H,W) 选择步长为8的那一层作为image_features

        B, N, _ = points.size() 
        scale = self.sparse_query_rate
        qH, qW = H//scale, W//scale

        # hide and jitter point cloud
        jittered_cloud = points.clone()
        jittered_cloud[(points_mask==3).unsqueeze(-1).repeat(1, 1, 3)] += torch.rand((points_mask==3).sum()*3, device=points.device)*0.1 - 0.05
        jittered_cloud = jittered_cloud * (unmaksed_known_points.unsqueeze(-1))

        key_c2d = points[:, 6:8]  #(B,N,2)
        key_f2d = img2pc(image_features, key_c2d).transpose(-1, -2)
        key_f3d = self.xyzd_embedding(jittered_cloud) * (unmaksed_known_points.unsqueeze(-1)) + \
                  self.unknown_f3d.view(1, 1, -1).repeat(B, N, 1) * (~unmaksed_known_points.unsqueeze(-1))

        query_c2d = (build_image_location_map_single(qH, qW, device)*scale).view(1, -1, 2).repeat(B, 1, 1)     # (B, H*W, 2)
        query_f2d = img2pc(image_features, query_c2d).transpose(-1, -2)                         # (B, H*W, Ci)
        query_f3d = self.unknown_f3d.view(1, 1, -1).repeat(B, query_f2d.size(1), 1)

        # Self-attention to decode missing 3D features
        # only unmasked known foreground will be attended
        attn_mask = unmaksed_known_points
        query_f3d, key_f3d, cls_f3d = self.attention_layers(query_c2d, query_f2d, query_f3d, key_c2d, key_f2d, key_f3d, attn_mask)
        pred_key_coords_3d = self.xyz_head(key_f3d)  # pred_coords_3d   

        diff_xyz = (pred_key_coords_3d.detach() - points) * nonpadding_points.unsqueeze(-1)
        pred_key_foreground = self.foreground_head(torch.cat([key_f2d, key_f3d, diff_xyz], dim=-1))   # (B, N, 2) pred_foreground_logits

        pred_query_coords_3d = self.xyz_head(query_f3d)  # enriched_points
        pred_query_foreground = self.foreground_head(torch.cat([query_f2d, query_f3d, torch.zeros_like(pred_query_coords_3d)], dim=-1))

        return gt_coords_3d, pred_key_coords_3d, pred_key_foreground, 

    def loss(self, gt_coords, pred_coords, pred_key_foreground, real_point_mask):
        losses = {}
        loss_mask = pred_key_foreground.argmax(dim=-1).float()  # ignore background
        loss_mask = (loss_mask) * (real_point_mask!=0)          # ignore padding
        loss_depth = nn.SmoothL1Loss(reduction='none')(pred_coords, gt_coords).sum(dim=-1)
        loss_depth = loss_depth * loss_mask

        # balance mask/jitter/impact points
        l = 0
        l = l + (loss_depth * (real_point_mask==1)).sum(dim=1) / (((real_point_mask==1)*loss_mask).sum(dim=1)+1e-6) * 0.1
        l = l + (loss_depth * (real_point_mask==2)).sum(dim=1) / (((real_point_mask==2)*loss_mask).sum(dim=1)+1e-6)
        assert (real_point_mask!=3).all()
        loss_depth = l.mean().sum()
        losses['loss_depth'] = loss_depth
        return losses

    def mean_pool(self, points_image, points_iamge_mask, kernel_size=4, stride=4):
        n, h, w = points_iamge_mask.shape
        mean_pools = []
        for i, per_img in enumerate(points_image):  # 表示几张图片
            mean_pool = []
            unfold_pt_img_mask = F.unfold(points_iamge_mask[i].reshape((1, 1, h, w)).float(), kernel_size=kernel_size, stride=stride)
            unfold_mask = unfold_pt_img_mask.sum(dim=1)
            # X,Y,Z
            for j in range(len(per_img)):
                unfold_pt_img = F.unfold(per_img[j].reshape((1, 1, h, w)).float(), kernel_size=kernel_size, stride=stride)
                unfold_sum = unfold_pt_img.sum(dim=1)
                # 如果分母为0，也就是没有点，那么池化后的结果是nan
                unfold_mean = unfold_sum / unfold_mask
                # 0替换nan
                unfold_mean = torch.nan_to_num(unfold_mean, nan=0)
                mean_pool.append(unfold_mean.reshape(int(h/kernel_size), int(w/kernel_size)))
            mean_pools.append(torch.stack(mean_pool, dim=0))
        return torch.stack(mean_pools, dim=0)

def get_points_img(points, img_metas):
    '''
    获得投影到某个img上的点云
    Input: points[B,N,12]
    Return: points=(N,(x,y,z,i,e,lidar_idx,img_x,img_y)), points_mask=(N,)
    '''
    batch_points = []
    batch_points_mask = []
    batch_ori_points_image = []
    batch_ori_points_image_mask = []
    for i, per_img_metas in enumerate(img_metas):
        sample_img_id = per_img_metas['sample_img_id']
        points = points[i]

        # 1. 过滤掉没有投影到相机的点
        mask = (points[:, 6] == 0) | (points[:, 7] == 0)  # 真值列表
        mask_id = torch.where(mask)[0]  # 全局索引值
        in_img_points = points[mask]

        # 2. 新建 点云映射图
        ori_points_image = torch.zeros((1280,1920,3),dtype=torch.float)
        ori_points_image_mask = torch.zeros((1280,1920))
        # 3. 新建一个points只保存八个数据
        new_points = in_img_points[:,8]  # (N,8)
        for i, point in enumerate(in_img_points):
            if point[6] == sample_img_id:
                x_0 = point[8]
                y_0 = point[10]
                new_points[i,6:8] = torch.tensor(x_0,y_0)
                ori_points_image[int(y_0), int(x_0)] = torch.tensor([point[0], point[1], point[2]])
                ori_points_image_mask[int(y_0), int(x_0)] = 1               
            else:
                x_1 = point[9]
                y_1 = point[11]
                new_points[i,6:8] = torch.tensor(x_1,y_1)
                ori_points_image[int(y_1), int(x_1)] = torch.tensor([point[0], point[1], point[2]])
                ori_points_image_mask[int(y_1), int(x_1)] = 1

        ori_points_image = ori_points_image.permute(2,0,1)  # (3,1280,1920)
        ori_points_image_mask = ori_points_image_mask

        batch_points.append(new_points)
        batch_points_mask.append(mask_id)
        batch_ori_points_image.append(ori_points_image)
        batch_ori_points_image_mask.append(ori_points_image_mask)
    
    batch_ori_points_image = torch.stack(batch_ori_points_image, dim=0)
    batch_ori_points_image_mask = torch.stack(batch_ori_points_image_mask, dim=0)
    
    return batch_points, batch_points_mask, batch_ori_points_image, batch_ori_points_image_mask                        
                   