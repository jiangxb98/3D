import torch
import torchvision
from torch import nn
import torch.nn.functional as F

from mmcv.runner import BaseModule, force_fp32
from mmdet3d.models.builder import MIDDLE_ENCODERS

import numpy as np
from .modality_mapper import img2pc
from .point_sa import AttentionPointEncoder
from .point_ops import build_image_location_map_single

from easydict import EasyDict

class FPN(nn.Module):
    def __init__(self, in_channel=3, inter_size=128, out_channel=None) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channel, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(inter_size, inter_size, 3, padding=1),
            nn.BatchNorm2d(inter_size),
            nn.ReLU()
        )
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(inter_size, inter_size, 3, padding=1),
        #     nn.BatchNorm2d(inter_size),
        #     nn.ReLU()
        # )
        self.raise_channel = nn.Conv2d(inter_size*3, out_channel, 1)
        
    def forward(self, img):
        # img: (B, C, H, W)
        x1 = self.layer1(img)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        # x4 = self.layer4(x3)

        fpn = torch.cat([x1, x2, x3], dim=1)
        fpn = self.raise_channel(fpn)
        return fpn, x3

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
                out_img_size=112,
                out_cloud_size=512,  # box img内的点云数量
                mask_ratio=[0.0, 0.95],
                point_attention_cfg=EasyDict(dict(
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
                ))):
        super(GenPoints, self).__init__()
        self.mask_and_jitter=mask_and_jitter
        self.sparse_query_rate=sparse_query_rate
        self.out_img_size = out_img_size
        self.out_cloud_size = out_cloud_size
        self.mask_ratio = mask_ratio
        self.img_coords = build_image_location_map_single(out_img_size, out_img_size, 'cpu')

        self.cnn = FPN(3, 128, cimg)

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

        self.train_transforms = torch.nn.Sequential(
            torchvision.transforms.RandomAutocontrast(p=0.5),
            torchvision.transforms.RandomAdjustSharpness(np.random.rand()*2, p=0.5),
            torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.5, 0.5, 0.5, 0.3)], p=0.5),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def forward(self, points, img, img_metas, gt_bboxes, gt_labels):
        '''
        点云特征和图片特征直接从backbone里拿就好了
        '''
        boxes_img_infos_list = self.crop_boxes_img(img_metas, img, gt_bboxes, points, gt_labels)

        image = torch.cat(boxes_img_infos_list['box_img_list'], dim=0).squeeze()                # (B*, C, H, W)选择步长为8的那一层作为image_features
        device = image.device  # cuda
        # overlap_masks = torch.cat(boxes_img_infos_list['overlap_mask_list'], dim=0).to(device)        # (B * box数量, 1, H, W)
        sub_cloud = torch.cat(boxes_img_infos_list['sub_cloud_list'], dim=0)[:, :, :3].to(device)       # (B, N, 3) for x, y, z point cloud
        sub_cloud2d = torch.cat(boxes_img_infos_list['sub_cloud2d_list'], dim=0).to(device)             # (B, N, 2) for projected point cloud
        # ori_cloud2d = torch.cat(boxes_img_infos_list['ori_clouds2d_list'], dim=0).to(device)          # (B, N, 2) for original 2D coords refer to the full image
        real_point_mask = torch.cat(boxes_img_infos_list['real_point_mask_list'], dim=0).to(device)     # (B, N), 1-realpoint; 0-padding; 2-mask; 3-jitter
        # foreground_label = torch.cat(boxes_img_infos_list['foreground_label_list'], dim=0).to(device) # (B, N), 1-foreground; 0-background; 2-unknown
        
        pred_dict = {
            'batch_size': len(img_metas),
            'gt_nums': [len(label) for label in gt_labels],
        }

        impact_points_mask = real_point_mask==1                             # (B, N), points with known 3D coords, unmasked, unjittered
        unmaksed_known_points = (impact_points_mask) + (real_point_mask==3) # (B, N), points has 3D coords, no masked, no padding
        nonpadding_points = (unmaksed_known_points) + (real_point_mask==2)  # (B, N), points known, no padding

        # normalize point cloud
        sub_cloud_center = (sub_cloud * impact_points_mask.unsqueeze(-1)).sum(dim=1) / (impact_points_mask.sum(dim=1, keepdim=True)+1e-6)  # (B, 3)
        # only norm x&y coords
        sub_cloud = sub_cloud - sub_cloud_center.unsqueeze(1)
        sub_cloud = sub_cloud * (nonpadding_points).unsqueeze(-1)
        pred_dict['subcloud_center'] = sub_cloud_center  # 记录box img的中心点，后面可以恢复点云

        pred_dict['gt_coords_3d'] = sub_cloud

        # 1. extract information of images 
        B, _, H, W = image.size()  # (B,C,H,W) 选择步长为8的那一层作为image_features
        image_features, _ = self.cnn(image)

        # 2. build new cloud, which contains blank slots to be interpolated
        B, N, _ = sub_cloud.size() 
        scale = self.sparse_query_rate
        qH, qW = H//scale, W//scale

        # hide and jitter point cloud
        jittered_cloud = sub_cloud.clone()
        jittered_cloud[(real_point_mask==3).unsqueeze(-1).repeat(1, 1, 3)] += torch.rand((real_point_mask==3).sum()*3, device=device)*0.1 - 0.05
        jittered_cloud = jittered_cloud * (unmaksed_known_points.unsqueeze(-1))

        key_c2d = sub_cloud2d                                           # (B,N,2)
        key_f2d = img2pc(image_features, key_c2d).transpose(-1, -2)     # (B, N, Ci)
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
        pred_dict['pred_coords_3d'] = pred_key_coords_3d

        diff_xyz = (pred_key_coords_3d.detach() - sub_cloud) * nonpadding_points.unsqueeze(-1)
        pred_key_foreground = self.foreground_head(torch.cat([key_f2d, key_f3d, diff_xyz], dim=-1))   # (B, N, 2) pred_foreground_logits
        pred_dict['pred_foreground_logits'] = pred_key_foreground

        pred_query_coords_3d = self.xyz_head(query_f3d)  # enriched_points
        # 注意这个地方，预测出来的点云是局部的还是全局的？是局部的！所以我这里写了一个全局的
        pred_dict['enriched_points'] = pred_query_coords_3d
        global_enriched_points = []
        for i in range(B):
            global_enriched_points.append(pred_query_coords_3d[i] + sub_cloud_center[i])
        pred_dict['global_enriched_points'] = torch.stack(global_enriched_points, dim=0).to(device)
        pred_query_foreground = self.foreground_head(torch.cat([query_f2d, query_f3d, torch.zeros_like(pred_query_coords_3d)], dim=-1))
        pred_dict['enriched_foreground_logits'] = pred_query_foreground

        pred_dict['real_point_mask'] = real_point_mask

        return pred_dict
    
    @force_fp32(apply_to=('pred_coords_3d',))
    def loss(self, real_point_mask, gt_coords_3d, pred_coords_3d, **kwargs):
        real_point_mask = real_point_mask
        losses = {}

        gt_coords = gt_coords_3d                     # (B, N, 3)
        pred_coords = pred_coords_3d                 # (B, N, 3)

        loss_mask = kwargs['pred_foreground_logits'].argmax(dim=-1).float()  # ignore background
        loss_mask = (loss_mask) * (real_point_mask!=0)                          # ignore padding

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

    def crop_boxes_img(self, img_metas, img_, gt_bboxes_, points_, labels_):
        '''
        输入的图片是降采样8倍的,640,960 -> 80,120(B,C,H,W)
        '''
        device = img_.device

        boxes_img_infos = dict()
        boxes_img_infos['gt_bboxes'] = []
        boxes_img_infos['labels'] = []
        boxes_img_infos['box_img_list'] = []
        boxes_img_infos['sub_cloud_list'] = []
        boxes_img_infos['sub_cloud2d_list'] = []
        # boxes_img_infos['ori_cloud2d_list'] = []
        boxes_img_infos['real_point_mask_list'] = []
        # boxes_img_infos['foreground_label_list'] = []
        boxes_img_infos['overlap_mask_list'] = []

        # 获取sample_img_id图片内的点云(背景点，前景点),这一步很耗费时间
        batch_points, _ = get_points(points_, img_metas)  # list [B, N1,N2...]，不同帧得到的点云N不同

        for i, per_img_meta in enumerate(img_metas):
            gt_bboxes = gt_bboxes_[i]
            points_xy = batch_points[i]
            img = img_[i]
            labels = labels_[i]
            overlap_mask = torch.ones_like(img[0:1,:,:])  # 这个是为了解决盒子重叠的问题,我这里空着，没写
            img = torch.cat([img, overlap_mask], dim=0)  #(C+1,H,W)=(4,H,W)

            box_img_list = []
            sub_cloud_list = []
            sub_cloud2d_list = []
            ori_cloud2d_list = []
            real_point_mask_list = []
            foreground_label_list = []
            overlap_mask_list = []
            
            for j, gt_bbox in enumerate(gt_bboxes):
                x_1, y_1, x_2, y_2 = gt_bbox
                x_1, y_1, x_2, y_2 = int(torch.floor(x_1)), int(torch.floor(y_1)), int(torch.ceil(x_2)), int(torch.ceil(y_2))

                # 获取box内的点云
                gt_mask = (((points_xy[:, 6] > gt_bbox[0]) & (points_xy[:, 6] < gt_bbox[2])) &
                            ((points_xy[:, 7] > gt_bbox[1]) & (points_xy[:, 7] < gt_bbox[3])))
                sub_cloud2d = points_xy[gt_mask][:, 6:8]  # in gt box points (N, 2)，
                points = points_xy[gt_mask][:, :4]

                box_img = img[:, y_1: y_2, x_1: x_2]  # (C+1,H,W)

                box_size = max(x_2-x_1, y_2-y_1)
                out_shape = self.out_img_size #112
                # 这里是对特征进行插值的,和rgb的插值区别大不？
                box_img = F.interpolate(box_img.unsqueeze(dim=0), scale_factor=out_shape/box_size, mode='bilinear', align_corners=True, recompute_scale_factor=False)
                h, w = box_img.shape[-2:]
                num_padding = (int(np.floor((out_shape-w)/2)), int(np.ceil((out_shape-w)/2)), int(np.floor((out_shape-h)/2)), int(np.ceil((out_shape-h)/2)))
                box_img = torch.nn.functional.pad(box_img, num_padding)     # zero-padding to make it square
                # 点云映射到2D上的坐标也跟着改变，所以要减去2D Box img左上角的坐标点，同时也要resize然后再pad
                crop_sub_cloud2d = (sub_cloud2d.cpu() - torch.tensor([x_1, y_1])) * (out_shape/box_size) + np.array([num_padding[0], num_padding[2]])
                crop_sub_cloud2d = crop_sub_cloud2d.to(device)
                box_img, overlap_mask = box_img[:, 0:-1, :, :], box_img[:, -1, :, :]

                # sampling the point cloud to fixed size
                out_sub_cloud = torch.ones((self.out_cloud_size, 4))* (-9999)       # -9999 for paddings
                out_sub_cloud2d = torch.ones((self.out_cloud_size, 2)) * (-9999)    # -9999 for paddings 对应到box裁切的图片的2D坐标
                out_ori_cloud2d = torch.ones((self.out_cloud_size, 2)) * (-9999)    # 对应到整张图片的2D坐标
                out_real_point_mask = torch.zeros((self.out_cloud_size))    # 0 for padding, 1 for real points, 2 for masked, 3 for jittered
                # out_foreground_label = torch.ones((self.out_cloud_size))*2   # 0 for background, 1 for foreground, 2 for unknown                
                
                sub_cloud = points  #(N, 4)
                sub_cloud2d = sub_cloud2d  #(N,2)
                # foreground_label = batch_pointstorch.ones((points.shape(0)))  #(N,)
                out_cloud_size = self.out_cloud_size
                if sub_cloud.shape[0] > out_cloud_size:
                    sample_idx = np.random.choice(np.arange(sub_cloud.shape[0]), out_cloud_size, replace=False)            # random sampling
                    out_sub_cloud[...] = sub_cloud[sample_idx]
                    out_sub_cloud2d[...] = crop_sub_cloud2d[sample_idx]
                    out_ori_cloud2d[...] = sub_cloud2d[sample_idx]
                    out_real_point_mask[...] = 1
                    # out_foreground_label[...] = foreground_label[sample_idx]  #(512,)
                elif sub_cloud.shape[0] <= out_cloud_size:
                    pc_size = sub_cloud.shape[0]
                    out_sub_cloud[:pc_size] = sub_cloud
                    out_sub_cloud2d[:pc_size] = crop_sub_cloud2d
                    out_ori_cloud2d[:pc_size] = sub_cloud2d
                    out_real_point_mask[:pc_size] = 1
                    # out_foreground_label[:pc_size] = foreground_label

                    # 3D 点少于25个，所以需要用2D点来弥补，采样的2D 点数量是512-pc_size
                    # sample 2D points, leave blank for 3D coords
                    p = ((box_img[0]!=0).all(dim=0) * 1).type(torch.float64)    # only sample pixels from not-padding-area
                    p = p / p.sum()
                    resample = (p>0).sum() < (out_cloud_size - pc_size)
                    sample_idx = np.random.choice(np.arange(out_shape * out_shape), out_cloud_size - pc_size, replace=resample.cpu().numpy(),
                                                p=p.reshape(-1).cpu().numpy())
                    sampled_c2d = self.img_coords.view(-1, 2)[sample_idx, :].numpy()
                    out_sub_cloud2d[pc_size:, :] = torch.tensor(sampled_c2d)
                    out_ori_cloud2d[pc_size:, :] = torch.tensor((sampled_c2d - np.array([num_padding[0], num_padding[2]])) / (out_shape/box_size) + np.array([x_1, y_1]))                 
                
                # random mask/jitter points
                if not self.training:
                    self.mask_ratio = [0.0, 0.0]
                num_real_points = (out_real_point_mask==1).sum()
                mask_ratio = np.random.rand() * (self.mask_ratio[1] - self.mask_ratio[0]) + self.mask_ratio[0]     # randomly choose from (r_min, r_max)
                num_mask = min(int(mask_ratio * num_real_points), max(0, num_real_points - 5))        # leave at least 5 points
                idx = np.random.choice(np.arange(num_real_points), num_mask, replace=False)
                mask_idx = idx
                out_real_point_mask[mask_idx] = 2   # 2 for masked

                box_img_list.append(box_img)
                sub_cloud_list.append(out_sub_cloud)
                sub_cloud2d_list.append(out_sub_cloud2d)
                ori_cloud2d_list.append(out_ori_cloud2d)
                real_point_mask_list.append(out_real_point_mask)
                # foreground_label_list.append(out_foreground_label)
                overlap_mask_list.append(overlap_mask)

            boxes_img_infos['gt_bboxes'].append(gt_bboxes)
            boxes_img_infos['labels'].append(labels)
            boxes_img_infos['box_img_list'].append(torch.stack(box_img_list, dim=0))
            boxes_img_infos['sub_cloud_list'].append(torch.stack(sub_cloud_list, dim=0))
            boxes_img_infos['sub_cloud2d_list'].append(torch.stack(sub_cloud2d_list, dim=0))
            # boxes_img_infos['ori_cloud2d_list'].append(torch.stack(ori_cloud2d_list, dim=0))
            boxes_img_infos['real_point_mask_list'].append(torch.stack(real_point_mask_list, dim=0))
            # boxes_img_infos['foreground_label_list'].append(torch.stack(foreground_label_list, dim=0))
            boxes_img_infos['overlap_mask_list'].append(torch.stack(overlap_mask_list, dim=0))

        return boxes_img_infos

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
        in_img_points = points
        # mask = (points[:, 6] == 0) | (points[:, 7] == 0)  # 真值列表
        mask_id = torch.where(points)[0]  # 全局索引值
        # in_img_points = points[mask]

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

def get_points(points_, img_metas):
    '''
    获得投影到某个img上的点云
    Input: points[B,N,12]
    Return: points=(N,(x,y,z,i,e,lidar_idx,img_x,img_y)), points_mask=(N,)
    '''
    batch_points = []
    batch_points_mask = []

    for i, per_img_metas in enumerate(img_metas):
        sample_img_id = per_img_metas['sample_img_id']
        points = points_[i]
        scale = per_img_metas['scale_factor'][0]
        device = points.device

        # 1. 过滤掉没有投影到相机的点
        in_img_points = points
        # mask = (points[:, 6] == sample_img_id) | (points[:, 7] == sample_img_id)  # 真值列表
        mask_id = torch.where(points)[0].to(device)  # 全局索引值
        # in_img_points = points[mask]

        # 2. 新建一个points只保存八个数据
        new_points = in_img_points[:,:8].to(device)  # (N,8)
        for i, point in enumerate(in_img_points):
            if point[6] == sample_img_id:
                x_0 = point[8] * scale
                y_0 = point[10] * scale
                new_points[i,6:8] = torch.tensor([x_0,y_0])          
            else:
                x_1 = point[9] * scale
                y_1 = point[11] * scale
                new_points[i,6:8] = torch.tensor([x_1,y_1])

        batch_points.append(new_points)
        batch_points_mask.append(mask_id)
    
    return batch_points, batch_points_mask      

            
