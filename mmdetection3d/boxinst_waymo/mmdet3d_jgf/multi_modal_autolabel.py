# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp
import torch
from torch.nn import functional as F
from scipy.sparse.csgraph import connected_components  # CCL

import mmcv
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32

from mmdet.core import multi_apply, bbox2result

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result, merge_aug_bboxes_3d, show_result)
from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector

from mmseg.models import SEGMENTORS
from mmdet3d.models.segmentors.base import Base3DSegmentor

from cowa3d_common.ops.ccl.ccl_utils import spccl, voxel_spccl, voxelized_sampling, sample
from .fsd_ops import scatter_v2, get_inner_win_inds
from .utils import pts_semantic_confusion_matrix


def filter_almost_empty(coors, min_points):
    new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
    cnt_per_point = unq_cnt[unq_inv]
    valid_mask = cnt_per_point >= min_points
    return valid_mask

def find_connected_componets(points, batch_idx, dist):

    device = points.device
    bsz = batch_idx.max().item() + 1
    base = 0
    components_inds = torch.zeros_like(batch_idx) - 1

    for i in range(bsz):
        batch_mask = batch_idx == i
        if batch_mask.any():
            this_points = points[batch_mask]
            dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
            dist_mat = (dist_mat ** 2).sum(2) ** 0.5
            adj_mat = dist_mat < dist
            adj_mat = adj_mat.cpu().numpy()
            c_inds = connected_components(adj_mat, directed=False)[1]
            c_inds = torch.from_numpy(c_inds).to(device).int() + base
            base = c_inds.max().item() + 1
            components_inds[batch_mask] = c_inds

    assert len(torch.unique(components_inds)) == components_inds.max().item() + 1

    return components_inds

def find_connected_componets_single_batch(points, batch_idx, dist):

    device = points.device

    this_points = points
    dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
    dist_mat = (dist_mat ** 2).sum(2) ** 0.5
    # dist_mat = torch.cdist(this_points[:, :2], this_points[:, :2], p=2)
    adj_mat = dist_mat < dist
    adj_mat = adj_mat.cpu().numpy()
    c_inds = connected_components(adj_mat, directed=False)[1]
    c_inds = torch.from_numpy(c_inds).to(device).int()

    return c_inds

def modify_cluster_by_class(cluster_inds_list):
    new_list = []
    for i, inds in enumerate(cluster_inds_list):
        cls_pad = inds.new_ones((len(inds),)) * i
        inds = torch.cat([cls_pad[:, None], inds], 1)
        # inds = F.pad(inds, (1, 0), 'constant', i)
        new_list.append(inds)
    return new_list

@SEGMENTORS.register_module()
@DETECTORS.register_module()
class VoteSegmentor(Base3DSegmentor):

    def __init__(self,
                 voxel_layer,
                 voxel_encoder,
                 middle_encoder,
                 backbone,
                 segmentation_head,
                 decode_neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 pretrained=None,
                 tanh_dims=None,
                 need_full_seg=False,
                 only_one_frame_label=True,
                 sweeps_num=1,
                 **extra_kwargs):
        super().__init__(init_cfg=init_cfg)

        self.voxel_layer = Voxelization(**voxel_layer)

        self.voxel_encoder = builder.build_voxel_encoder(voxel_encoder)
        self.middle_encoder = builder.build_middle_encoder(middle_encoder)
        self.backbone = builder.build_backbone(backbone)
        self.segmentation_head = builder.build_head(segmentation_head)
        self.segmentation_head.train_cfg = train_cfg
        self.segmentation_head.test_cfg = test_cfg
        self.decode_neck = builder.build_neck(decode_neck)

        assert voxel_encoder['type'] == 'DynamicScatterVFE'


        self.print_info = {}
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.cfg = train_cfg if train_cfg is not None else test_cfg
        self.num_classes = segmentation_head['num_classes']
        self.save_list = []
        self.point_cloud_range = voxel_layer['point_cloud_range']
        self.voxel_size = voxel_layer['voxel_size']
        self.tanh_dims = tanh_dims
        self.need_full_seg = need_full_seg
        self.only_one_frame_label = only_one_frame_label
        self.sweeps_num = sweeps_num
    
    def encode_decode(self, ):
        return None
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        return NotImplementedError

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.
        Args:
            points (list[torch.Tensor]): Points of each sample.
        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

    def extract_feat(self, points, img_metas):
        """Extract features from points."""
        batch_points, coors = self.voxelize(points)
        coors = coors.long()
        voxel_features, voxel_coors, voxel2point_inds = self.voxel_encoder(batch_points, coors, return_inv=True)
        voxel_info = self.middle_encoder(voxel_features, voxel_coors)
        x = self.backbone(voxel_info)[0]
        padding = -1
        voxel_coors_dropped = x['voxel_feats'] # bug, leave it for feature modification
        if 'shuffle_inds' not in voxel_info:
            voxel_feats_reorder = x['voxel_feats']
        else:
            # this branch only used in SST-based FSD 
            voxel_feats_reorder = self.reorder(x['voxel_feats'], voxel_info['shuffle_inds'], voxel_info['voxel_keep_inds'], padding) #'not consistent with voxel_coors any more'
        # 注意这里out[1]这个索引的意义
        out = self.decode_neck(batch_points, coors, voxel_feats_reorder, voxel2point_inds, padding)

        return out, coors, batch_points
    
    
    def reorder(self, data, shuffle_inds, keep_inds, padding=-1):
        '''
        Padding dropped voxel and reorder voxels.  voxel length and order will be consistent with the output of voxel_encoder.
        '''
        num_voxel_no_drop = len(shuffle_inds)
        data_dim = data.size(1)

        temp_data = padding * data.new_ones((num_voxel_no_drop, data_dim))
        out_data = padding * data.new_ones((num_voxel_no_drop, data_dim))

        temp_data[keep_inds] = data
        out_data[shuffle_inds] = temp_data

        return out_data
    

    def forward_train(self,
                      points,
                      img_metas,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      as_subsegmentor=False,
                      pts_semantic_mask=None,
                      gt_box_type=1):
        # if self.tanh_dims is not None:
        #     for p in points:
        #         p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        # elif points[0].size(1) in (4,5):
        #     # a hack way to scale the intensity and elongation in WOD
        #     points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]

        sweep_ind = None
        if self.sweeps_num > 1:
            sweep_ind = [p[:, -1] for p in points]
            points = [p[:, :-1].contiguous() for p in points]
        if not self.only_one_frame_label:
            # multiple frames with label
            sweep_ind = None
        # labels: B*N,1 vote_targets: B*N,3 vote_mask(bool) B*N,1
        labels, vote_targets, vote_mask = self.segmentation_head.get_targets(points, gt_bboxes_3d, gt_labels_3d, gt_box_type, img_metas)
        # neck_out(list):[points_feature(N,64+3), bool(N,)]pts_coors: N,4(batch_id,vx,vy,vz) points=input points 没变
        neck_out, pts_coors, points = self.extract_feat(points, img_metas)  # 提取每个点的特征，这里提取特征是通过体素化提取，然后将体素特征再传播给对应的点

        losses = dict()

        feats = neck_out[0]
        valid_pts_mask = neck_out[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]
        labels = labels[valid_pts_mask]
        vote_targets = vote_targets[valid_pts_mask]
        vote_mask = vote_mask[valid_pts_mask]
        if pts_semantic_mask is not None:
            pts_semantic_mask = torch.cat(pts_semantic_mask, dim=0)
            pts_semantic_mask = pts_semantic_mask[valid_pts_mask]

        assert feats.size(0) == labels.size(0)

        # select points with sweep_ind==0
        if sweep_ind is not None:
            sweep_ind = torch.cat(sweep_ind, dim=0)
            sweep_ind = sweep_ind[valid_pts_mask]
            sweep_ind_mask = sweep_ind == 0
            feats = feats[sweep_ind_mask]
            points = points[sweep_ind_mask]
            pts_coors = pts_coors[sweep_ind_mask]
            labels = labels[sweep_ind_mask]
            vote_targets = vote_targets[sweep_ind_mask]
            vote_mask = vote_mask[sweep_ind_mask]
            if pts_semantic_mask is not None:
                pts_semantic_mask = pts_semantic_mask[sweep_ind_mask]

        if as_subsegmentor:
            loss_decode, preds_dict = self.segmentation_head.forward_train(
                points, feats, img_metas, labels, vote_targets, vote_mask, 
                return_preds=True, pts_semantic_mask_full=pts_semantic_mask)
            losses.update(loss_decode)

            seg_logits = preds_dict['seg_logits']
            vote_preds = preds_dict['vote_preds']

            offsets = self.segmentation_head.decode_vote_targets(vote_preds)

            output_dict = dict(
                seg_points=points,
                seg_logits=preds_dict['seg_logits'],
                seg_vote_preds=preds_dict['vote_preds'],
                offsets=offsets,
                seg_feats=feats,
                batch_idx=pts_coors[:, 0],
                losses=losses
            )
        else:
            raise NotImplementedError
            loss_decode = self.segmentation_head.forward_train(feats, img_metas, labels, vote_targets, vote_mask, return_preds=False)
            losses.update(loss_decode)
            output_dict = losses

        return output_dict


    def simple_test(self, points, img_metas, gt_bboxes_3d=None, gt_labels_3d=None, rescale=False):

        # if self.tanh_dims is not None:
        #     for p in points:
        #         p[:, self.tanh_dims] = torch.tanh(p[:, self.tanh_dims])
        # elif points[0].size(1) in (4,5):
        #     points = [torch.cat([p[:, :3], torch.tanh(p[:, 3:])], dim=1) for p in points]
        # TODO output with sweep_ind
        sweep_ind = None
        if self.sweeps_num > 1:
            sweep_ind = [p[:, -1] for p in points]
            points = [p[:, :-1].contiguous() for p in points]

        seg_pred = []
        x, pts_coors, points = self.extract_feat(points, img_metas)
        feats = x[0]
        valid_pts_mask = x[1]
        points = points[valid_pts_mask]
        pts_coors = pts_coors[valid_pts_mask]

        # select points with sweep_ind==0
        if sweep_ind is not None:
            sweep_ind = torch.cat(sweep_ind, dim=0)
            sweep_ind = sweep_ind[valid_pts_mask]
            sweep_ind_mask = sweep_ind == 0
            feats = feats[sweep_ind_mask]
            points = points[sweep_ind_mask]
            pts_coors = pts_coors[sweep_ind_mask]

        if self.need_full_seg:
            seg_logits, seg_logits_full, vote_preds = \
                self.segmentation_head.forward_test(feats, img_metas, self.test_cfg, True)
        else:
            seg_logits, vote_preds = self.segmentation_head.forward_test(feats, img_metas, self.test_cfg)
            seg_logits_full = None

        offsets = self.segmentation_head.decode_vote_targets(vote_preds)

        output_dict = dict(
            seg_points=points,  # ori points 
            seg_logits=seg_logits,  # 
            seg_vote_preds=vote_preds,
            offsets=offsets,
            seg_feats=feats,
            batch_idx=pts_coors[:, 0],
            seg_logits_full=seg_logits_full
        )

        return output_dict

@DETECTORS.register_module()
class MultiModalAutoLabel(Base3DDetector):
    """Base class of Multi-modality autolabel."""

    def __init__(self,
                with_pts_branch=True,
                with_img_branch=True,
                img_backbone=None,  #
                img_neck=None,  #
                img_bbox_head=None,  #
                img_mask_branch=None,  #
                img_mask_head=None,  #
                img_segm_head= None,
                pretrained=None,
                img_roi_head=None,
                img_rpn_head=None,
                middle_encoder_pts=None,  # points completion 
                pts_segmentor=None, #
                pts_voxel_layer=None,
                pts_voxel_encoder=None,
                pts_middle_encoder=None,
                pts_backbone=None,  #
                pts_neck=None,
                pts_bbox_head=None, #
                pts_roi_head=None,  # 二阶段，暂时先不用
                pts_fusion_layer=None,            
                train_cfg=None,  # 记住cfg是分img和pts的
                test_cfg=None,  #
                cluster_assigner=None,  #
                init_cfg=None,
                only_one_frame_label=True,
                sweeps_num=1,
                gt_box_type=1):  # 1 is 3d, 2 is 2d
        super(MultiModalAutoLabel, self).__init__(init_cfg=init_cfg)

        # 这里也是个全局的配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        self.with_pts_branch = with_pts_branch
        self.with_img_branch = with_img_branch
        self.gt_box_type = gt_box_type

        # FSD
        if pts_segmentor is not None:  # 
            self.pts_segmentor = builder.build_detector(pts_segmentor)
        
        self.pts_backbone = builder.build_backbone(pts_backbone)  # 
        if pts_neck is not None:
            self.pts_neck = builder.build_neck(pts_neck)
        if pts_voxel_layer is not None:
            self.pts_voxel_layer = Voxelization(**pts_voxel_layer)
        if pts_voxel_encoder is not None:
            self.pts_voxel_encoder = builder.build_voxel_encoder(pts_voxel_encoder)
        if pts_middle_encoder is not None:
            self.pts_middle_encoder = builder.build_middle_encoder(pts_middle_encoder)

        if pts_bbox_head is not None:  # 
            pts_train_cfg = train_cfg.pts if train_cfg.pts else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg.pts else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
            self.num_classes = self.pts_bbox_head.num_classes

        self.pts_roi_head = pts_roi_head  # 
        if pts_roi_head is not None:
            rcnn_train_cfg = train_cfg.pts.rcnn if train_cfg.pts else None
            pts_roi_head.update(train_cfg=rcnn_train_cfg)
            pts_roi_head.update(test_cfg=test_cfg.pts.rcnn)
            pts_roi_head.pretrained = pretrained
            self.roi_head = builder.build_head(pts_roi_head)
        # 这里有个pts_cfg配置
        self.pts_cfg = self.train_cfg.pts if self.train_cfg.pts else self.test_cfg.pts
        if 'radius' in cluster_assigner:
            raise NotImplementedError
            self.cluster_assigner = SSGAssigner(**cluster_assigner)
        elif 'hybrid' in cluster_assigner:
            raise NotImplementedError
            cluster_assigner.pop('hybrid')
            self.cluster_assigner = HybridAssigner(**cluster_assigner)
        elif 'voxelspccl' in cluster_assigner:
            cluster_assigner.pop('voxelspccl')
            self.cluster_assigner = ClusterAssignerVoxelSPCCL(**cluster_assigner)
        else:
            self.cluster_assigner = ClusterAssigner(**cluster_assigner)

        self.cluster_assigner.num_classes = self.num_classes
        self.print_info = {}
        self.as_rpn = pts_bbox_head.get('as_rpn', False)

        self.runtime_info = dict()
        self.only_one_frame_label = only_one_frame_label
        self.sweeps_num = sweeps_num
        
        # 点云补全
        if middle_encoder_pts:
            self.middle_encoder_pts = builder.build_middle_encoder(middle_encoder_pts)

        # BoxInst
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)
        if img_bbox_head is not None:  # train_cfg.img 送入到SingleStageDetector的bbox_head
            img_train_cfg = train_cfg.img if train_cfg else None
            img_bbox_head.update(train_cfg=img_train_cfg)
            img_test_cfg = test_cfg.img if test_cfg else None
            img_bbox_head.update(test_cfg=img_test_cfg)
            self.img_bbox_head = builder.build_head(img_bbox_head)
        if img_mask_branch is not None:
            self.img_mask_branch = builder.build_head(img_mask_branch)
        if img_mask_head is not None:
            self.img_mask_head = builder.build_head(img_mask_head)           
        if img_segm_head is not None:
            self.img_segm_head = builder.build_head(img_segm_head)
        else:
            self.img_segm_head = None
        
    @property
    def with_pts_bbox(self):
        """bool: Whether the detector has a 3D box head."""
        return hasattr(self, 'pts_bbox_head') and self.pts_bbox_head is not None

    @property
    def with_img_bbox(self):
        """bool: Whether the detector has a 2D image box head."""
        return hasattr(self, 'img_bbox_head') and self.img_bbox_head is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 3D backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a neck in 3D detector branch."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_middle_encoder_pts(self):
        return hasattr(self, 'middle_encoder_pts') and self.middle_encoder_pts is not None

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats  # FPN 5layers, img_feats=5*[B,C,new_H,new_W]

    # single stage fsd extract pts feat
    def extract_pts_feat(self, points, pts_feats, pts_cluster_inds, img_metas, center_preds):
        """Extract features from points."""
        if not self.with_pts_backbone:
            return None
        if not self.with_pts_bbox:
            return None      
        cluster_xyz, _, inv_inds = scatter_v2(center_preds, pts_cluster_inds, mode='avg', return_inv=True)

        f_cluster = points[:, :3] - cluster_xyz[inv_inds]

        out_pts_feats, cluster_feats, out_coors = self.pts_backbone(points, pts_feats, pts_cluster_inds, f_cluster)
        out_dict = dict(
            cluster_feats=cluster_feats,
            cluster_xyz=cluster_xyz,
            cluster_inds=out_coors
        )
        if self.as_rpn:
            out_dict['cluster_pts_feats'] = out_pts_feats
            out_dict['cluster_pts_xyz'] = points

        return out_dict

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)  # FPN 5layers, img_feats=5*[B,C,new_H,new_W]
        # pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, None)

    def forward_train(self,
                      points=None,  # need
                      img_metas=None,  # need
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,  # need
                      gt_bboxes=None,  # need
                      gt_masks=None,
                      img=None, # need
                      proposals=None,
                      gt_bboxes_ignore=None,
                      runtime_info=dict(),
                      pts_semantic_mask=None,
                      pts_instance_mask=None,
                      gt_semantic_seg=None,
                      gt_yaw=None,
                      lidar_density=None,
                      roi_points=None,
                      ):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor, optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        self.runtime_info = runtime_info # stupid way to get arguements from children class
        if gt_bboxes_3d:
            gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
            gt_labels_3d = [l[l>=0] for l in gt_labels_3d]
        losses = dict()

        # 测试拿出来了，如果不测试就注释掉，这个是放在img_branch里的
        points = self.filter_points(img_metas=img_metas, points=points)

        # 1. Image Branch
        if self.with_img_backbone and self.with_img_branch:
            # 过滤掉没有投影到相机的点
            points = self.filter_points(img_metas=img_metas, points_=points)

            img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
            losses_img = self.forward_img_train(
                                                img,
                                                img_feats,
                                                points=points,
                                                img_metas=img_metas,
                                                gt_bboxes=gt_bboxes,
                                                gt_labels=gt_labels,
                                                gt_bboxes_ignore=gt_bboxes_ignore,
                                                gt_masks=gt_masks,
                                                proposals=proposals)
            losses.update(losses_img)

        # 2. Points Completion
        if self.with_middle_encoder_pts:
            encoder_points = self.encoder_pts(points, img, img_metas, gt_bboxes, gt_labels)
            loss_depth = self.middle_encoder_pts.loss(**encoder_points)
            losses.update(loss_depth)
            enriched_points = encoder_points['enriched_foreground_logits']

        # 3. FSD-Points Branch    
        if self.with_pts_backbone and self.with_pts_branch:
            # 若使用2d box做监督
            if self.gt_box_type == 2:
                gt_bboxes = self.combine_yaw_info(gt_bboxes, gt_yaw) 
                gt_bboxes_3d = gt_bboxes
                gt_labels_3d = gt_labels
                points = self.scale_cp_cor(points, img_metas) # 提前将3d对应的2d坐标，根据输入图片resize大小来放缩
            rpn_outs = self.forward_pts_train(points,
                                                img_metas,
                                                gt_bboxes_3d,
                                                gt_labels_3d,
                                                gt_bboxes_ignore,
                                                runtime_info,
                                                pts_semantic_mask,
                                                lidar_density,
                                                roi_points
                                                )
            losses.update(rpn_outs['rpn_losses'])

            # fsd second stage
            if self.pts_roi_head is not None:

                proposal_list = self.pts_bbox_head.get_bboxes(
                    rpn_outs['cls_logits'], rpn_outs['reg_preds'], rpn_outs['cluster_xyz'], rpn_outs['cluster_inds'], img_metas
                )

                assert len(proposal_list) == len(gt_bboxes_3d)

                pts_xyz, pts_feats, pts_batch_inds = self.prepare_multi_class_roi_input(
                    rpn_outs['all_input_points'],
                    rpn_outs['valid_pts_feats'],
                    rpn_outs['seg_feats'],
                    rpn_outs['pts_mask'],
                    rpn_outs['pts_batch_inds'],
                    rpn_outs['valid_pts_xyz']
                )

                roi_losses = self.pts_roi_head.forward_train(
                    pts_xyz,
                    pts_feats,
                    pts_batch_inds,
                    img_metas,
                    proposal_list,
                    gt_bboxes_3d,
                    gt_labels_3d,
                )

                losses.update(roi_losses)

        return losses

    def forward_pts_train(self,
                          points,
                          img_metas,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes_ignore=None,
                          runtime_info=None,
                          pts_semantic_mask=None,
                          lidar_density=None,  # (B, sample_roi_points)
                          roi_points=None      # (B, sample_roi_points, 3)
                          ):
        """Forward function for point cloud branch.
        """

        self.runtime_info = runtime_info # stupid way to get arguements from children class
        losses = {}
        gt_bboxes_3d = [b[l>=0] for b, l in zip(gt_bboxes_3d, gt_labels_3d)]
        gt_labels_3d = [l[l>=0] for l in gt_labels_3d]
        # seg_points=points(N,3/12)没变, seg_logits(N,3), seg_vote_preds(N,9), vote_preds decode后的偏置offsets(N,9) seg_feats(N,64+3)每个点的特征
        seg_out_dict = self.pts_segmentor(points=points, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True,
                                    pts_semantic_mask=pts_semantic_mask, gt_box_type=self.gt_box_type)
        # 上述这一步需要强调的是，本身的votesegmentor已经给每个点分配了一个(伪)label进行了监督，反而没有输出
        seg_feats = seg_out_dict['seg_feats']
        if self.train_cfg.get('detach_segmentor', False):
            seg_feats = seg_feats.detach()
        seg_loss = seg_out_dict['losses']
        losses.update(seg_loss)

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'].detach(),
            seg_vote_preds=seg_out_dict['seg_vote_preds'].detach(),
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets=seg_out_dict['offsets'].detach(),
        )
        if self.pts_cfg.get('pre_voxelization_size', None) is not None:  # 可以省略
            dict_to_sample = self.pre_voxelize(dict_to_sample)  # 体素化从原始点云的N点变成了体素Nv个点
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out

        # 过滤掉空的voxel然后进行聚类得到每个非空voxel对应的聚类结果list, [(cls_id, batch_idx, cluster_id),(非空,3),(非空,3)]，valid_mask_list(bool)[(topk-Nv,),(topk-Nv,),(topk-Nv,)]
        pts_cluster_inds, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], gt_bboxes_3d, gt_labels_3d,  origin_points=sampled_out['seg_points']) # per cls list
        if isinstance(pts_cluster_inds, list):
            pts_cluster_inds = torch.cat(pts_cluster_inds, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)
        # 得到多少个实例
        num_clusters = len(torch.unique(pts_cluster_inds, dim=0)) * torch.ones((1,), device=pts_cluster_inds.device).float()
        losses['num_clusters'] = num_clusters
        # 去除空的voxel，注意这里的输出结果是经过了两次过滤，一次是得分＞阈值，第二次是去掉空的
        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])
        # points是前面两次过滤后的点，不同类别下的cat在一块，它对应的batch，cls，cluster_id都存在pts_cluster_inds参数里
        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)
        losses['num_fg_points'] = torch.ones((1,), device=points.device).float() * len(points)
        # 第一次 SIR：输出的结果只是实例，实例包含多个点，但是都没有输出，所以最终的长度是小于等于上面的结果的
        extracted_outs = self.extract_pts_feat(points, pts_feats, pts_cluster_inds, img_metas, combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds'] # [class, batch, groups]
        # 最终得到instance的特征、坐标、索引信息
        assert (cluster_inds[:, 0]).max().item() < self.num_classes
        # 得到预测的结果cls, reg.每个输出都是按照类别输出的[(instance,1or8),(instance,1or8),(instacne,1or8)]
        outs = self.pts_bbox_head(cluster_feats, cluster_xyz, cluster_inds)
        loss_inputs = (outs['cls_logits'], outs['reg_preds']) + (cluster_xyz, cluster_inds) + (gt_bboxes_3d, gt_labels_3d, img_metas)
        det_loss = self.pts_bbox_head.loss(
            *loss_inputs, iou_logits=outs.get('iou_logits', None), 
            gt_bboxes_ignore=gt_bboxes_ignore, gt_box_type=self.gt_box_type,
            lidar_density=lidar_density, roi_points=roi_points
            )
        if hasattr(self.pts_bbox_head, 'print_info'):
            self.print_info.update(self.pts_bbox_head.print_info)
        losses.update(det_loss)
        losses.update(self.print_info)

        if self.as_rpn:
            output_dict = dict(
                rpn_losses=losses,
                cls_logits=outs['cls_logits'],
                reg_preds=outs['reg_preds'],
                cluster_xyz=cluster_xyz,
                cluster_inds=cluster_inds,
                all_input_points=dict_to_sample['seg_points'],
                valid_pts_feats=extracted_outs['cluster_pts_feats'],
                valid_pts_xyz=extracted_outs['cluster_pts_xyz'],
                seg_feats=dict_to_sample['seg_feats'],
                pts_mask=sampled_out['fg_mask_list'],
                pts_batch_inds=dict_to_sample['batch_idx'],
            )
            return output_dict
        else:
            output_dict = dict(rpn_losses=losses)
            return output_dict

    def forward_img_train(self,
                        img,
                        x,
                        points,
                        img_metas,
                        gt_bboxes,
                        gt_labels,
                        gt_bboxes_ignore=None,
                        gt_masks=None,
                        proposals=None,
                        **kwargs):
        """Forward function for image branch.

        This function works similar to the forward function of Faster R-CNN.

        Args:
            x (list[torch.Tensor]): Image features of shape (B, C, H, W)
                of multiple levels.
            points:
            img_metas (list[dict]): Meta information of images.
            gt_bboxes (list[torch.Tensor]): Ground truth boxes of each image
                sample.
            gt_labels (list[torch.Tensor]): Ground truth labels of boxes.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            gt_masks ()
            proposals (list[torch.Tensor], optional): Proposals of each sample.
                Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        cls_score, bbox_pred, centerness, param_pred = \
                self.img_bbox_head(x, self.img_mask_head.param_conv)
        img_bbox_head_loss_inputs = (cls_score, bbox_pred, centerness) + (
            gt_bboxes, gt_labels, img_metas)
        losses, coors, level_inds, img_inds, gt_inds = self.img_bbox_head.loss(
            *img_bbox_head_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        
        mask_feat = self.img_mask_branch(x)  # 1x16xHxW
        if self.img_segm_head is not None:
            segm_pred = self.img_segm_head(x[0])
            loss_segm = self.img_segm_head.loss(segm_pred, gt_masks, gt_labels)
            losses.update(loss_segm)

        inputs = (cls_score, centerness, param_pred, coors, level_inds, img_inds, gt_inds)
        param_pred, coors, level_inds, img_inds, gt_inds = self.img_mask_head.training_sample(*inputs)  # each image predict a maximum of 64 instance object
        mask_pred = self.img_mask_head(mask_feat, param_pred, coors, level_inds, img_inds)  # 64x1x2hx2w
        loss_mask = self.img_mask_head.loss(img, img_metas, mask_pred, gt_inds, gt_bboxes,
                                        gt_masks, gt_labels, points)
        losses.update(loss_mask)
        return losses

    # Base3DDetector的forward_test()内进入simple_test函数
    def simple_test(self, points, img_metas, img=None, rescale=False,
                    gt_bboxes_3d=None, gt_labels_3d=None, pts_semantic_mask=None):
        """Test function without augmentaiton."""

        results_list = [dict() for i in range(len(img_metas))]

        if self.with_pts_bbox and self.with_pts_branch:
            # list[i]={'boxes_3d':LiDARInstance3DBoxes,'scores_3d':(N3d,1),'labels_3d':(N3d,1)}
            pts_results = self.simple_test_pts(
                points, img_metas, rescale, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask)
            for i in range(len(img_metas)):
                results_list[i]['boxes_3d']=pts_results[i]['boxes_3d']    # (LiDARInstance3DBoxes)
                results_list[i]['scores_3d']=pts_results[i]['scores_3d']  # (N3d,1)
                results_list[i]['labels_3d']=pts_results[i]['labels_3d']  # (N3d,1)

        # finished
        # 这里需要注意的是输出是不是按照batch来的，我下面的赋值都是按照batch来索引赋值的
        if self.with_img_bbox and self.with_img_branch:
            img_feats = self.extract_img_feat(img=img, img_metas=img_metas)
            bbox_results, mask_results = self.simple_test_img(img_feats, img_metas, rescale=rescale) # bbox_results, mask_results
            for i in range(len(img_metas)):
                results_list[i]['img_bbox'] = bbox_results[i]   # 长度是类别数 x (N,5),左上角和右下角坐标+得分
                results_list[i]['img_mask'] = mask_results[i]   # 长度是类别数 x (1280,1920)

        return results_list

    # 检查完成 Return：list(zip(bbox_results, mask_results))
    def simple_test_img(self, img_feats, img_metas, proposals=None, rescale=False):
        feat = img_feats
        outputs = self.img_bbox_head.simple_test(
            feat, self.img_mask_head.param_conv, img_metas, rescale=rescale)  # 注意这个outputs的输出是不是按照batch来的
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(*outputs)  # 框的两个角位置坐标+得分，所属类别，滤波器参数(N,233)，box在当前level的位置(N,2)，所属的layer索引
        bbox_results = [
            bbox2result(det_bbox, det_label, self.img_bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]  # 转为list，长度是和num_classes一样的
        mask_feat = self.img_mask_branch(feat)
        mask_results = self.img_mask_head.simple_test(
            mask_feat,
            det_labels,
            det_params,
            det_coors,
            det_level_inds,
            img_metas,
            self.img_bbox_head.num_classes,
            rescale=rescale) # mask_results 是个列表
        # return list(zip(bbox_results, mask_results))
        return bbox_results, mask_results

    def simple_test_pts(self, points, img_metas, rescale, gt_bboxes_3d, gt_labels_3d, pts_semantic_mask):
        """Test function of point cloud branch."""
        out = []
        # first stage fsd output
        rpn_outs = self.simple_test_single_fsd(points=points,
                                               img_metas=img_metas,
                                               gt_bboxes_3d=gt_bboxes_3d,
                                               gt_labels_3d=gt_labels_3d,)  # proposal_list, seg_logits_full
        
        proposal_list = rpn_outs['proposal_list']  # LidarInstance3DBoxes, box_scores, box_label
        if self.test_cfg.pts.return_mode in [0, 2]:   # 0: both, 1: detection, 2: segmentation
            seg_logits_full = rpn_outs.get('seg_logits_full')
            assert isinstance(seg_logits_full, list)
            with_confusion_matrix = self.test_cfg.pts.get('with_confusion_matrix', False)
            for b_seg in seg_logits_full:
                if with_confusion_matrix and (pts_semantic_mask is not None):
                    assert len(pts_semantic_mask) == 1
                    if self.sweeps_num > 1 and self.only_one_frame_label:
                        pts_semantic_mask[0] = pts_semantic_mask[0][points[:, -1]==0]
                    b_pred = b_seg.argmax(1)
                    confusion_matrix = pts_semantic_confusion_matrix(
                        b_pred + 1,
                        pts_semantic_mask[0],
                        self.test_cfg.pts.get('num_seg_cls') + 1)    # add cls: unlabel
                    out.append(dict(seg3d_confusion_matrix=confusion_matrix))
                else:
                    out.append(dict(segmap_3d=F.softmax(b_seg, dim=1).argmax(1).cpu()))

        if self.test_cfg.pts.return_mode == 2:  # 只返回semantic segment结果
            return out

        if self.test_cfg.pts.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results  # type(bbox_results)=list len(list)=batch_size list[i]={'boxes_3d':LiDARInstance3DBoxes,'scores_3d':(N3d,1),'labels_3d':(N3d,1)}

        if self.num_classes > 1 or self.test_cfg.pts.get('enable_multi_class_test', False):
            prepare_func = self.prepare_multi_class_roi_input
        else:
            prepare_func = self.prepare_roi_input

        pts_xyz, pts_feats, pts_batch_inds = prepare_func(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )

        results = self.pts_roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )
        if self.test_cfg.pts.return_mode == 1: # 只返回detection结果
            return results
        assert len(out) == len(results)
        for idx in range(len(out)):
            out[idx].update(results[idx])
        return out
 
    def simple_test_single_fsd(self, points, img_metas, imgs=None, rescale=False, gt_bboxes_3d=None, gt_labels_3d=None):
        """Test function without augmentaiton."""
        if gt_bboxes_3d is not None:
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]
            assert isinstance(gt_bboxes_3d, list)
            assert isinstance(gt_labels_3d, list)
            assert len(gt_bboxes_3d) == len(gt_labels_3d) == 1, 'assuming single sample testing'

        seg_out_dict = self.pts_segmentor.simple_test(points, img_metas, rescale=False)

        seg_feats = seg_out_dict['seg_feats']
        seg_logits_full = seg_out_dict.get('seg_logits_full', None)
        assert seg_out_dict['batch_idx'].max() == 0     # for inference, batch size = 0.

        dict_to_sample = dict(
            seg_points=seg_out_dict['seg_points'],
            seg_logits=seg_out_dict['seg_logits'],
            seg_vote_preds=seg_out_dict['seg_vote_preds'],
            seg_feats=seg_feats,
            batch_idx=seg_out_dict['batch_idx'],
            vote_offsets = seg_out_dict['offsets']
        )
        if self.pts_cfg.get('pre_voxelization_size', None) is not None:
            dict_to_sample = self.pre_voxelize(dict_to_sample)
        sampled_out = self.sample(dict_to_sample, dict_to_sample['vote_offsets'], gt_bboxes_3d, gt_labels_3d) # per cls list in sampled_out 返回每个点加上前面预测的offsets,得到的投票点

        # we filter almost empty voxel in clustering, so here is a valid_mask 比较慢，计算量大
        pts_cluster_inds, valid_mask_list = self.cluster_assigner(sampled_out['center_preds'], sampled_out['batch_idx'], gt_bboxes_3d, gt_labels_3d, origin_points=sampled_out['seg_points']) # per cls list

        if isinstance(pts_cluster_inds, list):
            pts_cluster_inds = torch.cat(pts_cluster_inds, dim=0) #[N, 3], (cls_id, batch_idx, cluster_id)

        sampled_out = self.update_sample_results_by_mask(sampled_out, valid_mask_list)

        combined_out = self.combine_classes(sampled_out, ['seg_points', 'seg_logits', 'seg_vote_preds', 'seg_feats', 'center_preds'])

        points = combined_out['seg_points']
        pts_feats = torch.cat([combined_out['seg_logits'], combined_out['seg_vote_preds'], combined_out['seg_feats']], dim=1)
        assert len(pts_cluster_inds) == len(points) == len(pts_feats)
        # 这个函数里面涉及到
        extracted_outs = self.extract_pts_feat(points, pts_feats, pts_cluster_inds, img_metas,  combined_out['center_preds'])
        cluster_feats = extracted_outs['cluster_feats']  # N_instance vector
        cluster_xyz = extracted_outs['cluster_xyz']
        cluster_inds = extracted_outs['cluster_inds']
        assert (cluster_inds[:, 1] == 0).all()

        outs = self.pts_bbox_head(cluster_feats, cluster_xyz, cluster_inds)  # cls_logits, reg_preds 一个类别一个tensor(N,1),(N,8)
        # 这里用到一些参数nms_pre等 bbox_list: LidarInstance3DBoxes box_scores box_label
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs['cls_logits'], outs['reg_preds'],
            cluster_xyz, cluster_inds, img_metas,
            rescale=rescale,
            iou_logits=outs.get('iou_logits', None))

        if self.as_rpn:
            output_dict = dict(
                all_input_points=dict_to_sample['seg_points'],
                valid_pts_feats=extracted_outs['cluster_pts_feats'],
                valid_pts_xyz=extracted_outs['cluster_pts_xyz'],
                seg_feats=dict_to_sample['seg_feats'],
                pts_mask=sampled_out['fg_mask_list'],
                pts_batch_inds=dict_to_sample['batch_idx'],
                proposal_list=bbox_list,
                seg_logits_full=[seg_logits_full]
            )
            return output_dict
        else:
            # bbox_results = [
            #     bbox3d2result(bboxes, scores, labels)
            #     for bboxes, scores, labels in bbox_list
            # ]
            # return bbox_results
            output_dict = dict(
                proposal_list=bbox_list,
                seg_logits_full=[seg_logits_full])
            return output_dict

    # x
    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]
    # x
    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_img_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats
    # x
    def aug_test_pts(self, feats, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.pts_bbox_head(x)
            bbox_list = self.pts_bbox_head.get_bboxes(
                *outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas,
                                            self.pts_bbox_head.test_cfg)
        return merged_bboxes
    # x
    def show_results(self, data, result, out_dir):
        """Results visualization.

        Args:
            data (dict): Input points and the information of the sample.
            result (dict): Prediction results.
            out_dir (str): Output directory of visualization result.
        """
        for batch_id in range(len(result)):
            if isinstance(data['points'][0], DC):
                points = data['points'][0]._data[0][batch_id].numpy()
            elif mmcv.is_list_of(data['points'][0], torch.Tensor):
                points = data['points'][0][batch_id]
            else:
                ValueError(f"Unsupported data type {type(data['points'][0])} "
                           f'for visualization!')
            if isinstance(data['img_metas'][0], DC):
                pts_filename = data['img_metas'][0]._data[0][batch_id][
                    'pts_filename']
                box_mode_3d = data['img_metas'][0]._data[0][batch_id][
                    'box_mode_3d']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                pts_filename = data['img_metas'][0][batch_id]['pts_filename']
                box_mode_3d = data['img_metas'][0][batch_id]['box_mode_3d']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            file_name = osp.split(pts_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'
            inds = result[batch_id]['pts_bbox']['scores_3d'] > 0.1
            pred_bboxes = result[batch_id]['pts_bbox']['boxes_3d'][inds]

            # for now we convert points and bbox into depth mode
            if (box_mode_3d == Box3DMode.CAM) or (box_mode_3d
                                                  == Box3DMode.LIDAR):
                points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                                   Coord3DMode.DEPTH)
                pred_bboxes = Box3DMode.convert(pred_bboxes, box_mode_3d,
                                                Box3DMode.DEPTH)
            elif box_mode_3d != Box3DMode.DEPTH:
                ValueError(
                    f'Unsupported box_mode_3d {box_mode_3d} for conversion!')

            pred_bboxes = pred_bboxes.tensor.cpu().numpy()
            show_result(points, None, pred_bboxes, out_dir, file_name)

    # GenPoints Points Completion
    def encoder_pts(self, points, img, img_metas, gt_bboxes, gt_labels):
        pts_dict = self.middle_encoder_pts(points, img, img_metas, gt_bboxes, gt_labels)
        return pts_dict

    # FSD_two_stage
    def prepare_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        pts_mask = pts_mask[0]
        assert points.shape[0] == pts_seg_feats.shape[0] == pts_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()
        
        pad_feats = cluster_pts_feats.new_zeros(points.shape[0], cluster_pts_feats.shape[1])
        pad_feats[pts_mask] = cluster_pts_feats
        assert torch.isclose(points[pts_mask], cluster_pts_xyz).all()

        cat_feats = torch.cat([pad_feats, pts_seg_feats], dim=1)

        return points, cat_feats, pts_batch_inds

    def prepare_multi_class_roi_input(self, points, cluster_pts_feats, pts_seg_feats, pts_mask, pts_batch_inds, cluster_pts_xyz):
        assert isinstance(pts_mask, list)
        bg_mask = sum(pts_mask) == 0
        assert points.shape[0] == pts_seg_feats.shape[0] == bg_mask.shape[0] == pts_batch_inds.shape[0]

        if self.training and self.train_cfg.get('detach_seg_feats', False):
            pts_seg_feats = pts_seg_feats.detach()

        if self.training and self.train_cfg.get('detach_cluster_feats', False):
            cluster_pts_feats = cluster_pts_feats.detach()


        ##### prepare points for roi head
        fg_points_list = [points[m] for m in pts_mask]
        all_fg_points = torch.cat(fg_points_list, dim=0)

        assert torch.isclose(all_fg_points, cluster_pts_xyz).all()

        bg_pts_xyz = points[bg_mask]
        all_points = torch.cat([bg_pts_xyz, all_fg_points], dim=0)
        #####

        ##### prepare features for roi head
        fg_seg_feats_list = [pts_seg_feats[m] for m in pts_mask]
        all_fg_seg_feats = torch.cat(fg_seg_feats_list, dim=0)
        bg_seg_feats = pts_seg_feats[bg_mask]
        all_seg_feats = torch.cat([bg_seg_feats, all_fg_seg_feats], dim=0)

        num_out_points = len(all_points)
        assert num_out_points == len(all_seg_feats)

        pad_feats = cluster_pts_feats.new_zeros(bg_mask.sum(), cluster_pts_feats.shape[1])
        all_cluster_pts_feats = torch.cat([pad_feats, cluster_pts_feats], dim=0)
        #####

        ##### prepare batch inds for roi head
        bg_batch_inds = pts_batch_inds[bg_mask]
        fg_batch_inds_list = [pts_batch_inds[m] for m in pts_mask]
        fg_batch_inds = torch.cat(fg_batch_inds_list, dim=0)
        all_batch_inds = torch.cat([bg_batch_inds, fg_batch_inds], dim=0)


        # pad_feats[pts_mask] = cluster_pts_feats

        cat_feats = torch.cat([all_cluster_pts_feats, all_seg_feats], dim=1)

        # sort for roi extractor
        all_batch_inds, inds = all_batch_inds.sort()
        all_points = all_points[inds]
        cat_feats = cat_feats[inds]

        return all_points, cat_feats, all_batch_inds
    
    # 这个是fsd第二阶段的simple
    def simple_test_fsd_two_stage(self, points, img_metas, imgs=None, rescale=False,
                    gt_bboxes_3d=None, gt_labels_3d=None,
                    pts_semantic_mask=None):

        out = []

        rpn_outs = super().simple_test(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
        )

        proposal_list = rpn_outs['proposal_list']
        if self.test_cfg.return_mode in [0, 2]:   # 0: both, 1: detection, 2: segmentation
            seg_logits_full = rpn_outs.get('seg_logits_full')
            assert isinstance(seg_logits_full, list)
            with_confusion_matrix = self.test_cfg.get('with_confusion_matrix', False)
            for b_seg in seg_logits_full:
                if with_confusion_matrix and (pts_semantic_mask is not None):
                    assert len(pts_semantic_mask) == 1
                    if self.sweeps_num > 1 and self.only_one_frame_label:
                        pts_semantic_mask[0] = pts_semantic_mask[0][points[:, -1]==0]
                    b_pred = b_seg.argmax(1)
                    confusion_matrix = pts_semantic_confusion_matrix(
                        b_pred + 1,
                        pts_semantic_mask[0],
                        self.test_cfg.get('num_seg_cls') + 1)    # add cls: unlabel
                    out.append(dict(seg3d_confusion_matrix=confusion_matrix))
                else:
                    out.append(dict(segmap_3d=F.softmax(b_seg, dim=1).argmax(1).cpu()))
        if self.test_cfg.return_mode == 2:
            return out

        if self.test_cfg.get('skip_rcnn', False):
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in proposal_list
            ]
            return bbox_results

        if self.num_classes > 1 or self.test_cfg.get('enable_multi_class_test', False):
            prepare_func = self.prepare_multi_class_roi_input
        else:
            prepare_func = self.prepare_roi_input

        pts_xyz, pts_feats, pts_batch_inds = prepare_func(
            rpn_outs['all_input_points'],
            rpn_outs['valid_pts_feats'],
            rpn_outs['seg_feats'],
            rpn_outs['pts_mask'],
            rpn_outs['pts_batch_inds'],
            rpn_outs['valid_pts_xyz']
        )


        results = self.roi_head.simple_test(
            pts_xyz,
            pts_feats,
            pts_batch_inds,
            img_metas,
            proposal_list,
            gt_bboxes_3d,
            gt_labels_3d,
        )
        if self.test_cfg.return_mode == 1:
            return results
        assert len(out) == len(results)
        for idx in range(len(out)):
            out[idx].update(results[idx])
        return out
    
    def extract_fg_by_gt(self, point_list, gt_bboxes_3d, gt_labels_3d, extra_width):
        if isinstance(gt_bboxes_3d[0], list):
            assert len(gt_bboxes_3d) == 1
            assert len(gt_labels_3d) == 1
            gt_bboxes_3d = gt_bboxes_3d[0]
            gt_labels_3d = gt_labels_3d[0]

        bsz = len(point_list)

        new_point_list = []
        for i in range(bsz):
            points = point_list[i]
            gts = gt_bboxes_3d[i].to(points.device)
            if len(gts) == 0:
                this_fg_mask = points.new_zeros(len(points), dtype=torch.bool)
                this_fg_mask[:min(1000, len(points))] = True
            else:
                if isinstance(extra_width, dict):
                    this_labels = gt_labels_3d[i]
                    enlarged_gts_list = []
                    for cls in range(self.num_classes):
                        cls_mask = this_labels == cls
                        if cls_mask.any():
                            this_enlarged_gts = gts[cls_mask].enlarged_box(extra_width[cls])
                            enlarged_gts_list.append(this_enlarged_gts)
                    enlarged_gts = gts.cat(enlarged_gts_list)
                else:
                    enlarged_gts = gts.enlarged_box(extra_width)
                pts_inds = enlarged_gts.points_in_boxes(points[:, :3])
                this_fg_mask = pts_inds > -1
                if not this_fg_mask.any():
                    this_fg_mask[:min(1000, len(points))] = True
            
            new_point_list.append(points[this_fg_mask])
        return new_point_list

    # SingleStageFSD
    def pre_voxelize(self, data_dict):
        batch_idx = data_dict['batch_idx']
        points = data_dict['seg_points']  # (N,3or12)

        voxel_size = torch.tensor(self.pts_cfg.pre_voxelization_size, device=batch_idx.device)
        pc_range = torch.tensor(self.cluster_assigner.point_cloud_range, device=points.device)
        coors = torch.div(points[:, :3] - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').long()
        coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        new_coors, unq_inv  = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        # 这里体素化，从这里可以获得原始点云对应的voxel
        voxelized_data_dict = {}
        for data_name in data_dict:
            data = data_dict[data_name]
            if data.dtype in (torch.float, torch.float16):
                voxelized_data, voxel_coors = scatter_v2(data, coors, mode='avg', return_inv=False, new_coors=new_coors, unq_inv=unq_inv)
                voxelized_data_dict[data_name] = voxelized_data

        voxelized_data_dict['batch_idx'] = voxel_coors[:, 0]
        return voxelized_data_dict

    def sample(self, dict_to_sample, offset, gt_bboxes_3d=None, gt_labels_3d=None):

        if self.pts_cfg.get('group_sample', False):
            return self.group_sample(dict_to_sample, offset)

        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        if seg_logits.size(1) == self.num_classes:
            seg_scores = seg_logits.sigmoid()
        else:
            raise NotImplementedError

        offset = offset.reshape(-1, self.num_classes, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls

        batch_idx = dict_to_sample['batch_idx']
        batch_size = batch_idx.max().item() + 1
        for cls in range(self.num_classes):
            cls_score_thr = cfg['score_thresh'][cls]
            # 在 test 阶段返回的是 得分＞阈值的点
            fg_mask = self.get_fg_mask(seg_scores, seg_points, cls, batch_idx, gt_bboxes_3d, gt_labels_3d)

            if len(torch.unique(batch_idx[fg_mask])) < batch_size:
                one_random_pos_per_sample = self.get_sample_beg_position(batch_idx, fg_mask)
                fg_mask[one_random_pos_per_sample] = True # at least one point per sample

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, cls, :]
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list  # list(bool) [(Nv,1),(Nv,1),(Nv,1)]
        output_dict['center_preds'] = center_preds_list  # list [(fg_mask=True,3),(fg_mask=True,3),(fg_mask=True,3)]

        return output_dict

    def update_sample_results_by_mask(self, sampled_out, valid_mask_list):
        for k in sampled_out:
            old_data = sampled_out[k]
            if len(old_data[0]) == len(valid_mask_list[0]) or 'fg_mask' in k:
                if 'fg_mask' in k:
                    new_data_list = []
                    for data, mask in zip(old_data, valid_mask_list):
                        new_data = data.clone()
                        new_data[data] = mask
                        assert new_data.sum() == mask.sum()
                        new_data_list.append(new_data)
                    sampled_out[k] = new_data_list
                else:
                    new_data_list = [data[mask] for data, mask in zip(old_data, valid_mask_list)]
                    sampled_out[k] = new_data_list
        return sampled_out

    def combine_classes(self, data_dict, name_list):
        out_dict = {}
        for name in data_dict:
            if name in name_list:
                out_dict[name] = torch.cat(data_dict[name], 0)
        return out_dict

    def get_fg_mask(self, seg_scores, seg_points, cls_id, batch_inds, gt_bboxes_3d, gt_labels_3d):
        if self.training and self.train_cfg.pts.get('disable_pretrain', False) and not self.runtime_info.get('enable_detection', False):
            seg_scores = seg_scores[:, cls_id]
            topks = self.train_cfg.pts.get('disable_pretrain_topks', [100, 100, 100])
            k = min(topks[cls_id], len(seg_scores))
            top_inds = torch.topk(seg_scores, k)[1]
            fg_mask = torch.zeros_like(seg_scores, dtype=torch.bool)
            fg_mask[top_inds] = True
        else:
            seg_scores = seg_scores[:, cls_id]
            cls_score_thr = self.pts_cfg['score_thresh'][cls_id]
            if self.training:
                buffer_thr = self.runtime_info.get('threshold_buffer', 0)
            else:
                buffer_thr = 0
            fg_mask = seg_scores > cls_score_thr + buffer_thr

        # add fg points
        cfg = self.train_cfg.pts if self.training else self.test_cfg.pts

        # 没修改
        if cfg.get('add_gt_fg_points', False):
            import pdb;pdb.set_trace()
            bsz = len(gt_bboxes_3d)
            assert len(seg_scores) == len(seg_points) == len(batch_inds)
            point_list = self.split_by_batch(seg_points, batch_inds, bsz)
            gt_fg_mask_list = []

            for i, points in enumerate(point_list):
                
                gt_mask = gt_labels_3d[i] == cls_id
                gts = gt_bboxes_3d[i][gt_mask]

                if not gt_mask.any() or len(points) == 0:
                    gt_fg_mask_list.append(gt_mask.new_zeros(len(points), dtype=torch.bool))
                    continue
                
                gt_fg_mask_list.append(gts.points_in_boxes(points) > -1)
            
            gt_fg_mask = self.combine_by_batch(gt_fg_mask_list, batch_inds, bsz)
            fg_mask = fg_mask | gt_fg_mask
            
        return fg_mask

    def split_by_batch(self, data, batch_idx, batch_size):
        assert batch_idx.max().item() + 1 <= batch_size
        data_list = []
        for i in range(batch_size):
            sample_mask = batch_idx == i
            data_list.append(data[sample_mask])
        return data_list

    def combine_by_batch(self, data_list, batch_idx, batch_size):
        assert len(data_list) == batch_size
        if data_list[0] is None:
            return None
        data_shape = (len(batch_idx),) + data_list[0].shape[1:]
        full_data = data_list[0].new_zeros(data_shape)
        for i, data in enumerate(data_list):
            sample_mask = batch_idx == i
            full_data[sample_mask] = data
        return full_data

    def get_sample_beg_position(self, batch_idx, fg_mask):
        assert batch_idx.shape == fg_mask.shape
        inner_inds = get_inner_win_inds(batch_idx.contiguous())
        pos = torch.where(inner_inds == 0)[0]
        return pos

    def group_sample(self, dict_to_sample, offset):

        """
        For argoverse 2 dataset, where the number of classes is large
        """

        bsz = dict_to_sample['batch_idx'].max().item() + 1
        assert bsz == 1, "Maybe some codes need to be modified if bsz > 1"
        # combine all classes as fg class.
        cfg = self.train_cfg if self.training else self.test_cfg

        seg_logits = dict_to_sample['seg_logits']
        assert (seg_logits < 0).any() # make sure no sigmoid applied

        assert seg_logits.size(1) == self.num_classes + 1 # we have background class
        seg_scores = seg_logits.softmax(1)

        offset = offset.reshape(-1, self.num_classes + 1, 3)
        seg_points = dict_to_sample['seg_points'][:, :3]
        fg_mask_list = [] # fg_mask of each cls
        center_preds_list = [] # fg_mask of each cls


        cls_score_thrs = cfg['score_thresh']
        group_lens = cfg['group_lens']
        num_groups = len(group_lens)
        assert num_groups == len(cls_score_thrs)
        assert isinstance(cls_score_thrs, (list, tuple))
        grouped_score = self.gather_group(seg_scores[:, :-1], group_lens) # without background score

        beg = 0
        for i, group_len in enumerate(group_lens):
            end = beg + group_len

            fg_mask = grouped_score[:, i] > cls_score_thrs[i]

            if not fg_mask.any():
                fg_mask[0] = True # at least one point

            fg_mask_list.append(fg_mask)

            this_offset = offset[fg_mask, beg:end, :] 
            offset_weight = self.get_offset_weight(seg_logits[fg_mask, beg:end])
            assert torch.isclose(offset_weight.sum(1), offset_weight.new_ones(len(offset_weight))).all()
            this_offset = (this_offset * offset_weight[:, :, None]).sum(dim=1)
            this_points = seg_points[fg_mask, :]
            this_centers = this_points + this_offset
            center_preds_list.append(this_centers)
            beg = end
        assert end == 26, 'for 26class argo'


        output_dict = {}
        for data_name in dict_to_sample:
            data = dict_to_sample[data_name]
            cls_data_list = []
            for fg_mask in fg_mask_list:
                cls_data_list.append(data[fg_mask])

            output_dict[data_name] = cls_data_list
        output_dict['fg_mask_list'] = fg_mask_list
        output_dict['center_preds'] = center_preds_list

        return output_dict
    
    def get_offset_weight(self, seg_logit):
        mode = self.cfg['offset_weight']
        if mode == 'max':
            weight = ((seg_logit - seg_logit.max(1)[0][:, None]).abs() < 1e-6).float()
            assert ((weight == 1).any(1)).all()
            weight = weight / weight.sum(1)[:, None] # in case of two max values
            return weight
        else:
            raise NotImplementedError
    
    def gather_group(self, scores, group_lens):
        assert (scores >= 0).all()
        score_per_group = []
        beg = 0
        for group_len in group_lens:
            end = beg + group_len
            score_this_g = scores[:, beg:end].sum(1)
            score_per_group.append(score_this_g)
            beg = end
        assert end == scores.size(1) == sum(group_lens)
        gathered_score = torch.stack(score_per_group, dim=1)
        assert gathered_score.size(1) == len(group_lens)
        return  gathered_score

    def filter_points(self, img_metas, points):

        for i, per_img_metas in enumerate(img_metas):
            sample_img_id = per_img_metas['sample_img_id']
            # 过滤掉没有投影到相机的点
            mask = (points[i][:, 6] == sample_img_id) | (points[i][:, 7] == sample_img_id)  # 真值列表
            # mask_id = torch.where(mask)[0]  # 全局索引值
            points[i] = points[i][mask]

        return points

    def scale_cp_cor(self, points, img_metas):

        for i, per_img_metas in enumerate(img_metas):
            scale = img_metas[i]['scale_factor'][0]
            if points[i].shape[1] == 12:
                points[i][:,8:12] = points[i][:,8:12] * scale
        return points

    def combine_yaw_info(self, bboxes, gt_yaw):
        for i in range(len(bboxes)):
            bboxes[i] = torch.cat((bboxes[i], gt_yaw[i].unsqueeze(dim=1)), dim=1)
        return bboxes

class ClusterAssigner(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        min_points,
        point_cloud_range,
        connected_dist,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        cluster_inds_list, valid_mask_list = \
            multi_apply(self.forward_single_class, points_list, batch_idx_list, self.class_names, origin_points)
        cluster_inds_list = modify_cluster_by_class(cluster_inds_list)
        return cluster_inds_list, valid_mask_list

    def forward_single_class(self, points, batch_idx, class_name, origin_points):
        batch_idx = batch_idx.int()

        if isinstance(self.cluster_voxel_size, dict):
            cluster_vsize = self.cluster_voxel_size[class_name]
        elif isinstance(self.cluster_voxel_size, list):
            cluster_vsize = self.cluster_voxel_size[self.class_names.index(class_name)]
        else:
            cluster_vsize = self.cluster_voxel_size

        voxel_size = torch.tensor(cluster_vsize, device=points.device)
        pc_range = torch.tensor(self.point_cloud_range, device=points.device)
        coors = torch.div(points - pc_range[None, :3], voxel_size[None, :], rounding_mode='floor').int()
        # coors = coors[:, [2, 1, 0]] # to zyx order
        coors = torch.cat([batch_idx[:, None], coors], dim=1)

        valid_mask = filter_almost_empty(coors, min_points=self.min_points)
        if not valid_mask.any():
            valid_mask = ~valid_mask
            # return coors.new_zeros((3,0)), valid_mask

        points = points[valid_mask]
        batch_idx = batch_idx[valid_mask]
        coors = coors[valid_mask]
        # elif len(points) 

        sampled_centers, voxel_coors, inv_inds = scatter_v2(points, coors, mode='avg', return_inv=True)

        if isinstance(self.connected_dist, dict):
            dist = self.connected_dist[class_name]
        elif isinstance(self.connected_dist, list):
            dist = self.connected_dist[self.class_names.index(class_name)]
        else:
            dist = self.connected_dist

        if self.training:
            cluster_inds = find_connected_componets(sampled_centers, voxel_coors[:, 0], dist)
        else:
            cluster_inds = find_connected_componets_single_batch(sampled_centers, voxel_coors[:, 0], dist)
        assert len(cluster_inds) == len(sampled_centers)

        cluster_inds_per_point = cluster_inds[inv_inds]
        cluster_inds_per_point = torch.stack([batch_idx, cluster_inds_per_point], 1)
        return cluster_inds_per_point, valid_mask

class ClusterAssignerVoxelSPCCL(torch.nn.Module):
    ''' Generating cluster centers for each class and assign each point to cluster centers
    '''

    def __init__(
        self,
        cluster_voxel_size,
        min_points,
        point_cloud_range,
        connected_dist,
        class_names=['Car', 'Cyclist', 'Pedestrian'],
    ):
        super().__init__()
        self.cluster_voxel_size = cluster_voxel_size
        self.min_points = min_points
        self.connected_dist = connected_dist
        self.point_cloud_range = point_cloud_range
        self.class_names = class_names

    @torch.no_grad()
    def forward(self, points_list, batch_idx_list, gt_bboxes_3d=None, gt_labels_3d=None, origin_points=None):
        gt_bboxes_3d = None 
        gt_labels_3d = None
        assert self.num_classes == len(self.class_names)
        points = torch.cat(points_list).contiguous()
        batch_id = torch.cat(batch_idx_list).to(torch.int32)
        bsz = batch_id.max().item() + 1
        class_id = []
        voxel_config = []
        for i, p in enumerate(points_list):
            class_id.append((torch.ones(len(p), dtype=torch.int32) * i).to(points.device))
            cluster_vsize = self.cluster_voxel_size[self.class_names[i]]
            voxel_size = torch.tensor(cluster_vsize, device=points.device)
            pc_range = torch.tensor(self.point_cloud_range, device=points.device)
            voxel_config.append(torch.cat([pc_range[:3], voxel_size]))
        class_id = torch.cat(class_id)
        voxel_config = torch.stack(voxel_config)
        cluster_inds = voxel_spccl(points, batch_id, class_id, voxel_config, self.min_points, bsz)
        valid_mask = cluster_inds != -1
        cluster_inds = torch.stack([class_id, batch_id, cluster_inds], dim=-1)  #[N, 3], (cls_id, batch_idx, cluster_id)
        cluster_inds = cluster_inds[valid_mask]
        valid_mask_list = []
        for i in range(self.num_classes):
            valid_mask_list.append(valid_mask[class_id==i])

        return cluster_inds, valid_mask_list