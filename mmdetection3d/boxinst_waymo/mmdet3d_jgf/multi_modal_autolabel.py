# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from os import path as osp

import mmcv
import torch
from mmcv.ops import Voxelization
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet.core import multi_apply, bbox2result

from mmdet3d.models import builder
from mmdet3d.models.builder import DETECTORS
from mmdet3d.models.detectors.base import Base3DDetector

from cowa3d_common.ops.ccl.ccl_utils import spccl, voxel_spccl, voxelized_sampling, sample

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

@DETECTORS.register_module()
class MultiModalAutoLabel(Base3DDetector):
    """Base class of Multi-modality autolabel."""

    def __init__(self,
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
                sweeps_num=1):
        super(MultiModalAutoLabel, self).__init__(init_cfg=init_cfg)

        # FSD
        if pts_segmentor is not None:  # 
            self.pts_segmentor = builder.build_detector(pts_segmentor)
        
        self.backbone = builder.build_backbone(pts_backbone)  # 
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
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
            self.num_classes = self.pts_bbox_head.num_classes

        self.roi_head = pts_roi_head  # 
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
    
        # 这里也是个全局的配置
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
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
        return img_feats

    def extract_pts_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""
        # If no points backbone return None
        if not self.with_pts_backbone:
            return None
        if not self.with_pts_bbox:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                img_feats, img_metas)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch

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
                      gt_bboxes_ignore=None):
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
        img_feats, pts_feats = self.extract_feat(points, img=img, img_metas=img_metas)

        losses = dict()

        # 点云补全
        if self.with_middle_encoder_pts:
            encoder_points = self.encoder_pts(points, img, img_metas, gt_bboxes, gt_labels)
            loss_depth = self.middle_encoder_pts.loss(**encoder_points)
            losses.update(loss_depth)
            enriched_points = encoder_points['enriched_foreground_logits']
            
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)

        if img_feats:
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
        return losses

    def forward_pts_train(self,
                        pts_feats,
                        gt_bboxes_3d,
                        gt_labels_3d,
                        img_metas,
                        gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

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

    def simple_test_img(self, x, img_metas, proposals=None, rescale=False):
        feat = x
        outputs = self.img_bbox_head.simple_test(
            feat, self.img_mask_head.param_conv, img_metas, rescale=rescale)
        det_bboxes, det_labels, det_params, det_coors, det_level_inds = zip(*outputs)
        bbox_results = [
            bbox2result(det_bbox, det_label, self.img_bbox_head.num_classes)
            for det_bbox, det_label in zip(det_bboxes, det_labels)
        ]

        mask_feat = self.img_mask_branch(feat)
        mask_results = self.img_mask_head.simple_test(
            mask_feat,
            det_labels,
            det_params,
            det_coors,
            det_level_inds,
            img_metas,
            self.img_bbox_head.num_classes,
            rescale=rescale)
        return list(zip(bbox_results, mask_results))

    def simple_test_rpn(self, x, img_metas, rpn_test_cfg):
        """RPN test function."""
        rpn_outs = self.img_rpn_head(x)
        proposal_inputs = rpn_outs + (img_metas, rpn_test_cfg)
        proposal_list = self.img_rpn_head.get_bboxes(*proposal_inputs)
        return proposal_list

    def simple_test_pts(self, x, img_metas, rescale=False):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x)
        bbox_list = self.pts_bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    # Base3DDetector的forward_test()内进入simple_test函数
    def simple_test(self, points, img_metas, img=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if img_feats and self.with_img_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale) # bbox_img, mask_img
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    def aug_test(self, points, img_metas, imgs=None, rescale=False):
        """Test function with augmentaiton."""
        img_feats, pts_feats = self.extract_feats(points, img_metas, imgs)

        bbox_list = dict()
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.aug_test_pts(pts_feats, img_metas, rescale)
            bbox_list.update(pts_bbox=bbox_pts)
        return [bbox_list]

    def extract_feats(self, points, img_metas, imgs=None):
        """Extract point and image features of multiple samples."""
        if imgs is None:
            imgs = [None] * len(img_metas)
        img_feats, pts_feats = multi_apply(self.extract_feat, points, imgs,
                                           img_metas)
        return img_feats, pts_feats

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

    def encoder_pts(self, points, img, img_metas, gt_bboxes, gt_labels):
        pts_dict = self.middle_encoder_pts(points, img, img_metas, gt_bboxes, gt_labels)
        return pts_dict