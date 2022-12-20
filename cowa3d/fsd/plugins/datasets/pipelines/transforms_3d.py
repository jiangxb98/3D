import torch
import numpy as np
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import RandomFlip3D
from mmdet3d.core.bbox import Coord3DMode, Box3DMode


@PIPELINES.register_module()
class ObjectSampleWithSeg(object):
    def __init__(self, db_sampler, sample_2d=False, use_ground_plane=False):
        self.sampler_cfg = db_sampler
        self.sample_2d = sample_2d
        if 'type' not in db_sampler.keys():
            db_sampler['type'] = 'DataBaseSampler'
        self.db_sampler = build_from_cfg(db_sampler, OBJECTSAMPLERS)
        self.use_ground_plane = use_ground_plane

    @staticmethod
    def remove_points_in_boxes(points, boxes):
        masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
        points_mask = np.logical_not(masks.any(-1))
        points = points[points_mask]
        return points, points_mask

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)
        if self.sample_2d:
            img = input_dict['img']
            gt_bboxes_2d = input_dict['gt_bboxes']

        if self.use_ground_plane and 'plane' in input_dict['ann_info']:
            ground_plane = input_dict['ann_info']['plane']
            input_dict['plane'] = ground_plane
        # change to float for blending operation
        points = input_dict['points']
        sampled_dict = self.db_sampler.sample_all(input_dict, self.sample_2d)
        if sampled_dict is not None:
            sampled_gt_bboxes_3d = sampled_dict['gt_bboxes_3d']
            sampled_points = sampled_dict['points']
            sampled_gt_labels = sampled_dict['gt_labels_3d']
            sampled_pts_semantic_mask = sampled_dict.get('pts_semantic_mask', None)

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points, pts_mask = self.remove_points_in_boxes(
                                            points,
                                            sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])
            if pts_semantic_mask is not None:
                pts_semantic_mask = pts_semantic_mask[pts_mask]
                pts_semantic_mask = np.concatenate([sampled_pts_semantic_mask, 
                                                pts_semantic_mask], axis=0)

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points
        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f' sample_2d={self.sample_2d},'
        repr_str += f' data_root={self.sampler_cfg.data_root},'
        repr_str += f' info_path={self.sampler_cfg.info_path},'
        repr_str += f' rate={self.sampler_cfg.rate},'
        repr_str += f' prepare={self.sampler_cfg.prepare},'
        repr_str += f' classes={self.sampler_cfg.classes},'
        repr_str += f' sample_groups={self.sampler_cfg.sample_groups}'
        return repr_str


