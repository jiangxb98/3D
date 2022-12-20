import torch
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.core.bbox import Coord3DMode, Box3DMode
from mmcv.utils import build_from_cfg
import numpy as np
from mmdet3d.datasets.pipelines.transforms_3d import (RandomFlip3D,
                                                      ObjectRangeFilter)
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)


@PIPELINES.register_module(force=True)
class ObjectSample(object):
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
        points = points[np.logical_not(masks.any(-1))]
        return points

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
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

            gt_labels_3d = np.concatenate([gt_labels_3d, sampled_gt_labels],
                                          axis=0)
            gt_bboxes_3d = gt_bboxes_3d.new_box(
                np.concatenate(
                    [gt_bboxes_3d.tensor.numpy(), sampled_gt_bboxes_3d]))

            points = self.remove_points_in_boxes(points, sampled_gt_bboxes_3d)
            # check the points dimension
            points = points.cat([sampled_points, points])

            if self.sample_2d:
                sampled_gt_bboxes_2d = sampled_dict['gt_bboxes_2d']
                gt_bboxes_2d = np.concatenate(
                    [gt_bboxes_2d, sampled_gt_bboxes_2d]).astype(np.float32)

                input_dict['gt_bboxes'] = gt_bboxes_2d
                input_dict['img'] = sampled_dict['img']

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d.astype(np.int64)
        input_dict['points'] = points

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


@PIPELINES.register_module()
class LabelIDMap:
    def __init__(self, map):
        self.map = map
        olds, news = [], []
        for old, new in map:
            assert old not in olds
            olds.append(old)
            if new not in news:
                news.append(new)
        id_map = []
        for idx, (old, new) in enumerate(self.map):
            id_map.append((idx, news.index(new)))
        self.id_map = id_map

    def __call__(self, input_dict):
        for old, new in self.id_map:
            input_dict['gt_labels_3d'][input_dict['gt_labels_3d'] == old] = new

        if 'gt_names_3d' in input_dict:
            for old, new in self.map:
                input_dict['gt_names_3d'][
                    input_dict['gt_names_3d'] == old] = new

        if 'gt_names' in input_dict:
            for old, new in self.map:
                input_dict['gt_names'][input_dict['gt_names'] == old] = new

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(map={self.map})'
        return repr_str


@PIPELINES.register_module()
class FilterBoxWithMinimumPointsCount:
    def __init__(self, num_points=1):
        self.num_points = num_points

    def __call__(self, input_dict):
        points = input_dict['points'].convert_to(Coord3DMode.LIDAR)
        gt_boxes_lidar = input_dict['gt_bboxes_3d'].convert_to(Box3DMode.LIDAR)
        points_xyz = points.coord.numpy()
        gt_boxes_lidar = gt_boxes_lidar.tensor[:, :7].numpy()
        indices = box_np_ops.points_in_rbbox(points_xyz, gt_boxes_lidar)
        mask = (np.count_nonzero(indices, axis=0) >= self.num_points)
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][
            torch.from_numpy(mask)]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][mask]
        return input_dict


@PIPELINES.register_module(name='RandomFlip3D', force=True)
class RandomFlip3DRev(RandomFlip3D):
    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 **kwargs):
        flip_ratio = 0. if not sync_2d else flip_ratio_bev_horizontal
        super(RandomFlip3D, self).__init__(flip_ratio=flip_ratio, **kwargs)
        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1


@PIPELINES.register_module(name='ObjectRangeFilter', force=True)
class ObjectRangeFilterRev(ObjectRangeFilter):
    def __call__(self, input_dict):
        if isinstance(input_dict['gt_bboxes_3d'],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]
        elif isinstance(input_dict['gt_bboxes_3d'], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d = input_dict['gt_bboxes_3d']
        gt_labels_3d = input_dict['gt_labels_3d']
        mask = gt_bboxes_3d.in_range_bev(bev_range)
        mask_np = mask.numpy().astype(np.bool)
        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask_np]

        # limit rad to [-pi, pi]
        gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)
        input_dict['gt_bboxes_3d'] = gt_bboxes_3d
        input_dict['gt_labels_3d'] = gt_labels_3d
        if 'gt_2dto3d' in input_dict:
            new_idx = np.where(mask_np, 1, 0).cumsum() - 1
            new_idx[~mask_np] = -1
            input_dict['gt_2dto3d'] = [new_idx[idx] for idx in
                                       input_dict['gt_2dto3d']]
        return input_dict
