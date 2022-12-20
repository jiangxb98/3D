import mmcv
import numpy as np
from mmdet3d.core.points import get_points_type, BasePoints
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.datasets.builder import PIPELINES
import torch


@PIPELINES.register_module()
class LoadPoints(object):
    def __init__(self,
                 coord_type,
                 remove_close=False,
                 file_client_args=dict(backend='disk')):
        self.coord_type = coord_type
        self.remove_close = remove_close
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def _remove_close(self, points, radius=1.0):
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        not_close = np.linalg.norm(
            points_numpy[:, :2], ord=2, axis=-1) >= radius
        return points[not_close]

    def _load_points(self, results, token):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(token)
        points = results['pts_info']['pts_loader'](results, pts_bytes)
        return points

    def __call__(self, results):
        token = results['pts_info']['path']
        points = self._load_points(results, token)
        if self.remove_close:
            points = self._remove_close(points)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'coord_type={self.coord_type}, '
        repr_str += f'remove_close={self.remove_close}, '
        repr_str += f'file_client_args={self.file_client_args})'
        return repr_str


@PIPELINES.register_module()
class LoadSweeps(LoadPoints):
    def __init__(self,
                 sweeps_num,
                 coord_type,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 random_choose=False,
                 test_mode=False,
                 with_sweep_ind=False,
                 sample_ratio=1):
        super(LoadSweeps, self).__init__(
            coord_type, remove_close, file_client_args)
        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.random_choose = random_choose
        self.with_sweep_ind = with_sweep_ind
        self.sample_ratio = sample_ratio

    def __call__(self, results):
        points = results['points']
        sweep_ts = points.tensor.new_zeros((len(points), 1))
        sweep_points_list = [points]
        sweep_ts_list = [sweep_ts]
        pts_info = results['pts_info']
        ts = pts_info['timestamp']
        dts = pts_info['timestamp_step']
        sweep_ind_list = [points.tensor.new_zeros((len(points), 1))]
        if self.pad_empty_sweeps and len(pts_info['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    points_remove_close = self._remove_close(points)
                    sweep_ts_remove_close = points.tensor.new_zeros(
                        (len(points_remove_close), 1))
                    sweep_points_list.append(points_remove_close)
                    sweep_ts_list.append(sweep_ts_remove_close)
                else:
                    sweep_points_list.append(points)
                    sweep_ts_list.append(sweep_ts)
                sweep_ind_list.append((i + 1) * points.tensor.new_ones(
                    (len(sweep_points_list[-1]), 1)))
        else:
            if len(pts_info['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(pts_info['sweeps']))
            elif self.test_mode or (not self.random_choose):
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(pts_info['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = pts_info['sweeps'][idx]
                points_sweep = self._load_points(results, sweep['path'])
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp']
                rel_pose = sweep['rel_pose']
                points_sweep[:, :3] = points_sweep[:, :3] @ rel_pose[:3, :3].T
                points_sweep[:, :3] += rel_pose[:3, 3][None, :]
                points_sweep_ts = points.tensor.new_full(
                    (len(points_sweep), 1), (ts - sweep_ts) * dts)
                sweep_ts_list.append(points_sweep_ts)
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)
                sweep_ind_list.append((idx + 1) * points.tensor.new_ones(
                    (len(points_sweep), 1)))
            results['sweep_choices'] = choices

        # sample sweep frames
        if self.sample_ratio < 1:
            # sample_ind_list = []
            for i in range(1, len(sweep_points_list)):
                num_points = len(sweep_points_list[i])
                indices = torch.randperm(num_points)[:int(num_points * self.sample_ratio)]
                # sample_ind_list.append(indices)
                sweep_points_list[i] = sweep_points_list[i][indices]
                sweep_ts_list[i] = sweep_ts_list[i][indices]
                if self.with_sweep_ind:
                    sweep_ind_list[i] = sweep_ind_list[i][indices]


        points = points.cat(sweep_points_list)
        sweep_ts = torch.cat(sweep_ts_list, dim=0)
        new_points = torch.cat((points.tensor, sweep_ts), dim=-1)
        if self.with_sweep_ind:
            sweep_ind = torch.cat(sweep_ind_list, dim=0)
            new_points = torch.cat((new_points, sweep_ind), dim=-1)
        points = type(points)(new_points, points_dim=new_points.shape[-1],
                              attribute_dims=points.attribute_dims)
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class LoadAnnos3D(LoadAnnotations3D):
    def __init__(self,
                 *args,
                 **kwargs):
        super(LoadAnnos3D, self).__init__(*args, **kwargs)

    def _load_masks_3d(self, results):
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']
        pts_instance_mask_loader = results['ann_info'][
            'pts_instance_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mask_bytes = self.file_client.get(pts_instance_mask_path)
        pts_instance_mask = pts_instance_mask_loader(results, mask_bytes)
        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']
        pts_semantic_mask_loader = results['ann_info'][
            'pts_semantic_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mask_bytes = self.file_client.get(pts_semantic_mask_path)
        pts_semantic_mask = pts_semantic_mask_loader(results, mask_bytes)
        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results


@PIPELINES.register_module()
class LoadSegAnnos3D(LoadAnnotations3D):
    def __init__(self,
                 *args,
                 **kwargs):
        super(LoadSegAnnos3D, self).__init__(*args, **kwargs)

    def _load_masks_3d(self, results):
        pts_instance_mask_paths = results['ann_info']['pts_instance_mask_path']
        pts_instance_masks = []
        for pts_instance_mask_path in pts_instance_mask_paths:
            pts_instance_mask_loader = results['ann_info'][
                'pts_instance_mask_loader']

            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = pts_instance_mask_loader(results, mask_bytes, pts_instance_mask_path)
            pts_instance_masks.append(pts_instance_mask)

        results['pts_instance_mask'] = np.concatenate(pts_instance_masks, axis=0)
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_paths = results['ann_info']['pts_semantic_mask_path']
        pts_semantic_masks = []
        for pts_semantic_mask_path in pts_semantic_mask_paths:
            pts_semantic_mask_loader = results['ann_info'][
                'pts_semantic_mask_loader']

            if self.file_client is None:
                self.file_client = mmcv.FileClient(**self.file_client_args)
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            pts_semantic_mask = pts_semantic_mask_loader(results, mask_bytes, pts_semantic_mask_path)
            pts_semantic_masks.append(pts_semantic_mask)

        results['pts_semantic_mask'] = np.concatenate(pts_semantic_masks, axis=0)
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results


@PIPELINES.register_module()
class PointsRadiusRangeFilter(object):
    def __init__(self, point_radius_range):
        self.point_radius_range = point_radius_range

    def __call__(self, input_dict):
        points = input_dict['points']
        points_radius = points.tensor[:, :2].norm(p=2, dim=-1)
        points_mask = points_radius > min(self.point_radius_range)
        points_mask = points_mask.logical_and(
            points_radius < max(self.point_radius_range))
        clean_points = points[points_mask]
        input_dict['points'] = clean_points
        points_mask = points_mask.numpy()

        pts_instance_mask = input_dict.get('pts_instance_mask', None)
        pts_semantic_mask = input_dict.get('pts_semantic_mask', None)

        if pts_instance_mask is not None:
            input_dict['pts_instance_mask'] = pts_instance_mask[points_mask]

        if pts_semantic_mask is not None:
            input_dict['pts_semantic_mask'] = pts_semantic_mask[points_mask]

        return input_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(point_radius_range={self.point_radius_range})'
        return repr_str
