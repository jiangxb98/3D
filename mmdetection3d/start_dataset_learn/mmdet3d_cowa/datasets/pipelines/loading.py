import mmcv
import numpy as np
from mmdet3d.core.points import get_points_type, BasePoints
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet.datasets.pipelines import LoadAnnotations
from io import BytesIO
from mmdet3d.datasets.builder import PIPELINES
import torch
from mmdet.core import BitmapMasks, PolygonMasks

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
        # load points from oss
        points = self._load_points(results, token)
        if self.remove_close:
            points = self._remove_close(points)
        points_class = get_points_type(self.coord_type)
        points = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints
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
                 random_choose=True,
                 test_mode=False):
        super(LoadSweeps, self).__init__(
            coord_type, remove_close, file_client_args)
        self.sweeps_num = sweeps_num
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        self.random_choose = random_choose

    def __call__(self, results):
        points = results['points']
        sweep_ts = points.tensor.new_zeros((len(points), 1))
        sweep_points_list = [points]
        sweep_ts_list = [sweep_ts]
        pts_info = results['pts_info']
        ts = pts_info['timestamp']
        dts = pts_info['timestamp_step']
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

        points = points.cat(sweep_points_list)
        sweep_ts = torch.cat(sweep_ts_list, dim=0)
        new_points = torch.cat((points.tensor, sweep_ts), dim=-1)
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
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']  # instance mask and semantic have the same path
        pts_instance_mask_loader = results['ann_info']['pts_instance_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        mask_bytes = self.file_client.get(pts_instance_mask_path)
        pts_instance_mask = pts_instance_mask_loader(results, mask_bytes, 'instance_id')
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
        pts_semantic_mask = pts_semantic_mask_loader(results, mask_bytes, 'semseg_cls')
        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results


@PIPELINES.register_module()
class LoadImages(object):
    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 file_client_args=dict(backend='disk'),
                 pad_shape=None,
                 ):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_shape = pad_shape

    def _load_img(self, results, token):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        pts_bytes = self.file_client.get(token)
        # not need loader, use mmcv.imfrombytes
        # img = results['img_info']['img_loader'](results, pts_bytes)
        img = mmcv.imfrombytes(pts_bytes, flag=self.color_type)
        # or tf.image.decode_jpeg(pts_bytes) both two fun have same size (1280,1920,3)
        return img        

    def __call__(self, results):
        results['filename'] = []
        results['img'] = []
        results['img_shape'] = []
        results['ori_shape'] = []
        results['img_fields'] = ['img']
        results['lidar2img'] = []
        results['pad_shape'] = self.pad_shape
        for i in range(len(results['img_info']['img_path_info'])):
            filename = results['img_info']['img_path_info'][i]['filename']
            img = self._load_img(results, filename).transpose((1,0,2))
            results['img'].append(img)
            results['filename'].append(filename)
            results['img_shape'].append(img.shape)
            results['ori_shape'].append(img.shape)
            results['lidar2img'].append(results['img_info']['img_path_info'][i]['lidar2img'])
        return results
    
    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str       


@PIPELINES.register_module()
class LoadAnnos(LoadAnnotations):
    def __init__(self, *arg, **kwargs):
        super(LoadAnnos, self).__init__(*arg, **kwargs)
    
    def _load_bboxes(self, results):
        ann_info = results['ann_info']
        results['gt_bboxes'] = ann_info['gt_bboxes'].copy()  # some imgaes only front camera has 2d gt bbox

        if self.denorm_bbox:
            bbox_num = results['gt_bboxes'].shape[0]
            if bbox_num != 0:
                h, w = results['img_shape'][:2]
                results['gt_bboxes'][:, 0::2] *= w
                results['gt_bboxes'][:, 1::2] *= h
        results['bbox_fields'].append('gt_bboxes')

        return results
    
    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['gt_labels'].copy()
        return results

    def _load_semantic_seg(self, results):
        pan_semantic_mask_path = results['ann_info']['pan_semantic_mask_path']
        pan_semantic_mask_loader = results['ann_info']['pan_semantic_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        results['gt_semantic_seg'] = []

        for pan_seg in pan_semantic_mask_path:
            mask_bytes = self.file_client.get(pan_seg)
            pan_semantic_mask = pan_semantic_mask_loader(results, mask_bytes, 'panseg_cls')
            results['gt_semantic_seg'].append(pan_semantic_mask)

        results['seg_fields'].append('gt_semantic_seg')
        return results

    def _load_masks(self, results):
        pan_instance_mask_path = results['ann_info']['pan_instance_mask_path']  # instance mask and semantic have the same path
        pan_instance_mask_loader = results['ann_info']['pan_instance_mask_loader']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        results['gt_masks'] = []

        for pan_seg in pan_instance_mask_path:   
            mask_bytes = self.file_client.get(pan_seg)
            pan_instance_mask = pan_instance_mask_loader(results, mask_bytes, 'panseg_instance_id')
            # 需要确认mask的shape是HxW还是WxH
            h, w = pan_instance_mask.shape[0], pan_instance_mask.shape[1]
            if self.poly2mask:
                gt_masks = BitmapMasks(self._poly2mask(pan_instance_mask, h, w), h, w)
            else:
                gt_masks = PolygonMasks(self.process_polygons(pan_instance_mask))

            results['gt_masks'].append(gt_masks)

        results['mask_fields'].append('gt_masks')
        return results