import torch
import numpy as np
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops, Coord3DMode, Box3DMode
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import RandomFlip3D

import warnings
import mmcv
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

@PIPELINES.register_module()
class PadMultiImage:
    """Pad the image & masks & segmentation map.

    Args:
        size is list[(tuple, optional)]: Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Default: False.
        pad_val (dict, optional): A dict for padding value, the default
            value is `dict(img=0, masks=0, seg=255)`.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_to_square=False,
                 pad_val=dict(img=0, masks=-1, seg=255)):
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, float) or isinstance(pad_val, int):
            warnings.warn(
                'pad_val of float type is deprecated now, '
                f'please use pad_val=dict(img={pad_val}, '
                f'masks={pad_val}, seg=255) instead.', DeprecationWarning)
            pad_val = dict(img=pad_val, masks=pad_val, seg=255)
        assert isinstance(pad_val, dict)
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None and size_divisor is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)
        for key in results.get('img_fields', ['img']):
            multi_pad_img = []
            for i in range(len(results['img'])):
                if self.pad_to_square:
                    max_size = max(results[key][i].shape[:2])
                    self.size = (max_size, max_size)
                if self.size is not None:
                    padded_img = mmcv.impad(
                        results[key][i], shape=self.size[i], pad_val=pad_val)
                elif self.size_divisor is not None:
                    padded_img = mmcv.impad_to_multiple(
                        results[key][i], self.size_divisor, pad_val=pad_val)
                multi_pad_img.appen(padded_img)
            results[key] = np.array(multi_pad_img)
        results['pad_shape'] = [padded_img.shape for padded_img in multi_pad_img]
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            multi_pad_masks = []
            for i in range(len(results['img'])):
                pad_shape = results['pad_shape'][i][:2]
                multi_pad_masks.append(results[key][i].pad(pad_shape, pad_val=pad_val))
            results[key] = np.array(multi_pad_masks)

    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', 255)
        for key in results.get('seg_fields', []):
            multi_pad_seg = []
            for i in range(len(results['img'])):
                multi_pad_seg.append(mmcv.impad(results[key], shape=results['pad_shape'][:2], pad_val=pad_val))
            results[key]=np.array(multi_pad_seg)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val})'
        return repr_str

@PIPELINES.register_module()
class ResizeMultiViewImage:
    """Resize images & bbox & mask.
    """

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 keep_ratio=True,
                 bbox_clip_border=True,
                 backend='cv2',
                 interpolation='bilinear',):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        self.backend = backend
        self.multiscale_mode = multiscale_mode
        self.keep_ratio = keep_ratio
        # TODO: refactor the override option in Resize
        self.interpolation = interpolation
        self.bbox_clip_border = bbox_clip_border

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        for key in results.get('img_fields', ['img']):
            multi_view_img = []
            multi_scale_factor = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    img, scale_factor = mmcv.imrescale(
                        results['img'][i],
                        results['scale'],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results['img'][i],
                        results['scale'],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                multi_view_img.append(img)
                multi_scale_factor.append(np.array([w_scale, h_scale, w_scale, h_scale],
                        dtype=np.float32))
                
            results[key] = np.array(multi_view_img)
            scale_factor = np.array(multi_scale_factor)

            results['img_shape'] = [img.shape for img in multi_view_img]
            # in case that there is no padding
            results['pad_shape'] = [img.shape for img in multi_view_img]
            results['scale_factor'] = scale_factor
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        """Resize bounding boxes with ``results['scale_factor']``."""
        for key in results.get('bbox_fields', []):
            multi_gt_bboxes = []
            for i in range(len(results['img'])):
                bboxes = results[key][i] * results['scale_factor'][i]
                if self.bbox_clip_border:
                    img_shape = results['img_shape'][i]
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                multi_gt_bboxes.append(bboxes)
            results[key] = np.array(multi_gt_bboxes)

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            multi_gt_masks = []
            for i in range(len(results['img'])):
                if results[key] is None:
                    continue
                if self.keep_ratio:
                    results[key][i] = results[key][i].rescale(results['scale'][i])  # to contrast
                else:
                     multi_gt_masks.append(results[key][i].resize(results['img_shape'][i][:2]))
            results[key] = np.array(multi_gt_masks)

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            multi_seg = []
            for i in range(len(results['img'])):
                if self.keep_ratio:
                    gt_seg = mmcv.imrescale(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                else:
                    gt_seg = mmcv.imresize(
                        results[key][i],
                        results['scale'][i],
                        interpolation='nearest',
                        backend=self.backend)
                multi_seg.append(gt_seg)
            results[key] = np.array(multi_seg)

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)
        self._resize_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'multiscale_mode={self.multiscale_mode}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str
