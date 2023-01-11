import math
import random
from ...utils import ground_segmentation, calculate_ground
import torch
import numpy as np
from scipy.sparse.csgraph import connected_components  # CCL
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops, Coord3DMode, Box3DMode
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import RandomFlip3D
from mmdet.core import BitmapMasks, PolygonMasks
from mmdet3d.core.points import get_points_type, BasePoints
import warnings
import mmcv
from PIL import Image
from skimage.util.shape import view_as_windows
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

@PIPELINES.register_module(force=True)
class MyFilterBoxWithMinimumPointsCount:
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

# No Test
@PIPELINES.register_module()
class PadMultiViewImage:
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
                 pad_val=dict(img=0, masks=0, seg=-1)):
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
            for i in range(len(results[key])):
                if self.size[i] == results[key][i].shape[:2]:
                    continue
                if self.pad_to_square:
                    max_size = max(results[key][i].shape[:2])
                    self.size = (max_size, max_size)
                if self.size is not None:
                    # if only pad top. del: shape=, add parm: padding=(0,1280-886,0,0)
                    padded_img = mmcv.impad(
                        results[key][i], shape=self.size[i][:2], pad_val=pad_val)
                elif self.size_divisor is not None:
                    padded_img = mmcv.impad_to_multiple(
                        results[key][i], self.size_divisor, pad_val=pad_val)
                # Image.fromarray(np.uint8(results[key][i])).save("ori_img_{}.jpeg".format(i))
                # Image.fromarray(np.uint8(padded_img)).save("pad_img_{}.jpeg".format(i))
                results[key][i] = padded_img
        results['pad_shape'] = [padded_img.shape for padded_img in results[key]]
        results['img_shape'] = results['pad_shape']
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor


    def _pad_masks(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        pad_val = self.pad_val.get('masks', 0)
        for key in results.get('mask_fields', []):
            for i in range(len(results['img'])):
                if self.size[i] == results[key][i].masks.shape[1:3]:
                    continue
                pad_shape = results['pad_shape'][i][:2]
                results[key][i] = results[key][i].pad(pad_shape, pad_val=pad_val)


    def _pad_seg(self, results):
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        pad_val = self.pad_val.get('seg', -1)
        for key in results.get('seg_fields', []):
            for i in range(len(results['img'])):
                if self.size[i] == results[key][i].shape:
                    continue
                results[key][i] = mmcv.impad(results[key][i], shape=results['pad_shape'][i][:2], pad_val=pad_val)

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
    
    Args:
        results['img'] is a list or array, to resize fixed size 
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
                        results['scale'][i],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                    # the w_scale and h_scale has minor difference
                    # a real fix should be done in the mmcv.imrescale in the future
                    new_h, new_w = img.shape[:2]
                    h, w = results[key][i].shape[:2]
                    w_scale = new_w / w
                    h_scale = new_h / h
                else:
                    img, w_scale, h_scale = mmcv.imresize(
                        results['img'][i],
                        results['scale'][i],
                        return_scale=True,
                        interpolation=self.interpolation,
                        backend=self.backend)
                multi_view_img.append(img)
                multi_scale_factor.append(np.array([w_scale, h_scale, w_scale, h_scale],
                        dtype=np.float32))
                
            results[key] = multi_view_img
            scale_factor = multi_scale_factor

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
                # if None
                if (results[key][i] == np.array([-1,-1,-1,-1])).all():
                    multi_gt_bboxes.append(np.array([-1,-1,-1,-1]))
                # if  Not None
                else:
                    bboxes = results[key][i] * results['scale_factor'][i]
                    if self.bbox_clip_border:
                        img_shape = results['img_shape'][i]
                        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                    multi_gt_bboxes.append(bboxes)

            results[key] = multi_gt_bboxes

    def _resize_masks(self, results):
        """Resize masks with ``results['scale']``"""
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            for i in range(len(results['img'])):
                # !!! both rescale and resize both are BitmapMasks(), not a np.array
                if self.keep_ratio:
                    # to contrast origin of transforms
                    results[key][i] = results[key][i].rescale(results['scale'][i])
                else:
                    new_shape=results['img_shape'][i][:2]
                    results[key][i] = results[key][i].resize(new_shape)

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
                # if results['scale'][i][::-1] == results[key][i].shape, the gt_seg=results[key][i]
                multi_seg.append(gt_seg)

            results[key] = multi_seg

    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor', \
                'keep_ratio' keys are added into result dict.
        """
        results['scale'] = self.img_scale
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

# No Test
@PIPELINES.register_module()
class MultiViewCrop:
    def __init__(self,
                crop_size=(886, 1920),
                crop_type='absolute',
                bbox_clip_border=True):
        self.crop_size = crop_size
        self.bbox_clip_border = bbox_clip_border
        self.crop_type = crop_type
        # The key correspondence from bboxes to labels and masks.
        self.bbox2label = {
            'gt_bboxes': 'gt_labels',
        }
        self.bbox2mask = {
            'gt_bboxes': 'gt_masks',
        }
    
    def _crop_data(self, results, crop_size):
        assert crop_size[0] > 0 and crop_size[1] > 0
        for key in results.get('img_fields', ['img']):
            imgs = results[key] # mutli images = 5
            img_crop = []
            offset_wh = []
            bbox_crop = []
            # mask_crop = []
            sem_crop = []
            for i in range(len(results['img'])):

                offset_h = max(imgs[i].shape[0] - crop_size[0], 0)  # 480
                offset_w = max(imgs[i].shape[1] - crop_size[1], 0)  # 0
                offset_wh.append(np.array([offset_w, offset_h, offset_w, offset_h],
                                   dtype=np.float32))
                # if no offset
                if offset_h == 0 and offset_w == 0:
                    img_crop.append(img)
                    continue
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

                # crop the image
                img = imgs[i][crop_y1:crop_y2, crop_x1:crop_x2, ...]
                img_crop.append(img)
            results[key] = img_crop
        results['img_shape'] = [crop_size for i in range(len(results['img']))]

        # crop bboxes accordingly and clip to the image boundary
        for key in results.get('bbox_fields', []):
    
            offset_h = offset_wh[i][1]
            offset_w = offset_wh[i][0]
            crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
            crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

            # e.g. gt_bboxes
            for i in range(len(results['img'])):
                img_shape = results['img_shape'][i]
                # if gt_bboxes is None, skip this image, continue
                if ((results[key][i] == np.array([-1,-1,-1,-1])).all() and key == 'gt_bboxes'):
                    bbox_crop.append(np.array([-1,-1,-1,-1]))
                    continue
                bboxes = results[key][i] - offset_wh[i]
                # if no offset continue
                if offset_wh[i][0] == 0 and offset_wh[i][1] == 0:
                    bbox_crop.append(bboxes)
                    continue
                if self.bbox_clip_border:
                    bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1])
                    bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0])
                # If the crop does not contain any gt-bbox area, skip continue
                valid_inds = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])
                if key == 'gt_bboxes' and not valid_inds.any():
                    bbox_crop.append(np.array([-1,-1,-1,-1]))

                # label fields. e.g. gt_labels and gt_labels_ignore
                label_key = self.bbox2label.get(key)
                if label_key in results:
                    if not valid_inds.any():
                        results[label_key][i] = np.array([-1])
                    else:
                        results[label_key][i] = results[label_key][i][valid_inds]
            
            results[key] = bbox_crop

        # crop semantic seg
        for key in results.get('seg_fields', []):
            for i in range(len(results['img'])):
                offset_h = offset_wh[i][1]
                offset_w = offset_wh[i][0]
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
                if offset_h == 0 and offset_w == 0:
                    sem_crop.append(results[key][i])
                    continue
                sem_crop.append(results[key][i][crop_y1:crop_y2, crop_x1:crop_x2])

            results[key] = np.stack(sem_crop, axis=0)

        # mask fields
        for key in results.get('mask_fields',[]):
            for i in range(len(results['img'])):
                offset_h = offset_wh[i][1]
                offset_w = offset_wh[i][0]
                crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
                crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]
                if offset_h == 0 and offset_w == 0:
                    continue
                results[key][i] = results[key][i].crop(np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
        return results

    def __call__(self, results):
        results = self._crop_data(results, self.crop_size)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiViewImage:
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """
    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.get('img_fields', ['img']):
            for i in range(len(results[key])):
                results[key][i] = mmcv.imnormalize(results[key][i], self.mean, self.std,
                                                self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class FilterLabelByImage:
    '''filter pan and segment label of 2d and 3d

    - filter_class_name: the class to use
    - with_mask: True or False
    - with_seg: True or False
    - with_mask_3d: True or False
    - with_seg_3d: True or False
    '''
    def __init__(self,
                filter_class_name=None,
                with_mask=True,
                with_seg=True,
                with_mask_3d=True,
                with_seg_3d=True,
                *arg, **kwargs):
        self.filter_class_name = filter_class_name
        self.with_mask = with_mask
        self.with_seg = with_seg
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        # Car=2,Pedestrian=9,Cyclist=10
        self.seg_calss = [2,9,10]
        # Car=1,Pedestrian=7,Bicyclist=6
        self.seg_3d_class = [1,7,6]
    
    def __call__(self, results):
        '''return the new semantic label, and instance only filtered
        - label: Car=0, Pedestrian=1, Cyclist=2
        - seg -1 is uncorrelated, instance only filtered
        '''
        img_num = len(results['img'])
        for i in range(img_num):
            if self.with_seg:
                semantic_seg = results['gt_semantic_seg'][i].squeeze()
                tmp = np.zeros(semantic_seg.shape)
                for j in range(len(self.seg_calss)):
                    tmp += np.where(semantic_seg==self.seg_calss[j], j+1, 0)
                results['gt_semantic_seg'][i] = tmp - 1
            if self.with_mask:
                seg_ind = (tmp-1 >= 0)  # True or False array
                new_mask = results['gt_masks'][i].masks * seg_ind  # new_mask shape=(1,1280,1920)
                results['gt_masks'][i] = BitmapMasks(new_mask, new_mask.shape[1], new_mask.shape[2])
        
        if self.with_mask_3d:
            if 'pts_semantic_mask' in results.keys():
                semantic_seg_3d = results['pts_semantic_mask'].squeeze()
                tmp_3d = np.zeros(semantic_seg_3d.shape)
                for i in range(len(self.seg_3d_class)):
                    tmp_3d += np.where(semantic_seg_3d==self.seg_3d_class[i], i+1, 0)
                results['pts_semantic_mask'] = tmp_3d - 1
        if self.with_seg_3d:
            if 'pts_instance_mask' in results.keys():
                seg_ind_3d = (tmp_3d-1 >= 0)
                results['pts_instance_mask'] = results['pts_instance_mask'] * seg_ind_3d

        return results


@PIPELINES.register_module()
class SampleFrameImage:
    def __init__(self,
                sample = 'random',
                guide = 'gt_bboxes',
                training = True):
        self.sample = sample
        self.guide = guide
        self.training = training
    
    def _random_sample(self, results):
        ''''each frame random select a image which has 2d gt_bboxes
        '''
        results['sample_img_id'] = []
        if self.training:
            if self.guide == 'gt_bboxes':
                for i in range(len(results['gt_labels'])):
                    gt_label = results['gt_labels'][i]
                    if (gt_label==-1).all():
                        continue
                    else:
                        results['sample_img_id'].append(i)
                sample_image_id = random.choice(results['sample_img_id'])
                sample_image_id = 0  # test阶段，均采样第一张图片
                results['sample_img_id'] = sample_image_id
        else:
            sample_image_id = random.choice(range(5))
            results['sample_img_id'] = sample_image_id

        results['img'] = results['img'][sample_image_id]
        results['img_shape'] = results['img_shape'][sample_image_id]
        results['ori_shape'] = results['ori_shape'][sample_image_id]
        results['pad_shape'] = results['pad_shape'][sample_image_id]
        results['scale'] = results['scale'][sample_image_id]
        results['scale_factor'] = results['scale_factor'][sample_image_id]
        results['lidar2img'] = results['lidar2img'][sample_image_id]
        results['pad_fixed_size'] = results['pad_fixed_size'][sample_image_id]
        if 'gt_labels' in results.keys():
            results['gt_labels'] = results['gt_labels'][sample_image_id]
        if 'gt_bboxes' in results.keys():
            results['gt_bboxes'] = results['gt_bboxes'][sample_image_id]
        if 'gt_masks' in results.keys():
            results['gt_masks'] = results['gt_masks'][sample_image_id]
        if 'gt_semantic_seg' in results.keys():
            results['gt_semantic_seg'] = results['gt_semantic_seg'][sample_image_id]
        results.update(dict(img_sample='random'))


        return results
        
    def _resample(self, results):
        pass

    def __call__(self, results):

        if self.sample == 'random':
            results = self._random_sample(results)
        elif self.sample == 'resample':
            results = self._resample(results)

        return results

@PIPELINES.register_module()
class RemoveGroundPoints:

    def __init__(self, coord_type):
        self.coord_type = coord_type

    def __call__(self, results):
        points = results['points'].tensor.numpy()
        # RANSAC remove ground points
        ground_points, segment_points = ground_segmentation(points)
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(segment_points, points_dim=segment_points.shape[-1])  # 实例化，LiDARPoints
        return results      

@PIPELINES.register_module()
class FilterPointsByImage:
    """
    The project point cloud is obtained by the Image idx
    """
    def __init__(self, coord_type, kernel_size=3, threshold_depth=0.5):
        self.coord_type = coord_type
        self.kernel_size = kernel_size
        self.threshold_depth = threshold_depth

    def _filter_points(self, results):
        points = results['points'].tensor.numpy()
        sample_img_id = results['sample_img_id']
        mask = (points[:,6]==sample_img_id) | (points[:,7]==sample_img_id)
        in_img_points = points[mask]
        
        results['ori_points'] = results['points']
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(in_img_points, points_dim=in_img_points.shape[-1])  # 实例化，LiDARPoints

        return results
    
    def grad_guide_filter(self, results):
        points = results['points'].tensor.numpy()
        sample_img_id = results['sample_img_id']
        scale = results['scale_factor'][0]
        # 1. 获得深度距离图
        h, w, _ = results['img_shape']
        depth_img = np.zeros((h, w))
        img2points = np.ones((h, w)) * -1
        points_dist = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
        for i, point in enumerate(points):
            if point[6] == sample_img_id:
                x_0 = point[8] * scale
                y_0 = point[10] * scale
                depth_img[int(y_0), int(x_0)] = points_dist[i]
                img2points[int(y_0), int(x_0)] = i

            if point[7] == sample_img_id:
                x_1 = point[9] * scale
                y_1 = point[11] * scale
                depth_img[int(y_1), int(x_1)] = points_dist[i]
                img2points[int(y_1), int(x_1)] = i
        # 2.滑动窗口过滤噪声点
        pad_width = self.kernel_size // 2  # 给深度图四周补0
        pad_depth_img = np.pad(depth_img, pad_width = pad_width, mode = 'constant', constant_values = 0)
        # 划分窗口
        depth_img_windows = view_as_windows(pad_depth_img, (self.kernel_size, self.kernel_size), 1)
        depth_img_windows_999 = np.copy(depth_img_windows)
        depth_img_windows_mask = depth_img_windows == 0
        depth_img_windows_999[depth_img_windows_mask] = 999
        depth_img_mask = depth_img != 0
        # 计算梯度关系（相对距离关系）
        relative_dis = (depth_img - np.min(depth_img_windows_999, axis=(2,3))) / np.max(depth_img_windows, axis=(2,3))
        relative_dis[~depth_img_mask] = 999  # 999表示空的像素位置
        points_near_mask = relative_dis < self.threshold_depth
        # points_far_mask = (relative_dis >= self.threshold_depth) & (relative_dis != 999)
        # 通过img2points可以获得对应depth_img每个像素点对应的点云的索引
        near_points_indx = img2points[points_near_mask].astype(np.int)
        # far_points_indx = img2points[points_far_mask].astype(np.int)
        points = points[near_points_indx]
        points_class = get_points_type(self.coord_type)
        results['points'] = points_class(points, points_dim=points.shape[-1])  # 实例化，LiDARPoints
        return results
        
    def __call__(self, results):
        results = self._filter_points(results)
        results = self.grad_guide_filter(results)
        return results

@PIPELINES.register_module()
class GetOrientation:
    """
    得到2D框内对象的朝向角
    """
    def __init__(self, gt_box_type=1, sample_roi_points=100, th_dx=4, dist=0.6, use_geomtry_loss=False):
        self.gt_box_type = gt_box_type
        self.sample_roi_points = sample_roi_points
        self.th_dx = th_dx  # 阈值
        self.dist = dist
        self.use_geomtry_loss = use_geomtry_loss

    def find_connected_componets_single_batch(self, points, dist):

        this_points = points
        dist_mat = this_points[:, None, :2] - this_points[None, :, :2] # only care about xy
        dist_mat = (dist_mat ** 2).sum(2) ** 0.5
        adj_mat = dist_mat < dist
        adj_mat = adj_mat
        c_inds = connected_components(adj_mat, directed=False)[1]

        return c_inds

    def get_in_2d_box_points(self, results):
        points = results['points'].tensor.numpy()
        gt_bboxes = results['gt_bboxes']
        labels = results['gt_labels']
        sample_img_id = results['sample_img_id']
        scale = results['scale_factor'][0]
        roi_batch_points = []
        
        out_gt_bboxes = []
        out_gt_labels = []

        for i, gt_bbox in enumerate(gt_bboxes):
            # 2d box是resize过得, 需要还原
            gt_bbox = gt_bbox / scale
            # if 0 cam 8,10（列，行）
            gt_mask_0 = (((points[:, 8] > gt_bbox[0]) & (points[:, 8] < gt_bbox[2])) &
                        ((points[:, 10] > gt_bbox[1]) & (points[:, 10] < gt_bbox[3])) &
                        (points[:, 6] == sample_img_id))
            # if 1 cam 9,11
            gt_mask_1 = (((points[:, 9] > gt_bbox[0]) & (points[:, 9] < gt_bbox[2])) &
                        ((points[:, 11] > gt_bbox[1]) & (points[:, 11] < gt_bbox[3])) &
                        (points[:, 7] == sample_img_id))

            gt_mask = gt_mask_0 | gt_mask_1
            in_box_points = points[gt_mask]
            # 如果2d框里没有点，那么就过滤掉对应的2d box和label
            if len(in_box_points) == 0:
                continue
            out_gt_bboxes.append(gt_bbox * scale)
            out_gt_labels.append(labels[i])
            roi_batch_points.append(in_box_points)  # 12
        assert len(roi_batch_points) == len(out_gt_bboxes)
        results['gt_labels'] = np.array(out_gt_labels)
        results['gt_bboxes'] = np.array(out_gt_bboxes)
        return roi_batch_points, results
    
    def points_assign_by_bboxes(self, results, roi_points):
        """2D box内的点云存在遮挡,依照距离远近来分配点云到对应的2D box
        Input: roi_points(list) [ndarray,ndarray,……] length is no empty 2D boxes numbers
        """
        all_roi_min_dist = np.ones((len(roi_points))) * 999
        for i in range(len(roi_points)):
            single_roi_points = roi_points[i]
            single_roi_points_dist = np.sqrt(single_roi_points[:, 0]**2 + single_roi_points[:, 1]**2)
            all_roi_min_dist[i] = np.min(single_roi_points_dist).astype(np.float32)
        # 解决overlap问题
        bboxes = results['gt_bboxes']  # gt_box是resize过的
        bboxes_nums = len(bboxes)
        # 1. 查找所有盒子的碰撞关系
        # 相交矩阵
        all_overlap_mask = np.zeros((bboxes_nums, bboxes_nums, 5))  # 5=[overlap_flag, inter(x1,y1,x2,y2)]
        for i in range(bboxes_nums):
            for j in range(bboxes_nums):
                flag, inter_box = self.if_overlap(bboxes[i], bboxes[j])
                if flag:
                    all_overlap_mask[i][j][0] = 1
                    all_overlap_mask[i][j][1:5] = inter_box
                else:
                    all_overlap_mask[i][j][0] = 0
        # 2. 判断相交的box内的点属于哪个box
        all_ignore_indx = [[] for _ in range(bboxes_nums)]  # 记录碰撞关系
        remove_points = []  # 记录删除的点
        for i in range(bboxes_nums):
            overlap_mask = all_overlap_mask[i][:, 0]  # 1表示有碰撞，0表示未碰撞
            overlap_indx = np.where(overlap_mask==1)[0]
            # 这个放前面是为了避免计算自身的碰撞盒关系
            for j in range(len(overlap_indx)):
                all_ignore_indx[overlap_indx[j]].append(i)
            ignore_indx = all_ignore_indx[i]
            # 3.获取overlap的盒子区域内的点云，然后选择正确的盒子
            for j in range(len(overlap_indx)):
                if overlap_indx[j] not in ignore_indx:
                    # 3.1 获取相交区域
                    inter_box = all_overlap_mask[i][overlap_indx[j]][1:5]
                    # 3.2 找到两个盒子中距离车最近的距离，然后将相交区域的点赋给最近的盒子，去掉远的盒子的相交区域的点
                    tmp_dist, overlap_dist = all_roi_min_dist[i], all_roi_min_dist[overlap_indx[j]]
                    # 如果当前的盒子在前面，那么保留当前盒子的点云不动，删除碰撞盒内的相交点云
                    if tmp_dist < overlap_dist:
                        eq_mask = np.isin(roi_points[overlap_indx[j]][:,0:3], roi_points[i][:,0:3])  # 维度大小和前一个值相同 np.isin(a1, a2)也就是a1
                        sum_eq_mask = np.sum(eq_mask,axis=1) == 3  # 过滤掉不是三个坐标值都相等的点
                        remove_points.append(roi_points[overlap_indx[j]][sum_eq_mask])
                        roi_points[overlap_indx[j]] = roi_points[overlap_indx[j]][~sum_eq_mask]
                    # 如果是当前盒子在后面，删除当前盒子的点
                    else:
                        eq_mask = np.isin(roi_points[i][:,0:3], roi_points[overlap_indx[j]][:,0:3])
                        sum_eq_mask = np.sum(eq_mask, axis=1) == 3
                        remove_points.append(roi_points[i][sum_eq_mask])
                        roi_points[i] = roi_points[i][~sum_eq_mask]
        remove_points = np.concatenate(remove_points, axis=0)
        # 过滤掉空的roi_points
        out_roi_points = []
        out_bboxes = []
        out_labels = []
        for i in range(len(bboxes)):
            if len(roi_points[i]) != 0:
                out_bboxes.append(bboxes[i])
                out_labels.append(results['gt_labels'][i])
                out_roi_points.append(roi_points[i])
        results['gt_labels'] = np.array(out_labels)
        results['gt_bboxes'] = np.array(out_bboxes)
        results['ori_roi_points'] = out_roi_points  # 这个保存的是处理掉overlap问题后每个box内的点云
        # all_roi_points = np.concatenate(out_roi_points, axis=0)
        return out_roi_points, results

    def if_overlap(self, box1, box2):

        min_x1, min_y1, max_x1, max_y1 = box1[0], box1[1], box1[2], box1[3]
        min_x2, min_y2, max_x2, max_y2 = box2[0], box2[1], box2[2], box2[3]

        top_x, top_y = max(min_x1, min_x2), max(min_y1, min_y2)
        bot_x, bot_y = min(max_x1, max_x2), min(max_y1, max_y2)
        
        if bot_x >= top_x and bot_y >= top_y:
            return True, np.array([top_x, top_y, bot_x, bot_y])
        else:
            return False, np.array([0, 0, 0, 0])

    def get_orientation(self, roi_batch_points, results):
        
        RoI_points = roi_batch_points  # (N,12)
        gt_bboxes = results['gt_bboxes']
        assert len(roi_batch_points) == len(gt_bboxes) and len(gt_bboxes) == len(results['gt_labels'])

        batch_RoI_points = np.zeros((gt_bboxes.shape[0], self.sample_roi_points, 3), dtype=np.float32)
        batch_lidar_y_center = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)  
        batch_lidar_orient = np.zeros((gt_bboxes.shape[0],), dtype=np.float32)
        batch_lidar_density = np.zeros((gt_bboxes.shape[0], self.sample_roi_points), dtype=np.float32)
        
        for i in range(len(roi_batch_points)):
            # 聚类过滤
            c_inds = self.find_connected_componets_single_batch(RoI_points[i][:, 0:3], self.dist)
            set_c_inds = list(set(c_inds))
            c_ind = np.argmax([np.sum(c_inds == i) for i in set_c_inds])
            c_mask = c_inds == set_c_inds[c_ind]
            RoI_points[i] = RoI_points[i][c_mask]
            
            z_coor = RoI_points[i][:, 2]  # height val
            batch_lidar_y_center[i] = np.mean(z_coor)
            z_thesh = (np.max(z_coor) + np.min(z_coor)) / 2
            z_ind = RoI_points[i][:, 2] < z_thesh  # 这里没看懂，为什么靠下？

            z_ind_points = RoI_points[i][z_ind]
            if z_ind_points.shape[0] < 10:
                z_ind_points = RoI_points[i]

            rand_ind = np.random.randint(0, z_ind_points.shape[0], 100)
            depth_points_sample = z_ind_points[rand_ind]
            batch_RoI_points[i] = depth_points_sample[:, 0:3]
            depth_points_np_xy = depth_points_sample[:, [0, 1]]  # 获得当前2d框内的点云的xy坐标

            '''orient'''
            orient_set = [(i[0] - j[0]) / - (i[1] - j[1]) for j in depth_points_np_xy  # 这里的y轴反向了一下，方便对照weakm3d得到的斜率k值
                          for i in depth_points_np_xy]  # 斜率，存在nan值，分母为0
            orient_sort = np.array(sorted(np.array(orient_set).reshape(-1)))
            orient_sort = np.arctan(orient_sort[~np.isnan(orient_sort)])  # 过滤掉nan值，然后得到角度 [-pi/2,pi/2]
            orient_sort_round = np.around(orient_sort, decimals=1)  # 对输入浮点数执行5舍6入，5做特殊处理 decimals保留1位小数
            set_orenit = list(set(orient_sort_round))  # 去重，得到直方图的bin  [-1.6, 1.6]
            try:
                ind = np.argmax([np.sum(orient_sort_round == i) for i in set_orenit])  # 得到直方图最高的点
                orient = set_orenit[ind]
                if orient < 0:  # 角度是钝角时，＜0，需要加上pi,变换到正方向
                    orient += np.pi
                
                # weakm3d提到车的行驶方向通常是45度到135度，但如果超过了阈值距离dx，那么就会再回到旧的距离，下面有写判断
                if orient > np.pi / 2 + np.pi * 3 / 8:
                    orient -= np.pi / 2
                if orient < np.pi / 8:
                    orient += np.pi / 2

                if np.max(RoI_points[i][:, 1]) - np.min(RoI_points[i][:, 1]) > self.th_dx and \
                        (orient >= np.pi / 8 and orient <= np.pi / 2 + np.pi * 3 / 8):
                    if orient < np.pi / 2:
                        orient += np.pi / 2
                    else:
                        orient -= np.pi / 2
                    # 这一步出来的是kitti下的yaw角范围是[0, pi]
                # 转到wamoy坐标系下，yaw [-pi/2, pi/2]
                orient = orient - np.pi/2
            except:
                orient = 0  # 如果np.argmax得不到值，就默认为沿x轴方向
            batch_lidar_orient[i] = orient

            '''density'''
            p_dis = np.array([(i[0] - depth_points_sample[:, 0]) ** 2 + (i[1] - depth_points_sample[:, 1]) ** 2
                                 for i in depth_points_sample])
            batch_lidar_density[i] = np.sum(p_dis < 0.04, axis=1)

        results['gt_yaw'] = batch_lidar_orient.astype(np.float32)
        results['roi_points'] = batch_RoI_points.astype(np.float32)
        results['ori_roi_points'] = roi_batch_points
        if self.use_geomtry_loss:
            results['lidar_density'] = batch_lidar_density.astype(np.float32)
            # results['y_center'] = batch_lidar_y_center.astype(np.float32)
        return results

    def __call__(self, results):
        if self.gt_box_type == 2:
            roi_points, results = self.get_in_2d_box_points(results)  # (gt_box的数量, 3)
            roi_points, results = self.points_assign_by_bboxes(results, roi_points)
            results = self.get_orientation(roi_points, results)
        return results

def plt_fun(img, points, out_bboxes, sample_idx):
    # points N,12
    # bboxes N,4
    import cv2

    img = img.astype('uint8')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for gt_bbox in out_bboxes:
        cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 0), 2)

    for point in points:
        if point[6] == sample_idx:
            x_0 = int(point[8]*0.5)
            y_0 = int(point[10]*0.5)
            cv2.circle(img, (x_0, y_0), 1, (0, 255, 0), 1)
        if point[7] == sample_idx:
            x_1 = int(point[9]*0.5)
            y_1 = int(point[11]*0.5)
            cv2.circle(img, (x_1, y_1), 1, (0, 255, 0), 1)

    cv2.imwrite('test_img_{}.jpeg'.format(sample_idx), img)