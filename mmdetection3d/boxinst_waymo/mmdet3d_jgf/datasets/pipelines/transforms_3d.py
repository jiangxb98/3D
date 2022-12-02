import random
import torch
import numpy as np
from mmcv.utils import build_from_cfg
from mmdet3d.core.bbox import box_np_ops, Coord3DMode, Box3DMode
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.builder import OBJECTSAMPLERS
from mmdet3d.datasets.pipelines import RandomFlip3D
from mmdet.core import BitmapMasks, PolygonMasks
import warnings
import mmcv
from PIL import Image
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
class FilterLabelImage:
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
            semantic_seg_3d = results['pts_semantic_mask'].squeeze()
            tmp_3d = np.zeros(semantic_seg_3d.shape)
            for i in range(len(self.seg_3d_class)):
                tmp_3d += np.where(semantic_seg_3d==self.seg_3d_class[i], i+1, 0)
            results['pts_semantic_seg'] = tmp_3d - 1
        if self.with_seg_3d:
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
                results['sample_img_id'] = sample_image_id
        else:
            sample_image_id = random.choice(range(5))
            results['sample_img_id'] = sample_image_id

        results['img'] = results['img'][sample_image_id]
        results['img_shape'] = results['img_shape'][sample_image_id]
        results['ori_shape'] = results['ori_shape'][sample_image_id]
        results['pad_shape'] = results['pad_shape'][sample_image_id]
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
class FilterLidarPoint:
    """
    The corresponding point cloud is obtained 
    according to the sampled images idx
    """