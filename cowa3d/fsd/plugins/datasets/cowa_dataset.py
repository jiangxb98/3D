
import os
import copy
import mmcv
import numpy as np

from io import BytesIO
from functools import partial
from terminaltables import AsciiTable
from collections import OrderedDict, defaultdict

from mmcv.utils import print_log
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.core.bbox import (get_box_type, LiDARInstance3DBoxes, Box3DMode)

from ..core.bbox import LiDARInstance3DBoxesVel
from cowa3d_common.core.evaluation import eval_map_flexible


@DATASETS.register_module()
class COWADatasetDetSeg(Custom3DDataset):
    DET_CLASSES = (
        'big_vehicle', 'pedestrian', 'vehicle', 'bicycle', 'huge_vehicle',
        'tricycle', 'cone', 'barrier', 'barrel')
    SEG_CLASSES = (
        'unlabeled', 'road-plane', 'curb', 'other-ground', 'terrain',
        'vegetation', 'pillars', 'framework', 'building', 'fence',
        'traffic-sign', 'other-structure', 'noise', 'road-users', 'road-block')

    def __init__(self,
                 det_info_path=None,
                 seg_info_path=None,
                 seg_clip_parts=None,
                 sensor_filter=[0],
                 sensor_signals=None,
                 pipeline=None,
                 modality=None,
                 det_classes=None,
                 seg_classes=None,
                 seg_map_labels=None,
                 box_type_3d='Lidar',
                 filter_empty_gt=False,
                 test_mode=False,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 det_datainfo_client_args=None,
                 seg_datainfo_client_args=None,
                 load_interval=1):
        super(Custom3DDataset, self).__init__()
        self.det_info_path = det_info_path
        self.seg_info_path = seg_info_path
        self.seg_clip_parts = seg_clip_parts
        self.sensor_filter = sensor_filter
        self.sensor_signals = sensor_signals
        self.det_datainfo_client_args = det_datainfo_client_args
        self.seg_datainfo_client_args = seg_datainfo_client_args
        self.modality = modality
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.DET_CLASSES = det_classes
        self.SEG_CLASSES = seg_classes
        self.CLASSES = self.DET_CLASSES
        # self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if seg_map_labels is not None:
            self.seg_map_labels = np.array(seg_map_labels, dtype=np.uint8)
        self.data_infos = self.load_annotations(self.det_info_path,
                                                self.seg_info_path)
        self.det_infos_reader = None
        # self.seg_infos_reader = None
        self.pcd_limit_range = pcd_limit_range
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        if not self.test_mode:
            self._set_group_flag()

        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]
        
    
    def load_annotations(self, det_info_path, seg_info_path):
        if self.det_datainfo_client_args:
            _det_infos_reader = mmcv.FileClient(
                                    **self.det_datainfo_client_args,
                                    scope='main_process')
            det_data_infos = _det_infos_reader.client.query(
                                det_info_path,
                                projection=['context', 'frame_idx'])
            if self.seg_datainfo_client_args is None:
                # return only detection info
                return sorted([o['_id'] for o in det_data_infos])

            context_infos = defaultdict(dict)
            for o in det_data_infos:
                context_infos[o['context']][o['frame_idx']] = o['_id']


        data_infos = []
        if self.seg_datainfo_client_args:
            _seg_infos_reader = mmcv.FileClient(
                                    **self.seg_datainfo_client_args,
                                    scope='main_process')
            clips = _seg_infos_reader.client.query(seg_info_path,
                                                   projection=None)
            clip_dict = {}
            for clip in clips:
                clip_id = clip['_id']
                sequence = clip['sequence']

                assert clip_id not in clip_dict.keys()
                clip_dict[clip_id] = sequence

            for part in self.seg_clip_parts:
                sequence = clip_dict[part]
                for seq_idx, seq in enumerate(sequence):
                    point_file = seq[0]
                    idx = os.path.splitext(point_file)[0]
                    pts_loader = partial(
                        self.pts_loader,
                        sensors=self.sensor_filter,
                        sensor_signals=self.sensor_signals)
                    data_info = dict(
                        pts_info=dict(path=point_file,
                                      sample_idx=idx,
                                      pts_loader=pts_loader))

                    if 'x3' in part:
                        label_file = os.path.splitext(seq[0])[0] + '.label'
                    else:
                        label_file = seq[0]
                    
                    pts_semantic_mask_loader = partial(self.seg_mask_loader, 
                                                    map_labels=self.seg_map_labels)
                    pts_instance_mask_loader = partial(self.ins_mask_loader, 
                                                    map_labels=self.seg_map_labels)
                    anno_info = dict(
                        pts_semantic_mask_path=label_file,
                        pts_semantic_mask_loader=pts_semantic_mask_loader,
                        pts_instance_mask_path=label_file,
                        pts_instance_mask_loader=pts_instance_mask_loader)
                    if self.det_datainfo_client_args:
                        anno_info['bbox3d_info_id'] = context_infos[part][seq_idx]
                    data_info['ann_info'] = anno_info

                    data_infos.append(data_info)

        return data_infos
    

    def get_data_info(self, index):
        if self.det_datainfo_client_args is None:
            # only segmentation data
            return copy.deepcopy(self.data_infos[index])

        if self.det_infos_reader is None:
            self.det_infos_reader = mmcv.FileClient(**self.det_datainfo_client_args)
        if self.seg_datainfo_client_args:
            bbox3d_id = self.data_infos[index]['ann_info']['bbox3d_info_id']
        else:
            bbox3d_id = self.data_infos[index]
        info = self.det_infos_reader.get((self.det_info_path, bbox3d_id))
        info['pts_info']['pts_loader'] = partial(
                                            self.pts_loader,
                                            sensors=self.sensor_filter,
                                            sensor_signals=self.sensor_signals)
        sample_idx = info['sample_idx']
        imgs_info = []
        for img in info['images']:
            lidar2img = np.array(img['cam_intrinsic']) @ np.array(
                img['tf_lidar_to_cam'])
            imgs_info.append(dict(filename=img['path'], lidar2img=lidar2img))

        info['ego_pose'] = np.array(info['ego_pose'])
        for sweep in info['pts_info']['sweeps']:
            sweep['rel_pose'] = np.array(sweep['rel_pose'])

        
        pts_info = info['pts_info']
        if self.seg_datainfo_client_args is not None:
            # add segmentation msg
            pts_info.update(self.data_infos[index]['pts_info'])

        # modify pts pts/sweep path when using "cowa3d-base" database
        pts_info['path'] = pts_info['path'].split('/')[-1]
        for sweep in pts_info['sweeps']:
            sweep['path'] = sweep['path'].split('/')[-1]

        input_dict = dict(
            sample_idx=sample_idx,
            pts_info=pts_info,
            imgs_info=imgs_info)

        if 'annos' in info:
            annos = self.get_ann_info(info)
            input_dict['ann_info'] = annos
        if self.seg_datainfo_client_args:
            # add segmentation file path
            input_dict['ann_info'].update(
                copy.deepcopy(self.data_infos[index]['ann_info']))

        return input_dict


    def get_ann_info(self, info):
        annos = info['annos']
        difficulty = annos['difficulty']

        gt_names_3d = np.array(annos['name_3d'])
        gt_bboxes_3d = np.array(annos['bbox_3d']).astype(
            'float32').reshape(-1, 7)
        if 'velocity' in annos:
            gt_velocity = np.array(annos['velocity']).astype(
                'float32').reshape(-1, 2)
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d,
                                           gt_velocity], axis=-1)

        gt_bboxes_3d = LiDARInstance3DBoxesVel(gt_bboxes_3d,
                                               box_dim=gt_bboxes_3d.shape[-1])

        gt_labels_3d = self.class_name_to_label_index(gt_names_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names_3d=gt_names_3d,
            plane=None,
            difficulty=difficulty)
        return anns_results


    @staticmethod
    def pts_loader(result, pts_b, sensors, sensor_signals):
        pts = np.load(BytesIO(pts_b))
        sensor_mask = None
        if sensors is not None:
            sensor_mask = [pts['sensor'] == s for s in sensors]
            sensor_mask = np.stack(sensor_mask, axis=-1).any(axis=-1)
            pts = pts[sensor_mask]
        sig_vals = []
        for sig_name in sensor_signals:
            sig_val = pts[sig_name].astype(np.float32)
            if sig_name in ['intensity', 'value']:
                sig_val /= 255.0
            sig_vals.append(sig_val)
        pts = np.stack(sig_vals, axis=-1)
        if 'sensor_mask' not in result:
            result['sensor_mask'] = [sensor_mask]
        else:
            result['sensor_mask'].append(sensor_mask)
        return pts

    @staticmethod
    def ins_mask_loader(result, mask_b, map_labels):
        mask = np.load(BytesIO(mask_b))
        mask = map_labels[mask]
        sensor_mask = result.get('sensor_mask', None)
        if sensor_mask is not None:
            mask = mask[sensor_mask]
        return mask

    @staticmethod
    def seg_mask_loader(result, mask_b, map_labels, frame_id=0):
        mask = np.load(BytesIO(mask_b))
        mask = map_labels[mask]
        sensor_mask = result.get('sensor_mask', None)
        if sensor_mask is not None:
            mask = mask[sensor_mask[frame_id]]
        return mask

    def class_name_to_label_index(self, gt_names):
        gt_labels = []
        for cat in gt_names:
            if cat in self.DET_CLASSES:
                gt_labels.append(self.DET_CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        return np.array(gt_labels).astype(np.int64)


    def evaluate(self, results, **kwargs):
        if kwargs.get('do_seg', False):
            print(self.evaluate_seg(results, **kwargs))
        if kwargs.get('do_det', False):
            print(self.evaluate_det(results, **kwargs))
        return


    def evaluate_seg(self, results, metric='seg', ignore_class=['unlabeled'],
                    logger=None, show=False, out_dir=None, pipeline=None, **kwargs):
        if isinstance(results[0], dict):
            seg_results = [res['seg3d_confusion_matrix'] for res in results]
        histogram = np.sum(np.stack(seg_results, axis=0), axis=0)
        eval_mask = []
        eval_classes = []
        for c in self.SEG_CLASSES:
            eval_mask.append(c not in ignore_class)
            if c not in ignore_class:
                eval_classes.append(c)
        histogram = histogram[eval_mask][:, eval_mask]
        inter = np.diag(histogram)
        union = np.sum(histogram, axis=0) + np.sum(histogram, axis=1) - inter
        iou = inter / np.clip(union, 1, None)
        eval_result = OrderedDict()
        for c, score in zip(eval_classes, iou):
            eval_result[c] = score
        eval_result['all'] = iou.mean()

        table_data = [
            ['Class', 'mIoU']]
        for c, score in eval_result.items():
            table_data.append(
                [c, f'{100 * score:.3f}'])
        table = AsciiTable(table_data)
        table.inner_footing_row_border = True
        print_log('\n' + table.table, logger=logger)
        return eval_result
    

    def evaluate_det(self,
                 results,
                 metric='cowa',
                 class_names_mapping=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 **kwargs):
        """Evaluation in KITTI protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default: 'waymo'. Another supported metric is 'kitti'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            pklfile_prefix (str | None): The prefix of pkl files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            submission_prefix (str | None): The prefix of submission datas.
                If not specified, the submission data will not be generated.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.

        Returns:
            dict[str: float]: results of each evaluation metric
        """
        accept_type = ['cowa']
        if isinstance(metric, list):
            for m in metric:
                assert m in accept_type, f'invalid metric {m}'
        else:
            assert metric in accept_type, f'invalid metric {metric}'

        id_map = None
        eval_classes = self.DET_CLASSES
        if class_names_mapping is not None:
            olds, eval_classes = [], []
            for old, new in class_names_mapping:
                assert old not in olds
                olds.append(old)
                if new not in eval_classes:
                    eval_classes.append(new)
            id_map = []
            for idx, (old, new) in enumerate(class_names_mapping):
                id_map.append((idx, eval_classes.index(new)))

        gt_annos = []
        xy_reange = [self.pcd_limit_range[i] for i in [0, 1, 3, 4]]
        _infos_reader = mmcv.FileClient(**self.det_datainfo_client_args,
                                        scope='main_process')
        all_annos = _infos_reader.client.query(
            self.det_info_path, projection=['annos'])
        all_annos = {annos['_id']: annos for annos in all_annos}
        for _id in self.data_infos:
            if self.seg_datainfo_client_args:
                bbox3d_id = self.data_infos[_id]['ann_info']['bbox3d_info_id']
            else:
                bbox3d_id = _id

            gt_i = self.get_ann_info(all_annos[bbox3d_id])
            bboxes = gt_i['gt_bboxes_3d'].convert_to(
                Box3DMode.LIDAR).tensor.numpy()
            labels = gt_i['gt_labels_3d']
            if id_map is not None:
                for old, new in id_map:
                    labels[labels == old] = new
            in_range = gt_i['gt_bboxes_3d'].in_range_bev(xy_reange).numpy()
            if bboxes.shape[-1] == 9:   # with velocity
                bboxes, velocity = bboxes[..., :7], bboxes[..., 7:]
                gt_annos.append(dict(
                    bboxes=bboxes,
                    labels=labels,
                    velocity=velocity,
                    ignore=(~in_range)))
            else:
                gt_annos.append(dict(
                    bboxes=bboxes,
                    labels=labels,
                    ignore=(~in_range)))

        det_results = []
        for i in range(len(results)):
            res_i = results[i]
            if 'pts_bbox' in results[i]:
                res_i = results[i]['pts_bbox']
            bboxes = res_i['boxes_3d'].convert_to(
                Box3DMode.LIDAR).tensor.numpy()
            labels = res_i['labels_3d'].numpy()
            scores = res_i['scores_3d'].numpy()
            # mask = scores > 0.4
            # bboxes, labels, scores = bboxes[mask], labels[mask], scores[mask]
            if bboxes.shape[-1] == 9:   # with velocity
                bboxes, velocity = bboxes[..., :7], bboxes[..., 7:]
                det_results.append(dict(bboxes=bboxes,
                                        labels=labels,
                                        velocity=velocity,
                                        scores=scores))
            else:
                det_results.append(dict(bboxes=bboxes,
                                        labels=labels,
                                        scores=scores))

        return eval_map_flexible(
            det_results, gt_annos, match_thrs=[0.3, 0.5, 0.7],
            breakdowns=[
                dict(
                    type='RangeBreakdown',
                    ranges=dict(
                        Dist_Near=(0, 30),
                        Dist_Middle=(30, 50),
                        Dist_Far=(50, 100),
                        Dist_VeryFar=(100, 10000)))],
            matcher=dict(type='MatcherCoCo'),
            tp_metrics=[
                        # dict(type='MeanAverageVelocityError'),
                        dict(type='MeanIOU3D')],
            classes=eval_classes, logger=logger,
            report_config=[
                ('car_70',
                 lambda x: x['Class'] == 'vehicle' and
                 x['Thres'] == 0.7 and
                 x['Breakdown'] == 'All'),
                ('bcar_70',
                 lambda x: x['Class'] == 'big_vehicle' and
                 x['Thres'] == 0.7 and
                 x['Breakdown'] == 'All'),
                ('ped_50',
                 lambda x: x['Class'] == 'pedestrian' and
                 x['Thres'] == 0.5 and
                 x['Breakdown'] == 'All'),
                ('cyc_50',
                 lambda x: x['Class'] == 'motorcycle_bicycle' and
                 x['Thres'] == 0.5 and
                 x['Breakdown'] == 'All'),
                ('tri_50',
                 lambda x: x['Class'] == 'tricycle' and
                 x['Thres'] == 0.5 and
                 x['Breakdown'] == 'All'),
                ('barrier_30',
                 lambda x: x['Class'] == 'barrier' and
                 x['Thres'] == 0.3 and
                 x['Breakdown'] == 'All'),
            ], nproc=None)


@DATASETS.register_module()
class COWADatasetDetSegFSD(COWADatasetDetSeg):
    def __init__(self, pipeline=None, *args, **kwargs):
        super(COWADatasetDetSegFSD, self).__init__(pipeline=pipeline, *args, **kwargs)

        self.pipeline_types = [p['type'] for p in pipeline]
        self._skip_type_keys = None
    
    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)

        # example = self.pipeline(input_dict)
        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)
        
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example
    
    def update_skip_type_keys(self, skip_type_keys):
        self._skip_type_keys = skip_type_keys
