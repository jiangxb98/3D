import numpy as np
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import (get_box_type, LiDARInstance3DBoxes, Box3DMode)
from ..core.evaluation import eval_map_flexible
from io import BytesIO
import mmcv
from functools import partial
import os.path as osp
from collections import OrderedDict
from terminaltables import AsciiTable
from mmcv.utils import print_log


@DATASETS.register_module(force=True)
class CowaDataset(Custom3DDataset):
    CLASSES = ()
    def __init__(self,
                 info_path,
                 datainfo_client_args=None,
                 task='seg',
                 pipeline=None,
                 det_classes=None,
                 seg_classes=None,
                 test_mode=False,
                 modality=None,
                 box_type_3d='LiDAR',
                 load_interval=1,
                 filter_empty_gt=False,
                 sensors=[0],
                 sensor_signals=['x', 'y', 'z', 'intensity'],
                 map_labels=None,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5]):
        super(Custom3DDataset, self).__init__()
        self.info_path = info_path
        self.datainfo_client_args = datainfo_client_args
        self.task = task
        assert self.task in ['seg', 'det', 'multi']
        self.modality = modality
        assert self.modality is not None
        self.test_mode = test_mode
        self.filter_empty_gt = filter_empty_gt
        self.pcd_limit_range = pcd_limit_range

        # WARN, may will be  dismissed one day
        self.sensors = sensors
        self.sensor_signals = sensor_signals
        self.map_labels = np.array(map_labels, dtype=np.uint8)

        self.DET_CLASSES = self.get_classes(det_classes)
        self.SEG_CLASSES = self.get_classes(seg_classes)

        self.infos_reader = None
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.pts_loader = partial(self.load_points, sensors=self.sensors,
                                  sensor_signals=self.sensor_signals)
        self.pts_semantic_mask_loader = partial(self.semantic_seg_loader, 
                                            map_labels=self.map_labels)
        self.pts_instance_mask_loader = partial(self.instance_seg_loader, 
                                            map_labels=self.map_labels)
        
        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()
        
        # load annotations
        self.data_infos = self.load_annotations(self.info_path)

        load_index = []
        for i, info in enumerate(self.data_infos):
            if (info % load_interval) == (load_interval - 1):
                load_index.append(i)
        self.data_infos = [self.data_infos[i] for i in load_index]
        
        if hasattr(self, 'flag'):
            self.flag = np.array([self.flag[i] for i in load_index])

    def load_annotations(self, info_path):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(info_path))
        return data_infos
    
    def mongo2minio(self, filename):
        '''convert mongo filename to minio filename'''
        return filename
    
    def pts2semseg(self, filename):
        '''convert pts filename to sem_seg_label filename'''
        return filename
    
    def pts2insseg(self, filename):
        '''convert pts filename to ins_seg_label filename'''
        return filename

    def get_data_info(self, index):
        if self.infos_reader is None:
            self.infos_reader = mmcv.FileClient(**self.datainfo_client_args)
        info = self.infos_reader.get((self.info_path, self.data_infos[index]))
        info['pts_info']['pts_loader'] = self.pts_loader
        info['pts_info']['path'] = self.mongo2minio(info['pts_info']['path'])
        info['ego_pose'] = np.array(info['ego_pose'])
        
        for sweep in info['pts_info']['sweeps']:
            sweep['rel_pose'] = np.array(sweep['rel_pose'])
            sweep['path'] = self.mongo2minio(sweep['path'])
            if 'det' not in self.task:
                sweep['semantic_seg_file'] = self.pts2semseg(sweep['path'])
                sweep['instance_seg_file'] = self.pts2insseg(sweep['path'])
        
        imgs_info = []
        for img in info['images']:
            lidar2img = np.array(img['cam_intrinsic']) @ np.array(
                img['tf_lidar_to_cam'])
            imgs_info.append(dict(filename=img['path'], lidar2img=lidar2img))

        annos = {}
        if 'seg' not in self.task: # det or multi
            det_annos = self.get_ann_info(info)
            annos.update(det_annos)
        
        if 'det' not in self.task: # seg or multi
            semantic_seg_file = self.pts2semseg(info['pts_info']['path'])
            instance_seg_file = self.pts2insseg(info['pts_info']['path'])
            seg_annos = dict(
                pts_semantic_mask_path=[semantic_seg_file],
                pts_semantic_mask_loader=self.pts_semantic_mask_loader,
                pts_instance_mask_path=[instance_seg_file],
                pts_instance_mask_loader=self.pts_instance_mask_loader)
            annos.update(seg_annos)

        sample_idx = info['sample_idx']
        input_dict = dict(
            sample_idx=sample_idx,
            pts_info=info['pts_info'],
            imgs_info=imgs_info,
            ann_info=annos)
    
        return input_dict

    def class_name_to_label_index(self, gt_names):
        gt_labels = []
        for cat in gt_names:
            if cat in self.DET_CLASSES:
                gt_labels.append(self.DET_CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        return np.array(gt_labels).astype(np.int64)

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

        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d,
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
    def load_points(result, pts_b, token, sensors, sensor_signals):
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
        key = osp.splitext(token)[0]
        if result.get('sensor_mask') == None:
            result['sensor_mask'] = {}
        result['sensor_mask'].update({key: sensor_mask})
        return pts

    @staticmethod
    def instance_seg_loader(result, mask_b, token, map_labels):
        mask = np.load(BytesIO(mask_b))
        mask = map_labels[mask]
        key = osp.splitext(token)[0]
        sensor_mask = result.get('sensor_mask', None)
        if sensor_mask is not None:
            mask = mask[sensor_mask[key]]
        return mask

    @staticmethod
    def semantic_seg_loader(result, mask_b, token, map_labels):
        mask = np.load(BytesIO(mask_b))
        mask = map_labels[mask]
        key = osp.splitext(token)[0]
        sensor_mask = result.get('sensor_mask', None)
        if sensor_mask is not None:
            mask = mask[sensor_mask[key]]
        return mask

    def evaluate_det(self,
                     results,
                     metric='cowa',
                     class_names_mapping=None,
                     logger=None,
                     pklfile_prefix=None,
                     submission_prefix=None,
                     show=False,
                     out_dir=None):
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
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        all_annos = _infos_reader.client.query(
            self.info_path, projection=['annos'])
        all_annos = {annos['_id']: annos for annos in all_annos}
        for _id in self.data_infos:
            gt_i = self.get_ann_info(all_annos[_id])
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
                        Dist_Far=(50, 10000)))],
            matcher=dict(type='MatcherCoCo'),
            tp_metrics=[dict(type='MeanAverageVelocityError'),
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
        
    def evaluate_seg(self,
                     results,
                     metric='seg',
                     ignore_class=['unlabeled'],
                     logger=None,
                     show=False,
                     out_dir=None,
                     pipeline=None):
            histogram = np.sum(np.stack(results, axis=0), axis=0)
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