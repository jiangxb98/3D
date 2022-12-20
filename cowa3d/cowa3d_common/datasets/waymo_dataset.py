import tempfile
from os import path as osp
import csv
import mmcv
import numpy as np
from mmcv.utils import print_log
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.core.bbox import get_box_type, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from io import BytesIO


@DATASETS.register_module(name='WaymoDataset', force=True)
class WaymoDataset(Custom3DDataset):
    CLASSES = ('Car', 'Pedestrian', 'Cyclist')

    def __init__(self,
                 info_path,
                 pipeline=None,
                 classes=None,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 load_interval=1,
                 pcd_limit_range=[-85, -85, -5, 85, 85, 5],
                 datainfo_client_args=None,
                 load_semseg=False,
                 semseg_classes=(),
                 semseg_info_path=''):
        super(Custom3DDataset, self).__init__()
        self.info_path = info_path
        self.datainfo_client_args = datainfo_client_args
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.load_semseg = load_semseg

        self.CLASSES = self.get_classes(classes)
        self.infos_reader = None
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if self.load_semseg:
            self.SEMSEG_CLASSES = self.get_classes(semseg_classes)
            self.segCat2id = {name: i for i, name in enumerate(self.SEMSEG_CLASSES)}
            self.semseg_info_path = semseg_info_path

        self.pipeline_types = [p['type'] for p in pipeline]
        self._skip_type_keys = None

        # load annotations
        self.data_infos = self.load_annotations(self.info_path)
        if self.load_semseg:
            self.semseg_frame_info = self.read_semseg_info(semseg_info_path)

        # process pipeline
        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        # set group flag for the samplers
        if not self.test_mode:
            self._set_group_flag()

        assert self.modality is not None
        self.pcd_limit_range = pcd_limit_range
        self.data_infos = self.data_infos[::load_interval]
        if hasattr(self, 'flag'):
            self.flag = self.flag[::load_interval]

    def load_annotations(self, info_path):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(info_path))
        return data_infos

    def read_semseg_info(semseg_info_path):
        result = {}
        with open(semseg_info_path) as f:
            reader = csv.reader(f)
            for idx, line in enumerate(reader):
                if idx == 0: continue
                items = line.strip().split(',')
                # {"sample_idx": "minio_path"}
                result[items[0]] = items[4]
        return result

    def get_data_info(self, index):
        if self.infos_reader is None:
            self.infos_reader = mmcv.FileClient(**self.datainfo_client_args)
        info = self.infos_reader.get((self.info_path, self.data_infos[index]))
        sample_idx = info['sample_idx']

        if self.load_semseg and (not self.test_mode):
            if not sample_idx in self.semseg_frame_info:
                return None

        info['pts_info']['pts_loader'] = self.pts_loader
        imgs_info = []
        for img in info['images']:
            lidar2img = np.array(img['cam_intrinsic']) @ np.array(
                img['tf_lidar_to_cam'])
            imgs_info.append(dict(filename=img['path'], lidar2img=lidar2img))

        info['ego_pose'] = np.array(info['ego_pose'])
        for sweep in info['pts_info']['sweeps']:
            sweep['rel_pose'] = np.array(sweep['rel_pose'])

        input_dict = dict(
            sample_idx=sample_idx,
            pts_info=info['pts_info'],
            imgs_info=imgs_info)

        if 'annos' in info:
            annos = self.get_ann_info(info)
            input_dict['ann_info'] = annos
            if self.load_semseg:
                input_dict['ann_info']['pts_semantic_mask_loader'] = self.semseg_loader
                input_dict['ann_info']['pts_semantic_mask_path'] = self.semseg_frame_info.get(sample_idx, None)

        return input_dict

    @staticmethod
    def pts_loader(results, pts_bytes):
        points = np.load(BytesIO(pts_bytes))
        return np.stack([points['x'].astype('f4'),
                         points['y'].astype('f4'),
                         points['z'].astype('f4'),
                         #np.tanh(points['intensity'].astype('f4')),
                         points['intensity'].astype('f4'),
                         points['elongation'].astype('f4')], axis=-1)

    @staticmethod
    def semseg_loader(results, pts_bytes):
        semseg_labels = np.load(BytesIO(pts_bytes))
        return semseg_labels['semseg_cls']

    def class_name_to_label_index(self, gt_names):
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        return np.array(gt_labels).astype(np.int64)

    def get_ann_info(self, info):
        annos = info['annos']
        difficulty = annos['difficulty']
        # we need other objects to avoid collision when sample
        gt_names = [np.array(n) for n in annos['name']]
        gt_bboxes = [np.array(b, dtype=np.float32) for b in annos['bbox']]
        selected = [self.drop_arrays_by_name(n, ['DontCare', 'Sign']) for n in
                    gt_names]
        gt_names = [n[s] for n, s in zip(gt_names, selected)]
        gt_bboxes = [b[s] for b, s in zip(gt_bboxes, selected)]

        gt_names_3d = np.array(annos['name_3d'])
        gt_bboxes_3d = np.array(annos['bbox_3d']).astype('float32').reshape(-1,
                                                                            7)
        selected_3d = self.drop_arrays_by_name(gt_names_3d,
                                               ['DontCare', 'Sign'])
        gt_names_3d = gt_names_3d[selected_3d]
        gt_bboxes_3d = gt_bboxes_3d[selected_3d]
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)

        gt_labels = [self.class_name_to_label_index(n) for n in gt_names]
        gt_labels_3d = self.class_name_to_label_index(gt_names_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names_3d=gt_names_3d,
            gt_bboxes=gt_bboxes,
            gt_labels=gt_labels,
            gt_names=gt_names,
            plane=None,
            difficulty=difficulty)
        return anns_results

    @staticmethod
    def drop_arrays_by_name(gt_names, drop_classes):
        inds = [i for i, x in enumerate(gt_names) if x not in drop_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds

    @staticmethod
    def keep_arrays_by_name(gt_names, use_classes):
        inds = [i for i, x in enumerate(gt_names) if x in use_classes]
        inds = np.array(inds, dtype=np.int64)
        return inds


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

        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)

        # example = self.pipeline(input_dict)
        if self.filter_empty_gt and \
                (example is None or
                    ~(example['gt_labels_3d']._data != -1).any()):
            return None
        return example

    def update_skip_type_keys(self, skip_type_keys):
        self._skip_type_keys = skip_type_keys


    def format_results(self, outputs, pklfile_prefix=None):
        if 'pts_bbox' in outputs[0]:
            outputs = [out['pts_bbox'] for out in outputs]
        result_serialized = self.bbox2result_waymo(outputs)

        waymo_results_final_path = f'{pklfile_prefix}.bin'

        with open(waymo_results_final_path, 'wb') as f:
            f.write(result_serialized)

        return waymo_results_final_path

    def evaluate(self,
                 results,
                 metric=None,
                 logger=None,
                 pklfile_prefix=None,
                 submission_prefix=None,
                 show=False,
                 out_dir=None,
                 pipeline=None):
        waymo_root = 'data/waymo/waymo_format'
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
        self.format_results(results, pklfile_prefix)
        import subprocess
        ret_bytes = subprocess.check_output(
            f'./plugins/common_plugins/core/evaluation/waymo_utils/'
            'compute_detection_metrics_main '
            f'{pklfile_prefix}.bin '
            f'/disk/deepdata/dataset/waymo_v1.2/waymo_format/gt.bin',
            shell=True)
        ret_texts = ret_bytes.decode('utf-8')
        print_log(ret_texts, logger=logger)
        # parse the text to get ap_dict
        ap_dict = {
            'Vehicle/L1 mAP': 0,
            'Vehicle/L1 mAPH': 0,
            'Vehicle/L2 mAP': 0,
            'Vehicle/L2 mAPH': 0,
            'Pedestrian/L1 mAP': 0,
            'Pedestrian/L1 mAPH': 0,
            'Pedestrian/L2 mAP': 0,
            'Pedestrian/L2 mAPH': 0,
            'Sign/L1 mAP': 0,
            'Sign/L1 mAPH': 0,
            'Sign/L2 mAP': 0,
            'Sign/L2 mAPH': 0,
            'Cyclist/L1 mAP': 0,
            'Cyclist/L1 mAPH': 0,
            'Cyclist/L2 mAP': 0,
            'Cyclist/L2 mAPH': 0,
            'Overall/L1 mAP': 0,
            'Overall/L1 mAPH': 0,
            'Overall/L2 mAP': 0,
            'Overall/L2 mAPH': 0
        }
        mAP_splits = ret_texts.split('mAP ')
        mAPH_splits = ret_texts.split('mAPH ')
        for idx, key in enumerate(ap_dict.keys()):
            split_idx = int(idx / 2) + 1
            if idx % 2 == 0:  # mAP
                ap_dict[key] = float(mAP_splits[split_idx].split(']')[0])
            else:  # mAPH
                ap_dict[key] = float(mAPH_splits[split_idx].split(']')[0])
        ap_dict['Overall/L1 mAP'] = \
            (ap_dict['Vehicle/L1 mAP'] + ap_dict['Pedestrian/L1 mAP'] +
             ap_dict['Cyclist/L1 mAP']) / 3
        ap_dict['Overall/L1 mAPH'] = \
            (ap_dict['Vehicle/L1 mAPH'] + ap_dict['Pedestrian/L1 mAPH'] +
             ap_dict['Cyclist/L1 mAPH']) / 3
        ap_dict['Overall/L2 mAP'] = \
            (ap_dict['Vehicle/L2 mAP'] + ap_dict['Pedestrian/L2 mAP'] +
             ap_dict['Cyclist/L2 mAP']) / 3
        ap_dict['Overall/L2 mAPH'] = \
            (ap_dict['Vehicle/L2 mAPH'] + ap_dict['Pedestrian/L2 mAPH'] +
             ap_dict['Cyclist/L2 mAPH']) / 3
        if eval_tmp_dir is not None:
            eval_tmp_dir.cleanup()
        if show or out_dir:
            raise NotImplementedError
        return ap_dict

    def bbox2result_waymo(self, net_outputs):
        from waymo_open_dataset import label_pb2
        from waymo_open_dataset.protos import metrics_pb2
        class2proto = {
            'Car': label_pb2.Label.TYPE_VEHICLE,
            'Pedestrian': label_pb2.Label.TYPE_PEDESTRIAN,
            'Sign': label_pb2.Label.TYPE_SIGN,
            'Cyclist': label_pb2.Label.TYPE_CYCLIST,
        }
        label2proto = {}
        for c in class2proto:
            if c in self.cat2id:
                label2proto[self.cat2id[c]] = class2proto[c]

        assert len(net_outputs) == len(self.data_infos), \
            'invalid list length of network outputs'

        all_objects_serialized = []

        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        context_info = _infos_reader.client.query(
            self.info_path, projection=['context', 'timestamp'])
        context_info = {info['_id']: info for info in context_info}

        for idx, pred_dicts in enumerate(
                mmcv.track_iter_progress(net_outputs)):
            info = context_info[self.data_infos[idx]]
            objects_proto = metrics_pb2.Objects()
            for box_3d, score_3d, label_3d in zip(
                    pred_dicts['boxes_3d'].tensor.tolist(),
                    pred_dicts['scores_3d'].tolist(),
                    pred_dicts['labels_3d'].tolist()):
                o = objects_proto.objects.add()
                x, y, z, length, width, height, yaw = box_3d
                o.object.box.center_x = x
                o.object.box.center_y = y
                o.object.box.center_z = z + height / 2
                o.object.box.length = length
                o.object.box.width = width
                o.object.box.height = height
                o.object.box.heading = yaw
                o.object.type = label2proto[label_3d]
                o.score = score_3d
                o.context_name = info['context']
                o.frame_timestamp_micros = info['timestamp']

            all_objects_serialized.append(objects_proto.SerializeToString())
        all_objects_serialized = b''.join(all_objects_serialized)
        return all_objects_serialized
