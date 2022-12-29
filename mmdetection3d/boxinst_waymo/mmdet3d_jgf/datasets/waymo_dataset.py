import tempfile
from os import path as osp
import csv
import zlib
import mmcv
import tqdm
import numpy as np
from mmcv.utils import print_log
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.core.bbox import get_box_type, LiDARInstance3DBoxes
from mmdet3d.datasets.custom_3d import Custom3DDataset
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.points import get_points_type
from io import BytesIO
from multiprocessing import Pool

from waymo_open_dataset import dataset_pb2 as open_dataset
try:
    from waymo_open_dataset.protos import segmentation_metrics_pb2
    from waymo_open_dataset.protos import segmentation_submission_pb2
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-6-0==1.4.9" '
        'to install the official devkit first.')

TOP_LIDAR_ROW_NUM = 64
TOP_LIDAR_COL_NUM = 2650


@DATASETS.register_module(name='MyWaymoDataset', force=True)
class MyWaymoDataset(Custom3DDataset):
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
                 semseg_classes=None,
                 semseg_info_path=None,
                 load_img=False,
                 load_panseg=False,
                 panseg_classes=None,
                 panseg_info_path=None,
                 load_img_index=[0,1,2,3,4],):

        super(Custom3DDataset, self).__init__()
        self.info_path = info_path
        self.datainfo_client_args = datainfo_client_args
        self.test_mode = test_mode
        self.modality = modality
        self.filter_empty_gt = filter_empty_gt
        self.box_type_3d, self.box_mode_3d = get_box_type(box_type_3d)
        self.load_semseg = load_semseg
        self.load_panseg = load_panseg  # 2d segmentation
        self.load_img = load_img
        self.load_img_index = load_img_index  #collect images to use

        self.CLASSES = self.get_classes(classes)
        self.infos_reader = None
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        if self.load_semseg:
            self.SEMSEG_CLASSES = self.get_classes(semseg_classes)
            self.segCat2id = {name: i for i, name in enumerate(self.SEMSEG_CLASSES)}
            self.semseg_info_path = semseg_info_path
        if self.load_panseg:
            self.panseg_CLASSES = self.get_classes(panseg_classes)
            self.pansegCat2id = {name: i for i, name in enumerate(self.panseg_CLASSES)}
            self.panseg_info_path = panseg_info_path

        self.pipeline_types = [p['type'] for p in pipeline]  # pipline name list
        self._skip_type_keys = None

        # load annotations
        # 这个是所有在training/infos的数据(每帧点云)索引，如果用语义信息，就建议用语义的索引覆盖掉
        self.data_infos = self.load_annotations(self.info_path)  # 158081
        # 这个是load所有的语义标签在OSS上的路径
        if self.load_semseg:  # 23691
            self.semseg_frame_infos = self.read_semseg_infos(self.semseg_info_path, filter_sem='semseg_info')
            # intersection
            self.data_infos = list(set(self.data_infos).intersection(self.semseg_frame_infos))
        # train len(self.panseg_frame_infos)=12295
        if self.load_panseg:
            self.panseg_frame_infos = self.read_semseg_infos(self.panseg_info_path, filter_sem='panseg_info')
            # intersection
            self.data_infos = list(set(self.data_infos).intersection(self.panseg_frame_infos))
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

    # get sementic label indx in mongodb database
    def read_semseg_infos(self, semseg_info_path, filter_sem=None):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(semseg_info_path, filter_sem=filter_sem))
        return data_infos

    def get_data_info(self, index):
        if self.infos_reader is None:
            self.infos_reader = mmcv.FileClient(**self.datainfo_client_args)
        info = self.infos_reader.get((self.info_path, self.data_infos[index]))  # 进入mongodb.py改写的get方法 get a frame info
        sample_idx = info['sample_idx']

        if self.load_panseg and (not self.test_mode):
            if sample_idx not in self.panseg_frame_infos:
                return None
        if self.load_semseg and (not self.test_mode):
            if sample_idx not in self.semseg_frame_infos:
                return None

        # get lidar points from oss
        info['pts_info']['pts_loader'] = self.pts_loader  # 只是定义了函数，没有调用，用到才调用
        # get image info : image path, transform matrix
        img_path_info = []
        for img in info['images']:
            lidar2img = np.array(img['cam_intrinsic']) @ np.array(
                img['tf_lidar_to_cam'])
            img_path_info.append(dict(filename=img['path'], lidar2img=lidar2img))

        info['ego_pose'] = np.array(info['ego_pose'])
        for sweep in info['pts_info']['sweeps']:
            sweep['rel_pose'] = np.array(sweep['rel_pose'])

        input_dict = dict(
            sample_idx=sample_idx,
            pts_info=info['pts_info'],
            img_info=dict(img_loader=self.img_loader,
                img_path_info=img_path_info),)

        # 这里放入了2D的label
        #import pdb;pdb.set_trace()
        if 'annos' in info:
            annos = self.get_ann_info(info)
            input_dict['ann_info'] = annos
            # save semseg and panseg info{path, loader}
            if self.load_semseg:
                input_dict['ann_info']['pts_semantic_mask_loader'] = self.semseg_loader
                input_dict['ann_info']['pts_semantic_mask_path'] = info['semseg_info']['path']
                input_dict['ann_info']['pts_instance_mask_loader'] = self.semseg_loader
                input_dict['ann_info']['pts_instance_mask_path'] = info['semseg_info']['path']
            if self.load_panseg:
                panseg_path_info = []
                # each frame have 5 panseg path
                for panseg in info['panseg_info']:
                    panseg_path_info.append(panseg['path'])
                input_dict['ann_info']['pan_semantic_mask_loader'] = self.panseg_loader
                input_dict['ann_info']['pan_semantic_mask_path'] = panseg_path_info
                input_dict['ann_info']['pan_instance_mask_loader'] = self.panseg_loader
                input_dict['ann_info']['pan_instance_mask_path'] = panseg_path_info

        return input_dict

    @staticmethod
    def pts_loader(results, pts_bytes):
        points = np.load(BytesIO(pts_bytes))
        return np.stack([points['x'].astype('f4'),
                         points['y'].astype('f4'),
                         points['z'].astype('f4'),
                         #np.tanh(points['intensity'].astype('f4')),
                         points['intensity'].astype('f4'),
                         points['elongation'].astype('f4'),
                         points['lidar_idx'].astype('i2'),
                         points['cam_idx_0'].astype('i2'),
                         points['cam_idx_1'].astype('i2'),
                         points['cam_column_0'].astype('i2'),
                         points['cam_column_1'].astype('i2'),
                         points['cam_row_0'].astype('i2'),
                         points['cam_row_1'].astype('i2'),
                         ], axis=-1)

    @staticmethod
    def img_loader(results, pts_bytes):
        return None

    @staticmethod
    def semseg_loader(results, pts_bytes, semseg_name):
        semseg_labels = np.load(BytesIO(pts_bytes))
        return semseg_labels[semseg_name]  # return semantic or instance id

    @staticmethod
    def panseg_loader(results, pts_bytes, panseg_name):
        panseg_labels = np.load(BytesIO(pts_bytes))
        return panseg_labels[panseg_name]  # return semantic or instance id

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
        gt_names = [np.array(n) for n in annos['name']]  # 2d
        gt_bboxes = [np.array(b, dtype=np.float32) for b in annos['bbox']]  # 2d
        selected = [self.drop_arrays_by_name(n, ['DontCare', 'Sign']) for n in  # filter obejcts which we not need
                    gt_names]
        gt_names = [n[s] for n, s in zip(gt_names, selected)]  # select the need objects
        gt_bboxes = [b[s] for b, s in zip(gt_bboxes, selected)]

        gt_names_3d = np.array(annos['name_3d'])
        gt_bboxes_3d = np.array(annos['bbox_3d']).astype('float32').reshape(-1,
                                                                            7)
        selected_3d = self.drop_arrays_by_name(gt_names_3d,
                                               ['DontCare', 'Sign'])
        gt_names_3d = gt_names_3d[selected_3d]
        gt_bboxes_3d = gt_bboxes_3d[selected_3d]
        # exchange width&length
        #gt_bboxes_3d = gt_bboxes_3d[:, [0, 1, 2, 4, 3, 5, 6]]
        gt_bboxes_3d = LiDARInstance3DBoxes(gt_bboxes_3d)  # 里面是一些对框的操作，gt3d的相对位置关系是(0.5,0.5,0)

        # NOTE yaw: LiDAR -> CAM
        # gt_bboxes_3d.tensor[:, 6] = - gt_bboxes_3d.tensor[:, 6] - np.pi / 2
        # gt_bboxes_3d.tensor[:, 6][gt_bboxes_3d.tensor[:, 6] < -np.pi] += 2 * np.pi
        # gt_bboxes_3d.tensor[:, 6][gt_bboxes_3d.tensor[:, 6] > np.pi] -= 2 * np.pi


        gt_labels = [self.class_name_to_label_index(n) for n in gt_names]  # encode label
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

    # 由Custom3DDataset.__getitem__(self, idx)方法来进入这个函数
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
        self.pre_pipeline(input_dict)  # Initialization before data preparation
        # into pipline. eg: LoadPoints, LoadAnnos3D, 调用每个class的__call__(self,example)
        example = input_dict
        for transform, transform_type in zip(self.pipeline.transforms, self.pipeline_types):
            if self._skip_type_keys is not None and transform_type in self._skip_type_keys:
                continue
            example = transform(example)

        # example = self.pipeline(input_dict)
        if 'gt_labels_3d' in example.keys():
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

    def evaluate(self, results, **kwargs):

        if kwargs.get('eval_seg', False):
            print(self.evaluate_seg(results, **kwargs))

        if kwargs.get('eval_det', False):
            print(self.evaluate_det(results, **kwargs))

    def evaluate_det(self,
                    results,
                    metric=None,
                    logger=None,
                    pklfile_prefix=None,
                    submission_prefix=None,
                    show=False,
                    out_dir=None,
                    pipeline=None,
                    *args,
                    **kwargs):
        waymo_root = 'data/waymo/waymo_format'
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None
        self.format_results(results, pklfile_prefix)
        import subprocess
        ret_bytes = subprocess.check_output(
            f'/root/3D/waymo_utils/'
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

    def evaluate_seg(self,
                    results,
                    pc_range,
                    logger=None,
                    pklfile_prefix=None,
                    seg_format_worker=0,
                    ):
        self.seg_format_worker = seg_format_worker
        if pklfile_prefix is None:
            eval_tmp_dir = tempfile.TemporaryDirectory()
            pklfile_prefix = osp.join(eval_tmp_dir.name, 'results')
        else:
            eval_tmp_dir = None

        if 'segmap_3d' in results[0]:
            # 0~21 -> 1~22; Ignore 0 ('TYPE_UNDEFINED') when training
            outputs = [out['segmap_3d'].numpy() + 1 for out in results]
        result_serialized = self.format_segmap(outputs, pc_range)
        waymo_results_final_path = f'{pklfile_prefix}_seg.bin'

        with open(waymo_results_final_path, 'wb') as f:
            f.write(result_serialized)

        import subprocess
        ret_bytes = subprocess.check_output(
            f'/root/3D/waymo_utils/'
            'compute_segmentation_metrics_main '
            f'{pklfile_prefix}_seg.bin '
            f'/disk/deepdata/dataset/waymo_v1.4/bins/wod_semseg_val_set_gt.bin',
            shell=True)
        ret_texts = ret_bytes.decode('utf-8')
        print_log(ret_texts, logger=logger)

        iou_dict = dict()
        return iou_dict

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
                #x, y, z, width, length, height, yaw = box_3d   # NOTE deprecated
                o.object.box.center_x = x
                o.object.box.center_y = y
                o.object.box.center_z = z + height / 2
                o.object.box.length = length
                o.object.box.width = width
                o.object.box.height = height
                o.object.box.heading = yaw
                #o.object.box.heading = -yaw - np.pi / 2   # yam: CAM back to LiDAR  NOTE deprecated
                o.object.type = label2proto[label_3d]
                o.score = score_3d
                o.context_name = info['context']
                o.frame_timestamp_micros = info['timestamp']

            all_objects_serialized.append(objects_proto.SerializeToString())
        all_objects_serialized = b''.join(all_objects_serialized)
        return all_objects_serialized

    def format_one_frame_seg(self, args):
        idx, net_out, context_info, pc_range = args
        info = context_info[self.data_infos_seg_frames[idx]]
        range_image_pred = np.zeros(
            (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)
        range_image_pred_ri2 = np.zeros(
            (TOP_LIDAR_ROW_NUM, TOP_LIDAR_COL_NUM, 2), dtype=np.int32)

        # fill back seg labels to range image format
        points_bytes = self.file_client.get(info['pts_info']['path'])
        points = np.load(BytesIO(points_bytes))
        points_indexing_row = points['lidar_row']
        points_indexing_column = points['lidar_column']
        points_return_idx = points['return_idx']
        points_range_mask = np.ones_like(points_indexing_row, dtype=np.bool)
        if pc_range is not None:
            xyz = np.stack([points['x'].astype('f4'),
                            points['y'].astype('f4'),
                            points['z'].astype('f4')], axis=-1)
            points_class = get_points_type('LIDAR')
            xyz = points_class(xyz, points_dim=xyz.shape[-1])
            points_range_mask = xyz.in_range_3d(pc_range).numpy()
        points_lidar_mask = points['lidar_idx'] == 0    # 0 for open_dataset.LaserName.TOP

        points_mask = points_range_mask & points_lidar_mask
        points_indexing_row = points_indexing_row[points_mask]
        points_indexing_column = points_indexing_column[points_mask]
        points_return_idx = points_return_idx[points_mask]

        points_lidar_in_range_mask = points_lidar_mask[points_range_mask]
        net_out = net_out[points_lidar_in_range_mask]

        assert points_indexing_row.shape[0] == net_out.shape[0]
        assert points_indexing_row.shape[0] == points_indexing_column.shape[0]
        range_image_pred[points_indexing_row[points_return_idx==0],
                points_indexing_column[points_return_idx==0], 1] = net_out[:int((points_return_idx==0).sum())]
        range_image_pred_ri2[points_indexing_row[points_return_idx==1],
                points_indexing_column[points_return_idx==1], 1] = net_out[int((points_return_idx==0).sum()):]

        segmentation_frame = segmentation_metrics_pb2.SegmentationFrame()
        segmentation_frame.context_name = info['context']
        segmentation_frame.frame_timestamp_micros = info['timestamp']
        laser_semseg = open_dataset.Laser()
        laser_semseg.name = open_dataset.LaserName.TOP
        laser_semseg.ri_return1.segmentation_label_compressed = self.compress_array(
            range_image_pred, is_int32=True)
        laser_semseg.ri_return2.segmentation_label_compressed = self.compress_array(
            range_image_pred_ri2, is_int32=True)
        segmentation_frame.segmentation_labels.append(laser_semseg)

        return segmentation_frame

    def format_segmap(self, net_outputs, pc_range=None):

        semseg_frames = self.semseg_frame_info
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        context_info = _infos_reader.client.query(
            self.info_path, projection=['context', 'timestamp', 'pts_info'])
        context_info = {info['_id']: info for info in context_info if info['_id'] in semseg_frames}

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        # for idx, net_out in enumerate(mmcv.track_iter_progress(net_outputs)):
        seg_args = [[i, net_outputs[i], context_info, pc_range] for i in range(len(net_outputs))]
        if self.seg_format_worker == 0:
            segmentation_res = []
            for idx in mmcv.track_iter_progress(range(len(net_outputs))):
                segmentation_res.append(self.format_one_frame_seg(seg_args[idx]))
        else:
            # segmentation_res = mmcv.track_parallel_progress(self.format_one_frame_seg,
            #                                                 seg_args,
            #                                                 self.seg_format_worker)
            with Pool(self.seg_format_worker) as p:
                segmentation_res = list(tqdm.tqdm(p.imap(self.format_one_frame_seg, seg_args), total=len(seg_args)))
        
        segmentation_frame_list = segmentation_metrics_pb2.SegmentationFrameList()
        for segmentation_frame in segmentation_res:
            segmentation_frame_list.frames.append(segmentation_frame)

        return segmentation_frame_list.SerializeToString()
       
    def compress_array(self, array: np.ndarray, is_int32: bool = False):
        """Compress a numpy array to ZLIP compressed serialized MatrixFloat/Int32.

        Args:
            array: A numpy array.
            is_int32: If true, use MatrixInt32, otherwise use MatrixFloat.

        Returns:
            The compressed bytes.
        """
        if is_int32:
            m = open_dataset.MatrixInt32()
        else:
            m = open_dataset.MatrixFloat()
        m.shape.dims.extend(list(array.shape))
        m.data.extend(array.reshape([-1]).tolist())
        return zlib.compress(m.SerializeToString())

