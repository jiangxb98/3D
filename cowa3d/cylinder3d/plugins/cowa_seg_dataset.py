import mmcv
import os.path as osp

from functools import partial
from mmdet3d.datasets.builder import DATASETS
from common.datasets.cowa_dataset import CowaDataset

@DATASETS.register_module()
class CowaSegDataset(CowaDataset):
    def __init__(self,
                 info_path,
                 parts,
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
        self.parts = parts
        super(CowaSegDataset, self).__init__(
                                        info_path=info_path,
                                        datainfo_client_args=datainfo_client_args,
                                        task=task,
                                        pipeline=pipeline,
                                        det_classes=det_classes,
                                        seg_classes=seg_classes,
                                        test_mode=test_mode,
                                        modality=modality,
                                        box_type_3d=box_type_3d,
                                        load_interval=load_interval,
                                        filter_empty_gt=filter_empty_gt,
                                        sensors=sensors,
                                        sensor_signals=sensor_signals,
                                        map_labels=map_labels,
                                        pcd_limit_range=pcd_limit_range)
        self.evaluate = partial(self.evaluate_seg)
        
    def load_annotations(self, info_path):
        _infos_reader = mmcv.FileClient(**self.datainfo_client_args,
                                        scope='main_process')
        data_infos = sorted(_infos_reader.client.query_index(info_path))

        # filter unlabeled seg data here
        valid_data_infos = []
        for index in data_infos:
            info = _infos_reader.get((info_path, index))
            clip_id = info['context']
            if clip_id in self.parts:
                valid_data_infos.append(index)

        data_infos = valid_data_infos
        
        return data_infos
    
    def mongo2minio(self, filename):
        filename = osp.split(filename)[-1]
        return filename