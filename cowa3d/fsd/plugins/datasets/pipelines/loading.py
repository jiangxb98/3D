
import mmcv
import torch
import numpy as np
from io import BytesIO
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.datasets.pipelines import LoadAnnotations3D
from mmdet3d.core.points import get_points_type, BasePoints

from cowa3d_common.datasets.pipelines import LoadPoints

@PIPELINES.register_module()
class LoadFakeSweepsAnnos3D(LoadAnnotations3D):
    def __init__(self,
                 sweeps_num,
                 *args, 
                 **kwargs):
        self.sweeps_num = sweeps_num
        super().__init__(*args, **kwargs)

    def _load_semantic_seg_3d(self, results):
        points_num = len(results['points'])
        pts_semantic_mask = results['pts_semantic_mask']
        pts_semantic_mask_num = len(pts_semantic_mask)
        pts_semantic_mask = np.concatenate(
                                [pts_semantic_mask,
                                 np.zeros(points_num-pts_semantic_mask_num, 
                                    dtype=np.uint8)])
        results['pts_semantic_mask'] = pts_semantic_mask
        return results

@PIPELINES.register_module()
class LoadSweepsAnnos3D(LoadAnnotations3D):
    def __int__(self,
                sweeps_num,
                *args, 
                **kwargs):
        self.sweeps_num = sweeps_num
        super().__init__(*args, **kwargs)
    
    def _load_semantic_seg_3d(self, results):
        pts_semantic_mask_loader = results['ann_info'][
            'pts_semantic_mask_loader']
        pts_semantic_mask = results['pts_semantic_mask']
        pts_semantic_mask_list = [pts_semantic_mask]
        pts_info = results['pts_info']
        choices = results.get('sweep_choices', range(self.sweeps_num))
        for idx in choices:
            mask_bytes = self.file_client.get(pts_info['sweeps'][idx]['path'])
            pts_semantic_mask_list.append(pts_semantic_mask_loader(
                                            results, 
                                            mask_bytes,
                                            frame_id=idx+1))
        
        pts_semantic_mask = np.concatenate(pts_semantic_mask_list)
        results['pts_semantic_mask'] = pts_semantic_mask
        assert len(results['pts_semantic_mask']) == len(results['points'])
        return results



@PIPELINES.register_module()
class PointSegClassMerge(object):
    """Merge all labels to a small number of label set
    accroding to the given mapping dict.
    """

    def __init__(self, seg_label_mapping):
        assert isinstance(seg_label_mapping, dict)
        self.seg_label_mapping = seg_label_mapping
        self.num_cls = len(self.seg_label_mapping)
        self.map_table = np.ones(
            self.num_cls, dtype=np.int) * -1
        for ori_id in self.seg_label_mapping:
            self.map_table[ori_id] = self.seg_label_mapping[ori_id]
        assert np.min(self.map_table) != -1

    def __call__(self, results):
        
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.map_table[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(map_dict={self.seg_label_mapping}, '
        return repr_str