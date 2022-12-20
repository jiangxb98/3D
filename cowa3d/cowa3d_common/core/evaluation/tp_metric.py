import os

import numpy as np
from ...ops.eval.eval_utils import iou_3d_pairwise
from .builder import EVAL_TPMETRIC


@EVAL_TPMETRIC.register_module()
class MeanIOU3D:
    name = "MeanIOU3D"

    def __call__(self, tp_pairs, key=None):
        bboxes_pairs = tp_pairs['bboxes']
        iou = iou_3d_pairwise(bboxes_pairs[:, 0], bboxes_pairs[:, 1])
        # if key is not None:
        #     os.makedirs('tp_metric/', exist_ok=True)
        #     fname = 'tp_metric/' + self.name + '_' + '_'.join(
        #         f'{v}' for v in key.values()) + '.npy'
        #     np.save(fname, iou)
        return iou.mean()


@EVAL_TPMETRIC.register_module()
class MeanAverageVelocityError:
    name = "mAVE"

    def __call__(self, tp_pairs, key=None):
        velocity_pairs = tp_pairs['velocity']
        velocity_error = (velocity_pairs[:, 0] - velocity_pairs[:, 1])
        velocity_error = np.linalg.norm(velocity_error, ord=2, axis=-1)
        # if key is not None:
        #     os.makedirs('tp_metric/', exist_ok=True)
        #     fname = 'tp_metric/' + self.name + '_' + '_'.join(
        #         f'{v}' for v in key.values()) + '.npy'
        #     np.save(fname, iou)
        return velocity_error.mean()
