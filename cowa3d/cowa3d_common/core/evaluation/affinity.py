from ...ops.eval.eval_utils import trans_bev, iou_3d, iou_bev
from .builder import EVAL_AFFINITYCALS


@EVAL_AFFINITYCALS.register_module()
class LidarCenterTransBEV:
    LARGER_CLOSER = False

    def __call__(self, det, gt):
        return trans_bev(det['bboxes'], gt['bboxes'])


@EVAL_AFFINITYCALS.register_module()
class LidarIOU3D:
    LARGER_CLOSER = True

    def __init__(self, z_offset=0.5):
        self.z_offset = z_offset

    def __call__(self, det, gt):
        return iou_3d(det['bboxes'], gt['bboxes'], self.z_offset)


@EVAL_AFFINITYCALS.register_module()
class LidarIOUBEV:
    LARGER_CLOSER = True

    def __call__(self, det, gt):
        return iou_bev(det['bboxes'], gt['bboxes'])
