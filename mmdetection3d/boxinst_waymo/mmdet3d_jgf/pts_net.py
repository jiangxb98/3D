from mmcv.runner import BaseModule

from mmdet3d.models.builder import MIDDLE_ENCODERS


@MIDDLE_ENCODERS.register_module()
class GenPoints(BaseModule):
    '''
    定义attention的一些参数
    '''

    def __init__(self,):

        pass

    def forward(self, points, img, pts_feat, img_feat, training):
        if training:
            pass
        return points