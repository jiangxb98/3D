import torch

from mmdet.core.bbox import BaseBBoxCoder
from mmdet.core.bbox.builder import BBOX_CODERS
from .utils import lidar2img_fun

@BBOX_CODERS.register_module()
class BasePointBBoxCoder(BaseBBoxCoder):
    """Bbox coder for CenterPoint.
    Args:
        pc_range (list[float]): Range of point cloud.
        out_size_factor (int): Downsample factor of the model.
        voxel_size (list[float]): Size of voxel.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 post_center_range=None,
                 score_thresh=0.1,
                 num_classes=3,
                 max_num=500,
                 code_size=8):

        self.post_center_range = post_center_range
        self.code_size = code_size
        self.EPS = 1e-6
        self.score_thresh=score_thresh
        self.num_classes = num_classes
        self.max_num = max_num

    def encode(self, bboxes, base_points, gt_box_type=1, img_metas=None):
        """
        Get regress target given bboxes and corresponding base_points
        base_points 是聚类中心
        """

        if bboxes.size(1) in (7, 9, 10):
            assert bboxes.size(1) in (7, 9, 10), f'bboxes shape: {bboxes.shape}'
            assert bboxes.size(0) == base_points.size(0)
            xyz = bboxes[:,:3]  # gt box bottom center 这里还是有问题啊，为什么聚类中心往底面中心跑？
            dims = bboxes[:, 3:6]
            yaw = bboxes[:, 6:7]

            log_dims = (dims + self.EPS).log()

            dist2center = xyz - base_points  # 这里没有除以对角线，因为这是center-based，没有anchor

            delta = dist2center # / self.window_size_meter
            reg_target = torch.cat([delta, log_dims, yaw.sin(), yaw.cos()], dim=1)
            if bboxes.size(1) in (9, 10): # with velocity or copypaste flag
                assert self.code_size == 10
                reg_target = torch.cat([reg_target, bboxes[:, [7, 8]]], dim=1)

        elif bboxes.size(1) == 4:
            x_center_gt = (bboxes[:, 0] + bboxes[:, 2]) * 0.5
            y_center_gt = (bboxes[:, 1] + bboxes[:, 3]) * 0.5
            w_gt = bboxes[:, 2] - bboxes[:, 0]
            h_gt = bboxes[:, 3] - bboxes[:, 1]
            w_target = (w_gt + self.EPS).log()
            h_target = (h_gt + self.EPS).log()

            base_points_uv = lidar2img_fun(base_points, img_metas['lidar2img'], img_metas['scale_factor'][0])
            x_center_target = x_center_gt - base_points_uv[:, 0]
            y_center_target = y_center_gt - base_points_uv[:, 1]
            
            # 朝向角 先空着
            yaw = torch.zeros(x_center_gt.shape, device=base_points.device)

            reg_target = torch.cat([x_center_target, y_center_target, w_target, h_target, yaw.sin(), yaw.cos()], dim=1)

        return reg_target

    def decode(self, reg_preds, base_points, detach_yaw=False):

        assert reg_preds.size(1) in (8, 10)
        assert reg_preds.size(1) == self.code_size

        if self.code_size == 10:
            velo = reg_preds[:, -2:]
            reg_preds = reg_preds[:, :8] # remove the velocity

        dist2center = reg_preds[:, :3] # * self.window_size_meter
        xyz = dist2center + base_points

        dims = reg_preds[:, 3:6].exp() - self.EPS

        sin = reg_preds[:, 6:7]
        cos = reg_preds[:, 7:8]
        yaw = torch.atan2(sin, cos)
        if detach_yaw:
            yaw = yaw.clone().detach()
        bboxes = torch.cat([xyz, dims, yaw], dim=1)
        if self.code_size == 10:
            bboxes = torch.cat([bboxes, velo], dim=1)
        return bboxes
