
from mmcv.ops import points_in_boxes_part
from mmdet3d.core.bbox import LiDARInstance3DBoxes


class LiDARInstance3DBoxesVel(LiDARInstance3DBoxes):
    def points_in_boxes_part(self, points, boxes_override=None):
        
        if boxes_override is not None:
            boxes = boxes_override
        else:
            boxes = self.tensor[:, :7]    # modified here
        if points.dim() == 2:
            points = points.unsqueeze(0)
        box_idx = points_in_boxes_part(points,
                                       boxes.unsqueeze(0).to(
                                           points.device)).squeeze(0)
        return box_idx