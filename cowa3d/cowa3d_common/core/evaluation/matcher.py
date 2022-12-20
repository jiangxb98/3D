import numpy as np
from ...ops.eval.eval_utils import match_coco
from .builder import EVAL_MATCHERS


class BaseMatcher:
    def __init__(self, match_thrs, affinity_cost_negate=True):
        self._match_thrs = match_thrs
        self.negate = affinity_cost_negate

    @property
    def match_thrs(self):
        return self._match_thrs

    def __call__(self, affinity, gt):
        if self.negate:
            return self.match(-affinity,
                              -np.array(self.match_thrs, np.float32), gt)
        else:
            return self.match(affinity, np.array(self.match_thrs, np.float32),
                              gt)

    def match(self, affinity, match_thrs, gt):
        raise NotImplementedError


@EVAL_MATCHERS.register_module()
class MatcherCoCo(BaseMatcher):

    def match(self, affinity, match_thrs, gt):
        gt_ignore = gt.get('ignore', None)
        if gt_ignore is None:
            gt_ignore = np.zeros(affinity.shape[1], dtype=np.bool)
        gt_crowd = gt.get('crowd', None)
        if gt_crowd is None:
            gt_crowd = np.zeros(affinity.shape[1], dtype=np.bool)
        return match_coco(affinity, match_thrs, gt_ignore, gt_crowd)
