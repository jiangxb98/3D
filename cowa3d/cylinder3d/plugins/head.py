import torch
import torch.nn.functional as F

from torch import nn as nn
from mmdet.models import HEADS
from mmdet3d.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32, auto_fp16
from common.ops.voxel.scatter import Scatter
from .eval_utils import pts_semantic_confusion_matrix


@HEADS.register_module()
class Cylinder3dHead(BaseModule):
    def __init__(self,
                 grid_size,
                 ignore=0,
                 num_classes=20,
                 loss_func=dict(
                     type='CrossEntropyLoss',
                     reduction='mean',
                     avg_non_ignore=True,
                     loss_weight=1.0),
                 loss_lova=dict(
                     type='LovaszLoss',
                     per_image=False,
                     reduction='mean',
                     loss_weight=1.0),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.grid_size = grid_size
        self.ignore = ignore
        self.num_cls = num_classes

        loss_func.update({'ignore_index': self.ignore})
        loss_lova.update({'ignore': self.ignore})
        self.loss_func = build_loss(loss_func)
        self.loss_lova = build_loss(loss_lova)

    def get_targets(self,
                    coors,
                    labels):
        '''
        coors = (N, 4) = (N, (bs, i, j, k))
        labels = [label, label, ..., ], label.shape = (N)
        '''
        batch_size = len(labels)
        labels = torch.cat(labels, dim=0)
        scatter = Scatter(coors)
        labels = F.one_hot(labels.to(torch.int64), self.num_cls)
        voxel_labels_sum, voxel_coors = scatter.reduce(labels.float(), 'sum')
        grid_size = [batch_size] + list(self.grid_size)
        target = coors.new_ones(grid_size, dtype=torch.long) * self.ignore
        b, i, j, k = voxel_coors.long().T
        target[b, i, j, k] = torch.argmax(voxel_labels_sum, dim=1)
        return target

    @auto_fp16()
    def forward(self, feats):
        return feats

    @force_fp32()
    def loss(self,
             outputs,
             coors,
             pts_semantic_mask):
        voxel_label = self.get_targets(coors, pts_semantic_mask)
        loss_lova = self.loss_lova(
            torch.nn.functional.softmax(outputs, dim=1), voxel_label)
        loss_func = self.loss_func(outputs, voxel_label)
        loss_dict = {'loss_lova': loss_lova, 'loss_func': loss_func}
        return loss_dict

    @force_fp32(apply_to='x')
    def simple_test(self,
                    outputs,
                    coors,
                    img_metas,
                    rescale=False,
                    pts_semantic_mask=None,
                    cur_frame_mask=None):
        b, i, j, k = coors.long().T
        pts_outputs = outputs.argmax(dim=1)[b, i, j, k]
        pts_outputs = pts_outputs[cur_frame_mask]
        if pts_semantic_mask is not None:
            pts_semantic_mask = torch.cat(pts_semantic_mask, dim=0)
            pts_semantic_mask = pts_semantic_mask[cur_frame_mask]
            return [pts_semantic_confusion_matrix(pts_outputs,
                                                  pts_semantic_mask,
                                                  self.num_cls)]
        return [pts_outputs.to(torch.uint8).cpu().numpy()]
