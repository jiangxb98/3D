# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import spconv.pytorch as spconv

from mmcv.runner import BaseModule, auto_fp16
from mmdet3d.ops import make_sparse_convmodule
from mmdet3d.models.builder import MIDDLE_ENCODERS
from spconv.pytorch import SparseSequential

class ResContextBlock(BaseModule):
    def __init__(self,
                 in_filters,
                 out_filters,
                 conv_type='SubMConv3d',
                 norm_cfg=dict(type='BN1d'),
                 order=('conv', 'norm', 'act'),
                 indice_key=None,
                 init_cfg=None):
        super(ResContextBlock, self).__init__(init_cfg=init_cfg)
        self.conv1 = make_sparse_convmodule(in_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef1",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv1_2 = make_sparse_convmodule(out_filters,  # 3x1x3
                                              out_filters,
                                              kernel_size=(3, 1, 3),
                                              indice_key=indice_key + "bef2",
                                              stride=1,
                                              padding=(1, 0, 1),
                                              conv_type=conv_type,
                                              norm_cfg=norm_cfg,
                                              order=order)

        self.conv2 = make_sparse_convmodule(in_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key + "bef3",
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv3 = make_sparse_convmodule(out_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef4",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

    @auto_fp16()
    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(BaseModule):
    def __init__(self,
                 in_filters,
                 out_filters,
                 pooling=True,
                 height_pooling=False,
                 conv_type='SubMConv3d',
                 norm_cfg=dict(type='BN1d'),
                 order=('conv', 'norm', 'act'),
                 indice_key=None,
                 init_cfg=None):
        super(ResBlock, self).__init__(init_cfg=init_cfg)
        self.pooling = pooling

        self.conv1 = make_sparse_convmodule(in_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key + "bef1",
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv1_2 = make_sparse_convmodule(out_filters,  # 1x3x3
                                              out_filters,
                                              kernel_size=(1, 3, 3),
                                              indice_key=indice_key + "bef2",
                                              stride=1,
                                              padding=(0, 1, 1),
                                              conv_type=conv_type,
                                              norm_cfg=norm_cfg,
                                              order=order)

        self.conv2 = make_sparse_convmodule(in_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef3",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv3 = make_sparse_convmodule(out_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key + "bef4",
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        if pooling:
            if height_pooling:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=2,
                                                padding=1, indice_key=indice_key, bias=False)
            else:
                self.pool = spconv.SparseConv3d(out_filters, out_filters, kernel_size=3, stride=(2, 2, 1),
                                                padding=1, indice_key=indice_key, bias=False)

    @auto_fp16()
    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        resA = resA.replace_feature(resA.features + shortcut.features)

        if self.pooling:
            resB = self.pool(resA)
            return resB, resA
        else:
            return resA


class UpBlock(BaseModule):
    def __init__(self,
                 in_filters,
                 out_filters,
                 up_key=None,
                 conv_type='SubMConv3d',
                 norm_cfg=dict(type='BN1d'),
                 order=('conv', 'norm', 'act'),
                 indice_key=None,
                 init_cfg=None):
        super(UpBlock, self).__init__(init_cfg=init_cfg)

        self.trans_dilao = make_sparse_convmodule(in_filters,  # 3x3x3
                                                  out_filters,
                                                  kernel_size=(3, 3, 3),
                                                  indice_key=indice_key + "new_up",
                                                  stride=1,
                                                  padding=(1, 1, 1),  # p = 1
                                                  conv_type=conv_type,
                                                  norm_cfg=norm_cfg,
                                                  order=order)

        self.conv1 = make_sparse_convmodule(out_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key+'up1',
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv2 = make_sparse_convmodule(out_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key+'up2',
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.conv3 = make_sparse_convmodule(out_filters,  # 3x3x3
                                            out_filters,
                                            kernel_size=(3, 3, 3),  # ks = 3
                                            indice_key=indice_key + 'up3',
                                            stride=1,
                                            padding=(1, 1, 1),  # p = 1
                                            conv_type=conv_type,
                                            norm_cfg=norm_cfg,
                                            order=order)

        self.up_subm = spconv.SparseInverseConv3d(out_filters, out_filters, kernel_size=3, indice_key=up_key,
                                                  bias=False)

    @auto_fp16()
    def forward(self, x, skip):
        upA = self.trans_dilao(x)

        # upsample
        upA = self.up_subm(upA)

        upA = upA.replace_feature(upA.features + skip.features)

        upE = self.conv1(upA)

        upE = self.conv2(upE)

        upE = self.conv3(upE)

        return upE


class ReconBlock(BaseModule):
    def __init__(self,
                 in_filters,
                 out_filters,
                 conv_type='SubMConv3d',
                 norm_cfg=dict(type='BN1d'),
                 order=('conv', 'norm'),
                 indice_key=None,
                 init_cfg=None):
        super(ReconBlock, self).__init__(init_cfg=init_cfg)
        self.conv1 = SparseSequential(*[make_sparse_convmodule(in_filters,  # 3x1x1
                                                               out_filters,
                                                               kernel_size=(3, 1, 1),
                                                               indice_key=indice_key + "bef1",
                                                               stride=1,
                                                               padding=(1, 0, 0),
                                                               conv_type=conv_type,
                                                               norm_cfg=norm_cfg,
                                                               order=order), nn.Sigmoid()])
                                       

        self.conv1_2 = SparseSequential(*[make_sparse_convmodule(in_filters,  # 1x3x1
                                                                 out_filters,
                                                                 kernel_size=(1, 3, 1),
                                                                 indice_key=indice_key + "bef2",
                                                                 stride=1,
                                                                 padding=(0, 1, 0),
                                                                 conv_type=conv_type,
                                                                 norm_cfg=norm_cfg,
                                                                 order=order), nn.Sigmoid()])
                                         

        self.conv1_3 = SparseSequential(*[make_sparse_convmodule(in_filters,  # 1x1x3
                                                                 out_filters,
                                                                 kernel_size=(1, 1, 3),
                                                                 indice_key=indice_key + "bef3",
                                                                 stride=1,
                                                                 padding=(0, 0, 1),
                                                                 conv_type=conv_type,
                                                                 norm_cfg=norm_cfg,
                                                                 order=order), nn.Sigmoid()])
                                         

    @auto_fp16()
    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut2 = self.conv1_2(x)

        shortcut3 = self.conv1_3(x)

        shortcut = shortcut.replace_feature(
            shortcut.features + shortcut2.features + shortcut3.features)

        shortcut = shortcut.replace_feature(shortcut.features * x.features)

        return shortcut


@MIDDLE_ENCODERS.register_module()
class AsymSparseUnet(BaseModule):
    def __init__(self,
                 sparse_shape,
                 in_channels=128,
                 feat_channels=16,
                 num_classes=20,
                 strides=(64, 128, 256, 512),
                 height_poolings=[True, True, False, False],
                 conv_type='SubMConv3d',
                 norm_cfg=dict(type='BN1d'),
                 order=('conv', 'norm', 'act'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        
        down_strides = [feat_channels] + list(strides) # [32, 64, 128, 256, 512]
        up_strides = list(strides) + [strides[-1]]
        up_strides.reverse() # [512, 512, 256, 128, 64]

        assert len(down_strides) == len(up_strides)

        self.num_classes = num_classes
        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(in_channels, feat_channels, conv_type=conv_type, norm_cfg=norm_cfg, 
                                        order=order, indice_key="pre")
        
        indice_keys = []
        resBlocks = nn.ModuleList()
        for i in range(len(down_strides) - 1): # i = 0, 1, 2, 3
            in_c = down_strides[i]
            out_c = down_strides[i + 1]
            height_pooling = height_poolings[i]

            indice_key = "down" + str(i)
            indice_keys.append(indice_key)

            resBlocks.append(ResBlock(in_c, out_c, height_pooling=height_pooling, conv_type=conv_type, norm_cfg=norm_cfg,
                                      order=order, indice_key=indice_key))
        self.resBlocks = resBlocks

        upBlocks = nn.ModuleList()
        for i in range(len(up_strides) - 1):
            in_c = up_strides[i]
            out_c = up_strides[i + 1]
            
            up_key = indice_keys.pop()

            upBlocks.append(UpBlock(in_c, out_c, conv_type=conv_type, norm_cfg=norm_cfg,
                                    order=order, indice_key="up" + str(i), up_key=up_key))
        self.upBlocks = upBlocks

        self.ReconNet = ReconBlock(2 * feat_channels, 2 * feat_channels, conv_type=conv_type, 
                                   norm_cfg=norm_cfg, order=order[0:2], indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * feat_channels, num_classes, indice_key="logit", kernel_size=3, stride=1, padding=1, bias=True)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape, batch_size)
        downc = self.downCntx(ret)

        res = []
        for resBlock in self.resBlocks:
            downc, downb = resBlock(downc)
            res.append((downc, downb))

        upe = res[-1][0]
        for upBlock in self.upBlocks:
            _, downb = res.pop()
            upe = upBlock(upe, downb)

        up0e = self.ReconNet(upe)

        up0e = up0e.replace_feature(torch.cat((up0e.features, upe.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y