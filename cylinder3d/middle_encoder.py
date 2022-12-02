# -*- coding:utf-8 -*-
import torch
import numpy as np
import spconv.pytorch as spconv

from mmcv.runner import BaseModule, auto_fp16
from mmdet3d_gwd.ops import make_sparse_convmodule
from mmdet3d.models.builder import MIDDLE_ENCODERS


class ResContextBlock(BaseModule):
    def __init__(self,
                 in_filters,
                 out_filters,
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 indice_key=None,
                 init_cfg=None):
        super(ResContextBlock, self).__init__(init_cfg=init_cfg)
        self.conv1 = make_sparse_convmodule(in_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef1",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv1_2 = make_sparse_convmodule(out_filters,  # 3x1x3
                                              out_filters,
                                              kernel_size=(3, 1, 3),
                                              indice_key=indice_key + "bef2",
                                              stride=1,
                                              padding=(1, 0, 1),
                                              conv_type='SubMConv3d',
                                              norm_cfg=norm_cfg,
                                              act_cfg=act_cfg,
                                              order=('conv', 'norm', 'act'))

        self.conv2 = make_sparse_convmodule(in_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key + "bef3",
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv3 = make_sparse_convmodule(out_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef4",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

    @auto_fp16()
    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        resA = resA.replace_feature(resA.features + shortcut.features)

        return resA


class ResBlock(BaseModule):
    def __init__(self, in_filters,
                 out_filters,
                 pooling=True,
                 height_pooling=False,
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
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
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv1_2 = make_sparse_convmodule(out_filters,  # 1x3x3
                                              out_filters,
                                              kernel_size=(1, 3, 3),
                                              indice_key=indice_key + "bef2",
                                              stride=1,
                                              padding=(0, 1, 1),
                                              conv_type='SubMConv3d',
                                              norm_cfg=norm_cfg,
                                              act_cfg=act_cfg,
                                              order=('conv', 'norm', 'act'))

        self.conv2 = make_sparse_convmodule(in_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key + "bef3",
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv3 = make_sparse_convmodule(out_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key + "bef4",
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

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
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 indice_key=None,
                 init_cfg=None):
        super(UpBlock, self).__init__(init_cfg=init_cfg)

        self.trans_dilao = make_sparse_convmodule(in_filters,  # 3x3x3
                                                  out_filters,
                                                  kernel_size=(3, 3, 3),
                                                  indice_key=indice_key + "new_up",
                                                  stride=1,
                                                  padding=(1, 1, 1),  # p = 1
                                                  conv_type='SubMConv3d',
                                                  norm_cfg=norm_cfg,
                                                  act_cfg=act_cfg,
                                                  order=('conv', 'norm', 'act'))

        self.conv1 = make_sparse_convmodule(out_filters,  # 1x3x3
                                            out_filters,
                                            kernel_size=(1, 3, 3),
                                            indice_key=indice_key+'up1',
                                            stride=1,
                                            padding=(0, 1, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv2 = make_sparse_convmodule(out_filters,  # 3x1x3
                                            out_filters,
                                            kernel_size=(3, 1, 3),
                                            indice_key=indice_key+'up2',
                                            stride=1,
                                            padding=(1, 0, 1),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv3 = make_sparse_convmodule(out_filters,  # 3x3x3
                                            out_filters,
                                            kernel_size=(3, 3, 3),  # ks = 3
                                            indice_key=indice_key + 'up3',
                                            stride=1,
                                            padding=(1, 1, 1),  # p = 1
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

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
                 # dict(type='BN1d', eps=1e-3, momentum=0.01)
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='Sigmoid'),
                 indice_key=None,
                 init_cfg=None):
        super(ReconBlock, self).__init__(init_cfg=init_cfg)
        self.conv1 = make_sparse_convmodule(in_filters,  # 3x1x1
                                            out_filters,
                                            kernel_size=(3, 1, 1),
                                            indice_key=indice_key + "bef1",
                                            stride=1,
                                            padding=(1, 0, 0),
                                            conv_type='SubMConv3d',
                                            norm_cfg=norm_cfg,
                                            act_cfg=act_cfg,
                                            order=('conv', 'norm', 'act'))

        self.conv1_2 = make_sparse_convmodule(in_filters,  # 1x3x1
                                              out_filters,
                                              kernel_size=(1, 3, 1),
                                              indice_key=indice_key + "bef2",
                                              stride=1,
                                              padding=(0, 1, 0),
                                              conv_type='SubMConv3d',
                                              norm_cfg=norm_cfg,
                                              act_cfg=act_cfg,
                                              order=('conv', 'norm', 'act'))

        self.conv1_3 = make_sparse_convmodule(in_filters,  # 1x1x3
                                              out_filters,
                                              kernel_size=(1, 1, 3),
                                              indice_key=indice_key + "bef3",
                                              stride=1,
                                              padding=(0, 0, 1),
                                              conv_type='SubMConv3d',
                                              norm_cfg=norm_cfg,
                                              act_cfg=act_cfg,
                                              order=('conv', 'norm', 'act'))

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
                 norm_cfg=dict(type='BN1d'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.num_classes = num_classes

        self.sparse_shape = sparse_shape

        self.downCntx = ResContextBlock(
            in_channels, feat_channels, norm_cfg=norm_cfg, indice_key="pre")
        self.resBlock2 = ResBlock(
            feat_channels, 2 * feat_channels, norm_cfg=norm_cfg, height_pooling=True, indice_key="down2")
        self.resBlock3 = ResBlock(
            2 * feat_channels, 4 * feat_channels, norm_cfg=norm_cfg, height_pooling=True, indice_key="down3")
        self.resBlock4 = ResBlock(4 * feat_channels, 8 * feat_channels, norm_cfg=norm_cfg,
                                  pooling=True, height_pooling=False, indice_key="down4")
        self.resBlock5 = ResBlock(8 * feat_channels, 16 * feat_channels, norm_cfg=norm_cfg,
                                  pooling=True, height_pooling=False, indice_key="down5")

        self.upBlock0 = UpBlock(
            16 * feat_channels, 16 * feat_channels, norm_cfg=norm_cfg, indice_key="up0", up_key="down5")
        self.upBlock1 = UpBlock(
            16 * feat_channels, 8 * feat_channels, norm_cfg=norm_cfg, indice_key="up1", up_key="down4")
        self.upBlock2 = UpBlock(
            8 * feat_channels, 4 * feat_channels, norm_cfg=norm_cfg, indice_key="up2", up_key="down3")
        self.upBlock3 = UpBlock(
            4 * feat_channels, 2 * feat_channels, norm_cfg=norm_cfg, indice_key="up3", up_key="down2")

        self.ReconNet = ReconBlock(
            2 * feat_channels, 2 * feat_channels, norm_cfg=norm_cfg, indice_key="recon")

        self.logits = spconv.SubMConv3d(4 * feat_channels, num_classes, indice_key="logit", kernel_size=3, stride=1, padding=1,
                                        bias=True)

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        # x = x.contiguous()
        coors = coors.int()
        # import pdb
        # pdb.set_trace()
        ret = spconv.SparseConvTensor(voxel_features, coors, self.sparse_shape,
                                      batch_size)
        ret = self.downCntx(ret)
        down1c, down1b = self.resBlock2(ret)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down4c, down4b = self.resBlock5(down3c)

        up4e = self.upBlock0(down4c, down4b)
        up3e = self.upBlock1(up4e, down3b)
        up2e = self.upBlock2(up3e, down2b)
        up1e = self.upBlock3(up2e, down1b)

        up0e = self.ReconNet(up1e)

        up0e = up0e.replace_feature(
            torch.cat((up0e.features, up1e.features), 1))

        logits = self.logits(up0e)
        y = logits.dense()
        return y
