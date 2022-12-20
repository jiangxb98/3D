# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.cnn import build_norm_layer
from mmcv.runner import force_fp32
from torch import nn

from ...ops import Scatter
from mmdet3d.models import builder
from mmdet3d.models.builder import VOXEL_ENCODERS
from .utils import PointVoxelStatsCalculator


@VOXEL_ENCODERS.register_module(name='DynamicSimpleVFE', force=True)
class DynamicSimpleVFE(nn.Module):
    def __init__(self, *args, **kwargs):
        super(DynamicSimpleVFE, self).__init__()
        self.fp16_enabled = False

    @torch.no_grad()
    @force_fp32(out_fp16=True)
    def forward(self, features, coors):
        features, features_coors = Scatter(coors).reduce(features, 'mean')
        return features, features_coors


@VOXEL_ENCODERS.register_module()
class DynamicSimpleVFEWithVirtual(DynamicSimpleVFE):
    def __init__(self,
                 virtual_flag_dim=-2,
                 real_points_dims=[0, 1, 2, 3, -1],
                 virtual_points_dims=[*range(14), -1],
                 **kwargs):
        super(DynamicSimpleVFEWithVirtual, self).__init__(**kwargs)
        self.virtual_flag_dim = virtual_flag_dim
        self.real_points_dims = real_points_dims
        self.virtual_points_dims = virtual_points_dims

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        num_points = features.shape[0]
        virtual_flag = features[:, self.virtual_flag_dim]
        real_points_mask = virtual_flag == 1
        painted_points_mask = virtual_flag == 0
        virtual_points_mask = virtual_flag == -1
        real_points = features[real_points_mask][:, self.real_points_dims]
        real_coors = coors[real_points_mask]
        num_real = real_points.shape[0]

        painted_points = features[painted_points_mask][:,
                         self.virtual_points_dims]
        painted_coors = coors[painted_points_mask]
        num_painted = painted_points.shape[0]
        painted_points = torch.cat(
            [painted_points, painted_points.new_ones([num_painted, 1])],
            dim=-1)

        virtual_points = features[virtual_points_mask][:,
                         self.virtual_points_dims]
        virtual_coors = coors[virtual_points_mask]
        num_virtual = virtual_points.shape[0]
        virtual_points = torch.cat(
            [virtual_points, virtual_points.new_zeros([num_virtual, 1])],
            dim=-1)

        num_real_cols = len(self.real_points_dims)
        num_virtual_cols = len(self.virtual_points_dims) + 1
        num_scattered_cols = num_real_cols + num_virtual_cols + 2
        real_cols = slice(num_real_cols)
        virtual_cols = slice(num_real_cols, num_real_cols + num_virtual_cols)

        real_rows = slice(num_real)
        virtual_rows = slice(num_real, num_real + num_painted + num_virtual)

        features_pad = features.new_zeros([num_points, num_scattered_cols])
        features_pad[real_rows, real_cols] = real_points
        features_pad[real_rows, -2] = 1
        features_pad[virtual_rows, virtual_cols] = torch.cat(
            [painted_points, virtual_points], dim=0)
        features_pad[virtual_rows, -1] = 1
        coors_pad = torch.cat([real_coors, painted_coors, virtual_coors], dim=0)

        features, features_coors = Scatter(coors_pad).reduce(features_pad,
                                                             'sum')

        _real_cnt = features[:, [-2]].clamp(1)
        _virtual_cnt = features[:, [-1]].clamp(1)

        features_real_cols = features[:, real_cols] / _real_cnt
        features_virtual_cols = features[:, virtual_cols] / _virtual_cnt

        features = torch.cat([features_real_cols, features_virtual_cols],
                             dim=-1)
        return features, features_coors


@VOXEL_ENCODERS.register_module(name='DynamicVFE', force=True)
class DynamicVFE(nn.Module):
    def __init__(self,
                 in_channels=4,
                 feat_channels=(64,),
                 with_cluster_center=False,
                 with_cluster_center_offset=True,
                 with_covariance=False,
                 with_voxel_center=False,
                 with_voxel_point_count=False,
                 with_voxel_center_offset=True,
                 voxel_size=(0.2, 0.2, 4),
                 point_cloud_range=(0, -40, -3, 70.4, 40, 1),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 reduce_op='max',
                 fusion_layer=None,
                 return_point_feats=False):
        super(DynamicVFE, self).__init__()
        assert len(feat_channels) > 0

        self.stats_cal = PointVoxelStatsCalculator(
            voxel_size=voxel_size,
            point_cloud_range=point_cloud_range,
            with_cluster_center=with_cluster_center,
            with_cluster_center_offset=with_cluster_center_offset,
            with_covariance=with_covariance,
            with_voxel_center=with_voxel_center,
            with_voxel_point_count=with_voxel_point_count,
            with_voxel_center_offset=with_voxel_center_offset)

        self.in_channels = in_channels - 3 + self.stats_cal.out_channels
        self.return_point_feats = return_point_feats
        self.fp16_enabled = False

        self.point_cloud_range = point_cloud_range

        feat_channels = [self.in_channels] + list(feat_channels)
        vfe_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            vfe_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.vfe_layers = nn.ModuleList(vfe_layers)
        self.num_vfe = len(vfe_layers)

        self.fusion_layer = None
        if fusion_layer is not None:
            self.fusion_layer = builder.build_fusion_layer(fusion_layer)
        self.reduce_op = reduce_op

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        scatter = Scatter(coors)
        features = [self.stats_cal(features[:, :3], scatter), features[:, 3:]]
        features = torch.cat(features, dim=-1)
        for i, vfe in enumerate(self.vfe_layers):
            point_feats = vfe(features)
            if (i == len(self.vfe_layers) - 1 and self.fusion_layer is not None
                    and img_feats is not None):
                point_feats = self.fusion_layer(img_feats, points, point_feats,
                                                img_metas)
            voxel_feats, voxel_coors = scatter.reduce(point_feats,
                                                      self.reduce_op)
            if i != len(self.vfe_layers) - 1:
                # need to concat voxel feats if it is not the last vfe
                feat_per_point = scatter.mapback(voxel_feats)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        if self.return_point_feats:
            return point_feats
        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class DynamicVFEWithVirtual(DynamicVFE):
    def __init__(self, virtual_flag_dim=-2, **kwargs):
        super(DynamicVFEWithVirtual, self).__init__(**kwargs)
        self.virtual_flag_dim = virtual_flag_dim

    @force_fp32(out_fp16=True)
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):
        virtual_point_mask = features[:, self.virtual_flag_dim] == -1
        features[:, self.virtual_flag_dim] = 0
        features[virtual_point_mask, self.virtual_flag_dim] = 1
        return super(DynamicVFEWithVirtual, self).forward(features,
                                                          coors,
                                                          points,
                                                          img_feats,
                                                          img_metas)
