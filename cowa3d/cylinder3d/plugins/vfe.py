# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import VOXEL_ENCODERS
from common.ops.voxel.scatter import Scatter
from common.models.voxel_encoders.utils import PointVoxelStatsCalculator
from mmcv.runner import force_fp32


@VOXEL_ENCODERS.register_module()
class CylinderFeatureNet(nn.Module):
    def __init__(self,
                 in_channels=6,
                 feat_channels=(64, 128, 256),
                 pre_reduce_channels=(256,),
                 post_reduce_channels=(16,),
                 pfn_pre_norm=True,
                 pfn_cat_features=False,
                 with_cluster_center=False,
                 with_cluster_center_offset=False,
                 with_covariance=False,
                 with_voxel_center=False,
                 with_voxel_point_count=False,
                 with_voxel_center_offset=True,
                 cylinder_range=[-4, -3.141592653589793, 0,
                                 2, 3.141592653589793, 50],
                 cylinder_partition=[32, 360, 480],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 reduce_op='max',
                 **kwargs):
        super(CylinderFeatureNet, self).__init__()

        cylinder_range = torch.tensor(
            cylinder_range, dtype=torch.float32)
        cylinder_partition = torch.tensor(
            cylinder_partition, dtype=torch.float32)
        voxel_size = (cylinder_range[3:] -
                      cylinder_range[:3]) / cylinder_partition

        self.stats_cal = PointVoxelStatsCalculator(
            voxel_size=voxel_size,
            point_cloud_range=cylinder_range,
            with_cluster_center=with_cluster_center,
            with_cluster_center_offset=with_cluster_center_offset,
            with_covariance=with_covariance,
            with_voxel_center=with_voxel_center,
            with_voxel_point_count=with_voxel_point_count,
            with_voxel_center_offset=with_voxel_center_offset)

        self.fp16_enabled = False

        in_channels = in_channels - 3 + self.stats_cal.out_channels
        feat_channels = [in_channels] + list(feat_channels)
        pre_reduce_channels = [feat_channels[-1]] + list(pre_reduce_channels)
        post_reduce_channels = [
            pre_reduce_channels[-1]] + list(post_reduce_channels)

        if pfn_pre_norm:
            norm_name, self.pfn_pre_norm = build_norm_layer(
                norm_cfg, in_channels)
        else:
            self.pfn_pre_norm = nn.Identity()

        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if pfn_cat_features and i > 0:
                in_filters *= 2
            norm_name, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.num_pfn = len(pfn_layers)
        self.pfn_layers = nn.ModuleList(pfn_layers)

        pre_reduce_layers = []
        for i in range(len(pre_reduce_channels) - 1):
            in_filters = pre_reduce_channels[i]
            out_filters = pre_reduce_channels[i + 1]
            pre_reduce_layers.append(
                nn.Linear(in_filters, out_filters, bias=True))
        if len(pre_reduce_layers) == 0:
            pre_reduce_layers.append(nn.Identity())
        self.pre_reduce_layers = nn.Sequential(*pre_reduce_layers)

        post_reduce_layers = []
        for i in range(len(post_reduce_channels) - 1):
            in_filters = post_reduce_channels[i]
            out_filters = post_reduce_channels[i + 1]
            post_reduce_layers.append(
                nn.Linear(in_filters, out_filters, bias=True))
        if len(post_reduce_layers) > 0:
            post_reduce_layers.append(nn.ReLU(inplace=True))
        else:
            post_reduce_layers.append(nn.Identity())
        self.post_reduce_layers = nn.Sequential(*post_reduce_layers)

        self.reduce_op = reduce_op
        self.pfn_cat_features = pfn_cat_features

    @force_fp32(apply_to=('features', 'points', 'img_feats'))
    def forward(self,
                features,
                coors,
                points=None,
                img_feats=None,
                img_metas=None):

        scatter = Scatter(coors)
        features = [self.stats_cal(features[:, :3], scatter), features[:, 3:]]
        features = torch.cat(features, dim=-1)  # [N, 9]

        features = self.pfn_pre_norm(features)

        for i, pfn in enumerate(self.pfn_layers):
            features = pfn(features)
            if i != len(self.pfn_layers) - 1:
                if self.pfn_cat_features:
                    voxel_feats, voxel_coors = scatter.reduce(features,
                                                              self.reduce_op)
                    # need to concat voxel feats if it is not the last pfn
                    feat_per_point = scatter.mapback(voxel_feats)
                    features = torch.cat([features, feat_per_point], dim=1)

        features = self.pre_reduce_layers(features)
        voxel_feats, voxel_coors = scatter.reduce(features,
                                                  self.reduce_op)
        voxel_feats = self.post_reduce_layers(voxel_feats)
        return voxel_feats, voxel_coors


@VOXEL_ENCODERS.register_module()
class DynamicCylinderFeatureNet(nn.Module):
    def __init__(self,
                 in_channels=6,
                 feat_channels=(64, 128, 256),
                 pre_reduce_channels=(256,),
                 post_reduce_channels=(16,),
                 pfn_pre_norm=True,
                 pfn_cat_features=False,
                 with_cluster_center=False,
                 with_cluster_center_offset=False,
                 with_covariance=False,
                 with_voxel_center=False,
                 with_voxel_point_count=False,
                 with_voxel_center_offset=True,
                 cylinder_range=[-4, -3.141592653589793, 0,
                                 2, 3.141592653589793, 50],
                 cylinder_partition=[32, 360, 480],
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 reduce_op='max',
                 **kwargs):
        super(DynamicCylinderFeatureNet, self).__init__()

        cylinder_range = torch.tensor(
            cylinder_range, dtype=torch.float32)
        cylinder_partition = torch.tensor(
            cylinder_partition, dtype=torch.float32)
        voxel_size = (cylinder_range[3:] -
                      cylinder_range[:3]) / cylinder_partition
        
        self.stats_cal = PointVoxelStatsCalculator(
            voxel_size=voxel_size,
            point_cloud_range=cylinder_range,
            with_cluster_center=with_cluster_center,
            with_cluster_center_offset=with_cluster_center_offset,
            with_covariance=with_covariance,
            with_voxel_center=with_voxel_center,
            with_voxel_point_count=with_voxel_point_count,
            with_voxel_center_offset=with_voxel_center_offset)

        self.fp16_enabled = False

        in_channels = in_channels - 3 + self.stats_cal.out_channels
        feat_channels = [in_channels] + list(feat_channels)
        
        post_reduce_channels = [
            feat_channels[-1]] + list(post_reduce_channels)

        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i > 0:
                in_filters *= 2
            _, norm_layer = build_norm_layer(norm_cfg, out_filters)
            pfn_layers.append(
                nn.Sequential(
                    nn.Linear(in_filters, out_filters, bias=False), norm_layer,
                    nn.ReLU(inplace=True)))
        self.pfn_layers = nn.ModuleList(pfn_layers)

        post_reduce_layers = []
        for i in range(len(post_reduce_channels) - 1):
            in_filters = post_reduce_channels[i]
            out_filters = post_reduce_channels[i + 1]
            post_reduce_layers.append(
                nn.Linear(in_filters, out_filters, bias=True))
        if len(post_reduce_layers) > 0:
            post_reduce_layers.append(nn.ReLU(inplace=True))
        else:
            post_reduce_layers.append(nn.Identity())
        self.post_reduce_layers = nn.Sequential(*post_reduce_layers)

        self.reduce_op = reduce_op


    @force_fp32(apply_to=('features', 'points', 'img_feats'))
    def forward(self, features, coors, points=None, img_feats=None,
                img_metas=None):
        scatter = Scatter(coors)
        features = [self.stats_cal(features[:, :3], scatter), features[:, 3:]]
        features = torch.cat(features, dim=-1)

        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(features)
            voxel_feats, voxel_coors = scatter.reduce(point_feats,
                                                      self.reduce_op)
            if i != len(self.pfn_layers) - 1:
                # need to concat voxel feats if it is not the last pfn
                feat_per_point = scatter.mapback(voxel_feats)
                features = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = self.post_reduce_layers(voxel_feats)
        return voxel_feats, voxel_coors