import torch
import torch.nn as nn

from mmdet.models import BACKBONES
from mmdet3d.models import builder

from .voxel_encoder_fsd import SIRLayer


@BACKBONES.register_module()
class SIR(nn.Module):

    def __init__(
        self,
        num_blocks=5,
        in_channels=[],
        feat_channels=[],
        rel_mlp_hidden_dims=[],
        with_rel_mlp=True,
        with_distance=False,
        with_cluster_center=False,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[1.0, 1.0, 1.0],
        act='relu',
        dropout=0,
        unique_once=False,
        ):
        super().__init__()

        self.num_blocks = num_blocks
        self.unique_once = unique_once
        
        block_list = []
        for i in range(num_blocks):
            return_point_feats = i != num_blocks-1
            kwargs = dict(
                in_channels=in_channels[i],
                feat_channels=feat_channels[i],
                with_distance=with_distance,
                with_cluster_center=with_cluster_center,
                with_rel_mlp=with_rel_mlp,
                rel_mlp_hidden_dims=rel_mlp_hidden_dims[i],
                with_voxel_center=False,
                voxel_size=[0.1, 0.1, 0.1], # not used, placeholder
                point_cloud_range=[-74.88, -74.88, -2, 74.88, 74.88, 4], # not used, placeholder
                norm_cfg=norm_cfg,
                mode=mode,
                fusion_layer=None,
                return_point_feats=return_point_feats,
                return_inv=False,
                rel_dist_scaler=10.0,
                xyz_normalizer=xyz_normalizer,
                act=act,
                dropout=dropout,
            )
            encoder = SIRLayer(**kwargs)
            block_list.append(encoder)
        self.block_list = nn.ModuleList(block_list)
    
    def forward(self, points, features, coors, f_cluster=None):
        # points=(N,) coors = (cls_id, batch_idx, cluster_id)
        points=points[:, :5]
        if self.unique_once:
            new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)  # new_coors(N_clusters, 3),unq_inv(N,)
        else:
            new_coors = unq_inv = None
        
        out_feats = features  #(N,79)

        cluster_feat_list = []
        for i, block in enumerate(self.block_list):
            in_feats = torch.cat([points, out_feats], 1)  #(N,79+5)
            if i < self.num_blocks - 1:
                out_feats, out_cluster_feats = block(in_feats, coors, f_cluster, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            if i == self.num_blocks - 1:
                out_feats, out_cluster_feats, out_coors = block(in_feats, coors, f_cluster, return_both=True, unq_inv_once=unq_inv, new_coors_once=new_coors)
                cluster_feat_list.append(out_cluster_feats)
            
        final_cluster_feats = torch.cat(cluster_feat_list, dim=1)  # 3*(N_clusters,256)-->(N_clusters,768)

        return out_feats, final_cluster_feats, out_coors  # (N,128), (N_clusters,768), (N_clusters,3)
