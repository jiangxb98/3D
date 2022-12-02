from mmdet3d.models.builder import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.segmentors import Base3DSegmentor
import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from .voxelize import Cylinderization


@DETECTORS.register_module()
class Cylinder3D(Base3DSegmentor):
    def __init__(
            self,
            pts_voxel_layer=None,
            pts_voxel_encoder=None,
            pts_middle_encoder=None,
            pts_seg_head=None,
            train_cfg=None,
            test_cfg=None,
            pretrained=None,
            init_cfg=None):
        super(Cylinder3D, self).__init__(init_cfg=init_cfg)
        if pts_voxel_layer:
            self.pts_voxel_layer = Cylinderization(**pts_voxel_layer)
        if pts_voxel_encoder:
            self.pts_voxel_encoder = builder.build_voxel_encoder(
                pts_voxel_encoder)
        if pts_middle_encoder:
            self.pts_middle_encoder = builder.build_middle_encoder(
                pts_middle_encoder)
        if pts_seg_head:
            self.pts_seg_head = builder.build_head(pts_seg_head)

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_neck') and self.img_neck is not None

    @property
    def with_img_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_backbone') and self.img_backbone is not None

    @property
    def with_pts_neck(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'pts_neck') and self.pts_neck is not None

    @property
    def with_pts_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'pts_backbone') and self.pts_backbone is not None

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        out_coors = []
        out_points = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_points, res_coors = self.pts_voxel_layer(res)
            out_coors.append(res_coors)
            out_points.append(res_points)
        out_points = torch.cat(out_points, dim=0)
        out_coors_batch = []
        for i, coor in enumerate(out_coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            out_coors_batch.append(coor_pad)
        out_coors_batch = torch.cat(out_coors_batch, dim=0)
        return out_points, out_coors_batch

    def aug_test(self):
        pass

    def encode_decode(self, points, img, img_metas):
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        return img_feats, pts_feats

    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats

    def extract_feat(self, points, img, img_metas):
        img_feats = None
        if img and img_metas:
            img_feats = self.extract_img_feat(img, img_metas)
        pts_feats = self.extract_pts_feat(points, img_feats, img_metas)
        return (img_feats, pts_feats)

    def extract_pts_feat(self, points, img_feats=None, img_metas=None):
        # points_with_pol = (N, 6) = (N, (ro, theta, z, x, y, inten))
        points_with_pol, self.coors = self.voxelize(points)
        voxel_features, feature_coors = self.pts_voxel_encoder(
            points_with_pol, self.coors, points, img_feats, img_metas)
        batch_size = self.coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, feature_coors, batch_size)
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      pts_semantic_mask=None):
        img_feats, pts_feats = self.encode_decode(
            points, img=img, img_metas=img_metas)
        x = self.pts_seg_head(pts_feats)
        loss = self.pts_seg_head.loss(x, self.coors, pts_semantic_mask)
        return loss

    def simple_test(self, points, img_metas, img=None, rescale=False,
                    pts_semantic_mask=None):
        img_feats, pts_feats = self.encode_decode(
            points, img=img, img_metas=img_metas)
        x = self.pts_seg_head(pts_feats)
        if pts_semantic_mask is not None:
            pts_semantic_mask = pts_semantic_mask[0]
        results_list = self.pts_seg_head.simple_test(
            x, self.coors, img_metas, rescale, pts_semantic_mask)
        return results_list
