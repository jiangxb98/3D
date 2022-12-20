import torch
import logging
from cylinder3d.vfe import CylinderFeatureNet


def load_checkpoint(model, cylinder_model_path):
    weights = torch.load(cylinder_model_path, map_location='cpu')

    def sub(sd, prefix, exclude=None, name_mapping=None, value_mapping=None):
        if name_mapping is None:
            def _name_mapping(x): return x
        else:
            _name_mapping = name_mapping

        if value_mapping is None:
            def _value_mapping(x): return x
        else:
            _value_mapping = value_mapping

        if exclude is None:
            def _exclude(x): return False
        elif isinstance(exclude, (tuple, list)):
            def _exclude(x): return x in exclude
        else:
            _exclude = exclude

        mapped = type(sd)()
        for w in sd:
            if w.startswith(prefix + '.'):
                mapped[w[len(prefix):].strip('.')] = sd[w]
        sd = mapped
        mapped = type(sd)()
        for w in sd:
            if not _exclude(w):
                mapped[w] = sd[w]
        sd = mapped
        mapped = type(sd)()
        for w in sd:
            mapped[_name_mapping(w)] = sd[w]
        sd = mapped
        mapped = type(sd)()
        for w in sd:
            mapped[w] = _value_mapping(sd[w])
        sd = mapped
        return sd

    model.pfn_pre_norm.load_state_dict(
        sub(weights, 'PPmodel.0',
            value_mapping=lambda w: w[[5, 4, 3, 2, 1, 0, 6, 7, 8]] if w.shape == (9,) else w))
    model.pfn_layers[0][0].load_state_dict(
        sub(weights, 'PPmodel.1', exclude=['bias'],
            value_mapping=lambda w: w[:, [5, 4, 3, 2, 1, 0, 6, 7, 8]] if w.shape == (64, 9) else w))
    model.pfn_layers[0][1].load_state_dict(sub(weights, 'PPmodel.2'))
    model.pfn_layers[1][0].load_state_dict(
        sub(weights, 'PPmodel.4', exclude=['bias']))
    model.pfn_layers[1][1].load_state_dict(sub(weights, 'PPmodel.5'))
    model.pfn_layers[2][0].load_state_dict(
        sub(weights, 'PPmodel.7', exclude=['bias']))
    model.pfn_layers[2][1].load_state_dict(sub(weights, 'PPmodel.8'))

    model.pre_reduce_layers[0].load_state_dict(sub(weights, 'PPmodel.10'))
    model.post_reduce_layers[0].load_state_dict(
        sub(weights, 'fea_compression.0'))
    return model


def test_cylinder_feature_net():
    vfe = CylinderFeatureNet(
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
        cylinder_partition=[31, 359, 479],
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.01),
        reduce_op='max')
    vfe = load_checkpoint(vfe, 'cylinder3d/tests/data/vfe_params.pth').cuda()
    point_input, coors_input = torch.load(
        'cylinder3d/tests/data/vfe_input.pth', map_location='cuda')
    feat_output_tgt, coors_output_tgt = torch.load(
        'cylinder3d/tests/data/vfe_output.pth', map_location='cuda')
    # serial = coors_output_tgt[:, 0]
    # for i in [3, 2, 1]:
    #     serial = serial * \
    #         (coors_output_tgt[:, i].max() + 1) + coors_output_tgt[:, i]
    # idx = serial.argsort()
    # coors_output_tgt = coors_output_tgt[idx][:, [0, 3, 2, 1]]
    # feat_output_tgt = feat_output_tgt[idx]

    point_input = point_input[:, [5, 4, 3, 6, 7, 8]]
    # coors_input = coors_input[:, [0, 3, 2, 1]]
    feat_output, coors_output = vfe(point_input, coors_input)
    assert (coors_output == coors_output_tgt).all()
    assert torch.allclose(feat_output, feat_output_tgt, rtol=1e-4, atol=10)
