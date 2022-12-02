custom_imports = dict(
    imports=['cylinder3d'],
    allow_failed_imports=False)

model = dict(
    type='Cylinder3D',
    pts_voxel_layer=dict(
        max_num_points=-1,
        cylinder_range=[-4, -3.141592653589793, 0,
                        2, 3.141592653589793, 50],
        cylinder_partition=[32, 360, 480],
        max_voxels=-1,
        deterministic=False),
    pts_voxel_encoder=dict(
        type='CylinderFeatureNet',
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
        norm_cfg=dict(type='SyncBN', eps=1e-5, momentum=0.1),
        reduce_op='max'),
    pts_middle_encoder=dict(
        type='AsymSparseUnet',
        sparse_shape=[480, 360, 32],
        in_channels=16,
        feat_channels=32,
        num_classes=20,
        norm_cfg=dict(type='SyncBN', eps=1e-5, momentum=0.1)),
    pts_seg_head=dict(
        type='Cylinder3dHead',
        grid_size=[480, 360, 32],
        ignore=0,  # get_target & loss有用到
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
            loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict())
