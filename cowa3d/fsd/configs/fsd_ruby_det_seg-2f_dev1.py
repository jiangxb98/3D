_base_ = [
    '_base_/datasets/ruby_det_seg-2f.py',
    '../../base_configs/schedules/cosine_2x.py',
    '../../base_configs/default_runtime.py',
]

seg_voxel_size = (0.25, 0.25, 0.2)
sparse_shape = [32, 1000, 1000]
point_cloud_range = [-125, -125, -2, 125, 125, 4]
mapped_class_names = ['big_vehicle',
                      'pedestrian',
                      'vehicle',
                      'motorcycle_bicycle',
                      'tricycle',
                      'barrier']
num_classes = len(mapped_class_names)
seg_score_thresh = (0.3, 0.25, 0.3, 0.25, 0.25, 0.25)
semantic_class_names = (
    'unlabeled', 'road-plane', 'curb', 'other-ground', 'terrain',
    'vegetation', 'pillars', 'framework', 'building', 'fence',
    'traffic-sign', 'other-structure', 'noise', 'road-users', 'road-block')
BN = 'SyncBN'  # SyncBN  BN1d naiveSyncBN1d
points_sweeps_num = 2

segmentor = dict(
    type='VoteSegmentor',
    need_full_seg=True,
    sweeps_num=points_sweeps_num,

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        # feat_channels=[32, 32],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type=BN, eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type=BN, eps=1e-3, momentum=0.01),
        # base_channels=64,
        # output_channels=128,
        # encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        # encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        # decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        # decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
        # slim laryer
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (128, 128, 128)),
        encoder_paddings=((1, ), (1, 1, 1), ((0, 1, 1), 1, 1)),
        decoder_channels=((128, 128, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 0), (1, 0), (0, 1)), # decoder paddings seem useless in SubMConv
    ),


    decode_neck=dict(
        type='Voxel2PointScatterNeck',
        voxel_size=seg_voxel_size,
        point_cloud_range=point_cloud_range,
    ),

    segmentation_head=dict(
        type='VoteSegHead',
        in_channel=67,
        hidden_dims=[128, 128],
        # in_channel=35,
        # hidden_dims=[64, 64],
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type=BN),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_segment=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=100.0),
        loss_lova=dict(
            type='LovaszLoss_',
            per_image=False,
            reduction='mean',
            loss_weight=1.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
        need_full_seg=True,
        num_classes_full=len(semantic_class_names)-1,
        ignore_illegal_label=True,
    ),
    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=('big_vehicle',
                      'pedestrian',
                      'vehicle',
                      'motorcycle_bicycle',
                      'tricycle',
                      'barrier'), # for training log
        centroid_offset=False,
    ),
)

model = dict(
    type='FSD',
    sweeps_num=points_sweeps_num,

    segmentor=segmentor,

    backbone=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[96,] + [133, ] * 2,
        # in_channels=[63,] + [132, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True,
    ),

    bbox_head=dict(
        type='SparseClusterHeadV2',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder', code_size=10),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        loss_vel=dict(type='L1Loss', loss_weight=0.5),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=['big_vehicle',]),
            dict(class_names=['pedestrian',]),
            dict(class_names=['vehicle', ]),
            dict(class_names=['motorcycle_bicycle', ]),
            dict(class_names=['tricycle', ]),
            dict(class_names=['barrier', ]),
        ],
        class_names=mapped_class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128), vel=(2, 2, 128), # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='relu',
        ),
        as_rpn=True,
    ),
    roi_head=dict(
        type='GroupCorrectionHead',
        num_classes=num_classes,
        roi_extractor=dict(
             type='DynamicPointROIExtractor',
             extra_wlh=[0.5, 0.5, 0.5],
             max_inbox_point=256,
             debug=False,
        ),
        bbox_head=dict(
            type='FullySparseBboxHead',
            num_classes=num_classes,
            num_blocks=6,
            in_channels=[213, 146, 146, 146, 146, 146], 
            # in_channels=[180, 145, 145, 145, 145, 145], 
            feat_channels=[[128, 128], ] * 6,
            rel_mlp_hidden_dims=[[16, 32],] * 6,
            rel_mlp_in_channels=[13, ] * 6,
            reg_mlp=[512, 512],
            cls_mlp=[512, 512],
            mode='max',
            xyz_normalizer=[20, 20, 4],
            act='gelu',
            geo_input=True,
            with_corner_loss=True,
            corner_loss_weight=1.0,
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            norm_cfg=dict(type='LN', eps=1e-3),
            unique_once=True,

            loss_bbox=dict(
                type='L1Loss',
                reduction='mean',
                loss_weight=2.0),

            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=1.0),
            cls_dropout=0.1,
            reg_dropout=0.1,
        ),
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None
    ),

    train_cfg=dict(
        score_thresh=seg_score_thresh,
        sync_reg_avg_factor=True,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        disable_pretrain=True,
        disable_pretrain_topks=[600, 200, 600, 200, 200, 200],
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=None,
            score_thr=0.1,
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            assigner=[
                dict( # big_vehicle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict( # pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # vehicle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.45,
                    neg_iou_thr=0.45,
                    min_pos_iou=0.45,
                    ignore_iof_thr=-1
                ),
                dict( # motorcycle_bicycle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # tricycle
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
                dict( # barrier
                    type='MaxIoUAssigner',
                    iou_calculator=dict(
                        type='BboxOverlaps3D', coordinate='lidar'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1
                ),
            ],

            sampler=dict(
                #type='IoUNegPiecewiseSampler',
                type='PseudoSampler',
                num=256,
                pos_fraction=0.55,
                neg_piece_fractions=[0.8, 0.2],
                neg_iou_piece_thrs=[0.55, 0.1],
                neg_pos_ub=-1,
                add_gt_as_proposals=False,
                return_iou=True
            ),
            cls_pos_thr=(0.8, 0.65, 0.8, 0.65, 0.65, 0.5),
            cls_neg_thr=(0.2, 0.15, 0.2, 0.15, 0.15, 0.15),
            sync_reg_avg_factor=True,
            sync_cls_avg_factor=True,
            corner_loss_only_car=True,
            class_names=mapped_class_names,
        )
    ),
    test_cfg=dict(
        score_thresh=seg_score_thresh,
        pre_voxelization_size=(0.1, 0.1, 0.1),
        skip_rcnn=False,
        rpn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            score_thr=0.1, 
            min_bbox_size=0,
            max_num=500,
        ),
        rcnn=dict(
            use_rotate_nms=True,
            nms_pre=-1,
            nms_thr=0.25,
            score_thr=0.4, 
            min_bbox_size=0,
            max_num=500,
        ),
        # return_mode: (0: both, 1: detection, 2: segmentation)
        return_mode=1,
        with_confusion_matrix=True,
        num_seg_cls=len(semantic_class_names)-1,
    ),
    cluster_assigner=dict(
        cluster_voxel_size=dict(
            big_vehicle=(0.3, 0.3, 6),
            pedestrian=(0.1, 0.1, 6),
            vehicle=(0.3, 0.3, 6),
            motorcycle_bicycle=(0.2, 0.2, 6),
            tricycle=(0.2, 0.2, 6),
            barrier=(0.1, 0.1, 6),
        ),
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=dict(
            big_vehicle=0.6,
            pedestrian=0.1,
            vehicle=0.6,
            motorcycle_bicycle=0.4,
            tricycle=0.4,
            barrier=0.1,
        ), # xy-plane distance
        class_names=mapped_class_names,
    ),
)

# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=6)  # NOTE
evaluation = dict(interval=12)
workflow = [('train', 1)]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
custom_hooks = [
    dict(type='DisableAugmentationHook', num_last_epochs=1, skip_type_keys=('ObjectSampleWithSeg', 'RandomFlip3D', 'GlobalRotScaleTrans')),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000)
]

optimizer = dict(
    lr=3e-5,
)
# optimizer_config = dict(type='MultiTaskOptimizerHook', apply_multi_task=False)
checkpoint_config = dict(max_keep_ckpts=2)
custom_imports = dict(
    imports=['cowa3d_common.datasets',
             'fsd.plugins'],
    allow_failed_imports=False)


# tools/dist_train.sh research/configs_cowa/ruby/fsd_ruby-1f.py 8
# tools/dist_test.sh research/configs_cowa/ruby/fsd_ruby-1f.py work_dirs/fsd_ruby-1f-baseline/epoch_6.pth 8 --eval cowa
