_base_ = [
    './waymo-img-pts.py',
    './cosine_2x.py',
    './default_runtime.py',
]

seg_voxel_size = (0.25, 0.25, 0.2)
point_cloud_range = [-80, -80, -2, 80, 80, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
semantic_class_names = class_names
num_classes = len(class_names)
seg_score_thresh = (0.3, 0.25, 0.25)

pts_segmentor = dict(
    type='VoteSegmentor',
    need_full_seg=False,

    voxel_layer=dict(
        voxel_size=seg_voxel_size,
        max_num_points=-1,  # -1表示使用动态体素化
        point_cloud_range=point_cloud_range,
        max_voxels=(-1, -1)
    ),

    voxel_encoder=dict(
        type='DynamicScatterVFE',
        in_channels=5,
        feat_channels=[64, 64],
        voxel_size=seg_voxel_size,
        with_cluster_center=True,
        with_voxel_center=True,
        point_cloud_range=point_cloud_range,
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        unique_once=True,
    ),

    middle_encoder=dict(
        type='PseudoMiddleEncoderForSpconvFSD',
    ),

    backbone=dict(
        type='SimpleSparseUNet',
        in_channels=64,
        sparse_shape=[32, 640, 640],
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='naiveSyncBN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1)), # decoder paddings seem useless in SubMConv
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
        num_classes=num_classes,
        dropout_ratio=0.0,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='naiveSyncBN1d'),
        act_cfg=dict(type='ReLU'),
        loss_decode=dict(  # detect
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=1.0),
        loss_segment=dict(  # semantic segment
            type='FocalLoss',
            use_sigmoid=True,
            gamma=3.0,
            alpha=0.8,
            loss_weight=100.0),
        loss_lova=dict(
            type='LovaszLoss_',  # 对mean IOU loss进行的优化
            per_image=False,
            reduction='mean',
            loss_weight=1.0),
        loss_vote=dict(
            type='L1Loss',
            loss_weight=1.0),
        need_full_seg=False,
        num_classes_full=len(semantic_class_names),
        ignore_illegal_label=True,
        # segment_range=[-50, 50],
    ),

    train_cfg=dict(
        point_loss=True,
        score_thresh=seg_score_thresh, # for training log
        class_names=('Car', 'Ped', 'Cyc'), # for training log
        centroid_offset=False,
    ),
)

# model settings
model = dict(
    type='MultiModalAutoLabel',
    with_pts_branch=True,
    with_img_branch=False,
    gt_box_type=1, # 1 is 3d,2 is 2d
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe')),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),

    # 点云补全，目前用不上
    # middle_encoder_pts=dict(
    #     type='GenPoints',
    # ),

    img_bbox_head=dict(
        type='CondInstBoxHead',
        num_classes=3,
        in_channels=256,
        center_sampling=True,
        center_sample_radius=1.5,
        norm_on_bbox=True,
        stacked_convs=4,
        feat_channels=256,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
        loss_centerness=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0)),

    img_mask_branch=dict(
        type='CondInstMaskBranch',
        in_channels=256,
        in_indices=[0, 1, 2],
        strides=[8, 16, 32],
        branch_convs=4,
        branch_channels=128,
        branch_out_channels=16),

    img_mask_head=dict(
        type='CondInstMaskHead',
        in_channels=16,
        in_stride=8,
        out_stride=4,
        dynamic_convs=3,
        dynamic_channels=8,
        disable_rel_coors=False,
        bbox_head_channels=256,
        sizes_of_interest=[64, 128, 256, 512, 1024],
        max_proposals=-1,
        topk_per_img=64,  # 每张图片最多采样64个实例
        boxinst_enabled=True,
        bottom_pixels_removed=10,
        pairwise_size=3,  # 3*3大小 9-1=8
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        pairwise_warmup=1,  # 10000
        points_enabled=True),
    
    pts_segmentor = pts_segmentor,

    pts_backbone=dict(
        type='SIR',
        num_blocks=3,
        in_channels=[84,] + [133, ] * 2,
        feat_channels=[[128, 128], ] * 3,
        rel_mlp_hidden_dims=[[16, 32],] * 3,
        norm_cfg=dict(type='LN', eps=1e-3),
        mode='max',
        xyz_normalizer=[20, 20, 4],
        act='gelu',
        unique_once=True,
    ),

    pts_bbox_head=dict(
        type='SparseClusterHeadV2',
        num_classes=num_classes,
        bbox_coder=dict(type='BasePointBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_center=dict(type='L1Loss', loss_weight=0.5),
        loss_size=dict(type='L1Loss', loss_weight=0.5),
        loss_rot=dict(type='L1Loss', loss_weight=0.2),
        in_channel=128 * 3 * 2,
        shared_mlp_dims=[1024, 1024],
        train_cfg=None,
        test_cfg=None,
        norm_cfg=dict(type='LN'),
        tasks=[
            dict(class_names=['Car',]),
            dict(class_names=['Pedestrian',]),
            dict(class_names=['Cyclist',]),
        ],
        class_names=class_names,
        common_attrs=dict(
            center=(3, 2, 128), dim=(3, 2, 128), rot=(2, 2, 128),  # (out_dim, num_layers, hidden_dim)
        ),
        num_cls_layer=2,
        cls_hidden_dim=128,
        separate_head=dict(
            type='FSDSeparateHead',
            norm_cfg=dict(type='LN'),
            act='relu',
        ),
        as_rpn=False,
    ),

    # pts_roi_head=dict(
    #     type='GroupCorrectionHead',
    #     num_classes=num_classes,
    #     roi_extractor=dict(
    #          type='DynamicPointROIExtractor',
    #          extra_wlh=[0.5, 0.5, 0.5],
    #          max_inbox_point=256,
    #          debug=False,
    #     ),
    #     bbox_head=dict(
    #         type='FullySparseBboxHead',
    #         num_classes=num_classes,
    #         num_blocks=6,
    #         in_channels=[213, 146, 146, 146, 146, 146], 
    #         feat_channels=[[128, 128], ] * 6,
    #         rel_mlp_hidden_dims=[[16, 32],] * 6,
    #         rel_mlp_in_channels=[13, ] * 6,
    #         reg_mlp=[512, 512],
    #         cls_mlp=[512, 512],
    #         mode='max',
    #         xyz_normalizer=[20, 20, 4],
    #         act='gelu',
    #         geo_input=True,
    #         with_corner_loss=True,
    #         corner_loss_weight=1.0,
    #         bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
    #         norm_cfg=dict(type='LN', eps=1e-3),
    #         unique_once=True,

    #         loss_bbox=dict(
    #             type='L1Loss',
    #             reduction='mean',
    #             loss_weight=2.0),

    #         loss_cls=dict(
    #             type='CrossEntropyLoss',
    #             use_sigmoid=True,
    #             reduction='mean',
    #             loss_weight=1.0),
    #         cls_dropout=0.1,
    #         reg_dropout=0.1,
    #     ),
    #     train_cfg=None,
    #     test_cfg=None,
    #     pretrained=None,
    #     init_cfg=None
    # ),

    # training and testing settings
    train_cfg=dict(
        
        img=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.4,
                min_pos_iou=0,
                ignore_iof_thr=-1),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),

        pts=dict(   
            score_thresh=seg_score_thresh,
            sync_reg_avg_factor=True,
            pre_voxelization_size=(0.1, 0.1, 0.1),
            disable_pretrain=True,
            disable_pretrain_topks=[600, 200, 200],
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
                    dict( # Car
                        type='MaxIoUAssigner',
                        iou_calculator=dict(
                            type='BboxOverlaps3D', coordinate='lidar'),
                        pos_iou_thr=0.45,
                        neg_iou_thr=0.45,
                        min_pos_iou=0.45,
                        ignore_iof_thr=-1
                    ),
                    dict( # Ped
                        type='MaxIoUAssigner',
                        iou_calculator=dict(
                            type='BboxOverlaps3D', coordinate='lidar'),
                        pos_iou_thr=0.35,
                        neg_iou_thr=0.35,
                        min_pos_iou=0.35,
                        ignore_iof_thr=-1
                    ),
                    dict( # Cyc
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
                cls_pos_thr=(0.8, 0.65, 0.65),
                cls_neg_thr=(0.2, 0.15, 0.15),
                sync_reg_avg_factor=True,
                sync_cls_avg_factor=True,
                corner_loss_only_car=True,
                class_names=class_names,
            )),
    ),

    test_cfg=dict(

        img=dict(
            nms_pre=200,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=50
        ),

        pts=dict(
            score_thresh=seg_score_thresh,
            pre_voxelization_size=(0.1, 0.1, 0.1),
            skip_rcnn=True,
            rpn=dict(
                use_rotate_nms=True,
                nms_pre=-1,
                nms_thr=0.25,  # 用了一阶段得到分数进行筛选
                score_thr=0.1, 
                min_bbox_size=0,
                max_num=500,
            ),
            rcnn=dict(
                use_rotate_nms=True,
                nms_pre=-1,
                nms_thr=0.25,
                score_thr=0.1, 
                min_bbox_size=0,
                max_num=500,
            ),
            return_mode=1)     # 0: both, 1: detection, 2: segmentation
        ),

    cluster_assigner=dict(
        cluster_voxel_size=dict(
            Car=(0.3, 0.3, 6),
            Cyclist=(0.2, 0.2, 6),
            Pedestrian=(0.05, 0.05, 6),
        ),
        min_points=2,
        point_cloud_range=point_cloud_range,
        connected_dist=dict(
            Car=0.6,
            Cyclist=0.4,
            Pedestrian=0.1,
        ), # xy-plane distance
        class_names=class_names,
    ),
)


# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=1)  # NOTE
# runner = dict(type='IterBasedRunner', max_iters=1)
workflow = [('train', 1),('val', 1)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerFullHook')
    ])
    
custom_hooks = [
    #dict(type='SegmentationDataSwitchHook', ratio=1),
    dict(type='DisableAugmentationHook', num_last_epochs=1, skip_type_keys=('ObjectSample', 'RandomFlip3D', 'GlobalRotScaleTrans')),
    dict(type='EnableFSDDetectionHookIter', enable_after_iter=4000, threshold_buffer=0.3, buffer_iter=8000)
]

optimizer = dict(
    lr=3e-5,
)
# optimizer_config = dict(type='MultiTaskOptimizerHook', apply_multi_task=True)
checkpoint_config = dict(max_keep_ckpts=3)

custom_imports = dict(
    imports=['boxinst_waymo.mmdet3d_jgf'],
    allow_failed_imports=False)