_base_ = [
    '../_base_/datasets/dota_wsood.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='CondInst',
    backbone=dict(
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
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
    bbox_head=dict(
        type='CondInstBoxHead',
        num_classes=15,
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
    mask_branch=dict(
        type='CondInstMaskBranch',
        in_channels=256,
        in_indices=[0, 1, 2],
        strides=[8, 16, 32],
        branch_convs=4,
        branch_channels=128,
        branch_out_channels=16),
    mask_head=dict(
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
        topk_per_img=64,
        boxinst_enabled=True,
        bottom_pixels_removed=10,
        pairwise_size=3,
        pairwise_dilation=2,
        pairwise_color_thresh=0.3,
        pairwise_warmup=10000),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(1024, 1024)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='R2H', version={{_base_.data.train.version}}),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline))
optimizer = dict(type='SGD', lr=0.01)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
evaluation = dict(interval=1, metric=['bbox', 'segm'])
custom_imports = dict(
    imports=['boxinst_plugin'],
    allow_failed_imports=False)
