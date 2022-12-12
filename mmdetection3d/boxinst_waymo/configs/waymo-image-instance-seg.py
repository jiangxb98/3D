dataset_type = 'WaymoDataset'

file_client_args = dict(
    backend='MINIO',
    bucket='ai-waymo-v1.4',
    endpoint='ossapi.cowarobot.cn:9000',
    secure=False)

datainfo_client_args = dict(
    backend='MONGODB',
    database='ai-waymo-v1_4',
    host='mongodb://root:root@172.16.100.41:27017/')

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=True)
# the image is rgb so not convert to rgb, and mean std need exchange dimension
# 不需要将图片转为rgb，所以因为读出来的就是rgb
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=False)

train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadImages',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos',
        with_bbox=True,
        with_label=True,
        with_mask=True,
        with_seg=True,
        file_client_args=file_client_args),
    dict(
        type='FilterLabelImage',
        filter_calss_name=class_names,
        with_mask=True,
        with_seg=True,
        with_mask_3d=False,
        with_seg_3d=False),
    dict(
        type='ResizeMultiViewImage',
        # Target size (w, h)
        img_scale=[(960, 640),(960, 640),(960, 640),(960, 640),(960, 640)],
        # img_scale=[(1920,1280),(1920,1280),(1920,1280),(1920,1280),(1920,1280)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(
        type='PadMultiViewImage',
        # size=[(1280, 1920),(1280, 1920),(1280, 1920),
        #     (1280, 1920),(1280, 1920)]),
        size=[(640, 960),(640, 960),(640, 960),
            (640, 960),(640, 960)]),
    dict(
        type='SampleFrameImage', 
        sample='random',
        guide='gt_bboxes',
        training =True,
        ),  # random or resample
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),

    # 过滤掉2D Boxes外的点
    dict(type='FilterGTBboxesPoints', gt_boxes_enabled=True),

    # To DataContainer
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', 
        keys=['gt_bboxes','gt_labels', 'img', 'points'],
        meta_keys=['filename','img_shape','ori_shape','pad_shape',
            'scale','scale_factor','keep_ratio','lidar2img',
            'sample_idx','img_info','anno_info','pts_info',
            'img_norm_cfg', 'pad_fixed_size', 'pad_size_divisor',
            'sample_img_id','img_sample'],
    )
]

test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadImages',
        file_client_args=file_client_args,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(960, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='ResizeMultiViewImage',
                # Target size (w, h)
                img_scale=[(960, 640),(960, 640),(960, 640),(960, 640),(960, 640)],
                # img_scale=[(1920,1280),(1920,1280),(1920,1280),(1920,1280),(1920,1280)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='NormalizeMultiViewImage', **img_norm_cfg),
            dict(
                type='PadMultiViewImage',
                # size=[(1280, 1920),(1280, 1920),(1280, 1920),
                #     (1280, 1920),(1280, 1920)]),
                size=[(640, 960),(640, 960),(640, 960),
                    (640, 960),(640, 960)]),
            dict(
                type='SampleFrameImage', 
                sample='random',
                guide='gt_bboxes',
                training =False,
                ),  # random or resample    
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points','img'])
        ])
]

eval_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadImages',
        file_client_args=file_client_args,),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(960, 640),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='ResizeMultiViewImage',
                # Target size (w, h)
                # img_scale=[(1920, 1280),(1920, 1280),(1920, 1280),(1920, 1280),(1920, 1280)],
                img_scale=[(960, 640),(960, 640),(960, 640),(960, 640),(960, 640)],
                multiscale_mode='range',
                keep_ratio=True),
            dict(type='NormalizeMultiViewImage', **img_norm_cfg),
            dict(
                type='PadMultiViewImage',
                # size=[(1280, 1920),(1280, 1920),(1280, 1920),
                #     (1280, 1920),(1280, 1920)]),
                size=[(640, 960),(640, 960),(640, 960),
                    (640, 960),(640, 960)]),
            dict(
                type='SampleFrameImage', 
                sample='random',
                guide='gt_bboxes',
                training =False,
                ),  # random or resample    
            dict(type='DefaultFormatBundle3D', class_names=class_names),
            dict(type='Collect3D', keys=['points','img'])
        ])    
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        info_path='training/infos',
        classes=class_names,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        test_mode=False,
        # load one frame every five frames
        load_interval=1,
        datainfo_client_args=datainfo_client_args,
        # 3d segmentation
        load_semseg=True,
        semseg_classes=class_names,
        semseg_info_path='training/infos',  # semantic and instance same path
        # 2d segmentation
        load_panseg=True,  
        panseg_classes = class_names,
        panseg_info_path='training/infos',  # semantic and instance same path
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        info_path='training/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=False,
        semseg_classes=class_names,
        semseg_info_path='training/infos',
        # 2d segmentation
        load_panseg=True,
        panseg_classes = class_names,
        panseg_info_path='training/infos',),
    test=dict(
        type=dataset_type,
        info_path='training/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=False,
        semseg_classes=class_names,
        semseg_info_path='training/infos',
        # 2d segmentation
        load_panseg=False,  
        panseg_classes = class_names,
        panseg_info_path='training/infos',))        

evaluation = dict(interval=1, pipeline=eval_pipeline)
custom_imports = dict(
    imports=['boxinst_waymo.mmdet3d_jgf'],
    allow_failed_imports=False)
