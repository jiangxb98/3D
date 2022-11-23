dataset_type = 'WaymoDataset'

file_client_args = dict(
    backend='MINIO',
    bucket='ai-waymo-v1.4',
    endpoint='ossapi.cowarobot.cn:9000',
    secure=False)

datainfo_client_args = dict(
    backend='MONGODB',
    database='ai-waymo-v1_4',
    host='mongodb://root:root@172.16.110.100:27017/')

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=True)
# the image is rgb so not convert to rgb, and mean std Exchange dimension
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395], to_rgb=False)

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
    dict(type='NormalizeMultiViewImage', **img_norm_cfg),
    dict(
        type='PadMultiViewImage',
        size=[(1280, 1920),(1280, 1920),(1280, 1920),
            (1280, 1920),(1280, 1920)]),
    dict(
        type='SampleFrameImage', 
        sample='random',
        guide='gt_bboxes',
        ),  # random or resample
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    # dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    # To DataContainer
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', 
        keys=['gt_bboxes','gt_labels', 'img'],
        meta_keys=['filename','img_shape','ori_shape','pad_shape',
            'scale','scale_factor','keep_ratio','lidar2img',
            'sample_idx','img_info','anno_info','pts_info',
            'img_norm_cfg', 'pad_fixed_size', 'pad_size_divisor',
            'sample_img_id','img_sample','points'],
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
        img_scale=(1920, 1280),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizeMultiViewImage', **img_norm_cfg),
            dict(
                type='PadMultiViewImage',
                size=[(1280, 1920),(1280, 1920),(1280, 1920),
                    (1280, 1920),(1280, 1920)]),
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
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img'])    
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
        load_panseg=False,  
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

evaluation = dict(interval=24, pipeline=eval_pipeline)
custom_imports = dict(
    imports=['boxinst_waymo.mmdet3d_jgf'],
    allow_failed_imports=False)
