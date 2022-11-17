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
semseg_names = ['Car', 'Truck', 'Bus', 'Motorcyclist', 'Bicyclist', 'Pedestrian', \
    'Sign', 'Traffic' 'Light', 'Pole', 'Construction' 'Cone', 'Bicycle', \
    'Motorcycle', 'Building', 'Vegetation', 'Tree' 'Trunk', 'Curb', \
    'Road', 'Lane Marker', 'Walkable', 'Sidewalk', 'Other Ground', \
    'Other Vehicle', 'Undefined']
panseg_names = ['Car', 'Bus', 'Truck', 'Other Large Vehicle', 'Trailer',\
    'Ego Vehicle', 'Motorcycle', 'Bicycle', 'Pedestrian', 'Cyclist', \
    'Motorcyclist', 'Ground Animal', 'Bird', 'Pole', 'Sign', 'Traffic Light',\
    'Construction Cone', 'Pedestrian Object', 'Building', 'Road', 'Sidewalk',\
    'Road Marker', 'Lane Marker', 'Vegetation', 'Sky', 'Ground', 'Static', 'Dynamic']

point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=True)

db_sampler = dict(
    type='DataBaseSampler',
    info_path='training/dbinfos/',
    rate=1.0,
    filter=dict(
        Car=dict(difficulty={'$nin': [-1]},
                 num_points_in_gt={'$gte': 5}),
        Pedestrian=dict(difficulty={'$nin': [-1]},
                        num_points_in_gt={'$gte': 10}),
        Cyclist=dict(difficulty={'$nin': [-1]},
                     num_points_in_gt={'$gte': 10})),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    datainfo_client_args=datainfo_client_args,
    points_loader=[
        dict(
            type='LoadPoints',
            coord_type='LIDAR',
            file_client_args=file_client_args)])

train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    # dict(
    #     type='LoadImages',
    #     file_client_args=file_client_args,),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_mask_3d=True,
        with_seg_3d=True,
        file_client_args=file_client_args,
        ),
    dict(
        type='LoadAnnos',
        with_bbox=True,
        with_label=True,
        with_mask=True,
        with_seg=True,
        file_client_args=file_client_args
    ),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='PointsRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='FilterBoxWithMinimumPointsCount', num_points=1),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    # dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
    # DataContainer
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', \
        'pts_semantic_mask', 'pts_instance_mask', 'gt_bboxes','gt_labels'],
    )
]

test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter',
                 point_cloud_range=point_cloud_range),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]

data = dict(
    samples_per_gpu=4,
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
        semseg_classes=semseg_names,
        semseg_info_path='training/infos',  # semantic and instance same path
        # 2d segmentation
        load_panseg=False,  
        panseg_classes = panseg_names,
        panseg_info_path='training/infos',  # semantic and instance same path
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=True,
        semseg_classes=semseg_names,
        semseg_info_path='validation/infos',
        # 2d segmentation
        load_panseg=False,  
        panseg_classes = panseg_names,
        panseg_info_path='validation/infos',),
    test=dict(
        type=dataset_type,
        info_path='test/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        # 3d segmentation
        load_semseg=True,
        semseg_classes=semseg_names,
        semseg_info_path='test/infos',
        # 2d segmentation
        load_panseg=False,  
        panseg_classes = panseg_names,
        panseg_info_path='test/infos',))        

evaluation = dict(interval=24, pipeline=eval_pipeline)
custom_imports = dict(
    imports=['start_dataset_learn.mmdet3d_cowa'],
    allow_failed_imports=False)
