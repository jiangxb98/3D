dataset_type = 'WaymoDataset'

file_client_args = dict(
    backend='MINIO',
    bucket='ai-waymo-v1.4',
    endpoint='ossapi.cowarobot.cn:9000',
    secure=False)

datainfo_client_args = dict(
    backend='MONGODB',
    database='waymo',
    host='mongodb://root:root@172.16.110.100:27017/')

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    type='DataBaseSampler',
    info_path='training/dbinfos/',
    rate=1.0,
    filter=dict(
        Car=dict(difficulty={'$nin': [-1]},
                 num_points_in_gt={'$gte': 5}),
        Pedestrian=dict(difficulty={'$nin': [-1]},
                        num_points_in_gt={'$gte': 5}),
        Cyclist=dict(difficulty={'$nin': [-1]},
                     num_points_in_gt={'$gte': 5})),
    classes=class_names,
    sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    datainfo_client_args=datainfo_client_args,
    points_loader=[
        dict(
            type='LoadPoints',
            coord_type='LIDAR',
            file_client_args=file_client_args),
        dict(
            type='LoadSweeps',
            sweeps_num=1,
            coord_type='LIDAR',
            file_client_args=file_client_args)])

train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True),
    dict(type='ObjectSample', db_sampler=db_sampler),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(type='PointsRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='FilterBoxWithMinimumPointsCount', num_points=1),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
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
        type='LoadSweeps',
        sweeps_num=1,
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
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        info_path='training/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        # load one frame every five frames
        load_interval=1),
    val=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR'))

evaluation = dict(interval=24, pipeline=eval_pipeline)
