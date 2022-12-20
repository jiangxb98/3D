dataset_type = 'CowaDatasetFSD'

file_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-ruby-sync-70',
    endpoint='oss01-api.cowadns.com:30009',
    access_key='k3KWM6HxRRer1seb',
    secret_key='SqLxUWiIJauZ5PM8JSOXaoq36ucViPQK',
    secure=False)

datainfo_client_args = dict(
    backend='MONGODB',
    database='cowa3d-ruby-sync-70',
    host='mongodb://root:root@172.16.110.100:27017/')

class_names = ('big_vehicle',
               'pedestrian',
               'vehicle',
               'bicycle',
               'huge_vehicle',
               'tricycle',
               'cone',
               'barrier',
               'barrel')

class_names_mapping = (('big_vehicle', 'big_vehicle'),
                       ('pedestrian', 'pedestrian'),
                       ('vehicle', 'vehicle'),
                       ('bicycle', 'motorcycle_bicycle'),
                       ('huge_vehicle', 'big_vehicle'),
                       ('tricycle', 'tricycle'),
                       ('cone', 'barrier'),
                       ('barrier', 'barrier'),
                       ('barrel', 'barrier'))

mapped_class_names = ('big_vehicle',
                      'pedestrian',
                      'vehicle',
                      'motorcycle_bicycle',
                      'tricycle',
                      'barrier')

point_cloud_range = [-51.1, -76.7, -1.999, 102.3, 76.7, 3.999]
input_modality = dict(use_lidar=True, use_camera=False)
db_sampler = dict(
    type='DataBaseSampler',
    info_path='training/dbinfos/',
    rate=1.0,
    filter=dict(
        big_vehicle=dict(difficulty=0, num_points_in_gt={'$gte': 30}),
        motorcycle=dict(difficulty=0, num_points_in_gt={'$gte': 10}),
        pedestrian=dict(difficulty=0, num_points_in_gt={'$gte': 5}),
        vehicle=dict(difficulty=0, num_points_in_gt={'$gte': 10}),
        bicycle=dict(difficulty=0, num_points_in_gt={'$gte': 10}),
        huge_vehicle=dict(difficulty=0, num_points_in_gt={'$gte': 30}),
        tricycle=dict(difficulty=0, num_points_in_gt={'$gte': 10}),
        cone=dict(difficulty=0, num_points_in_gt={'$gte': 5}),
        barrier=dict(difficulty=0, num_points_in_gt={'$gte': 10}),
        barrel=dict(difficulty=0, num_points_in_gt={'$gte': 10})),
    classes=class_names,
    sample_groups=dict(big_vehicle=10,
                       pedestrian=15,
                       vehicle=15,
                       bicycle=15,
                       huge_vehicle=0,
                       tricycle=15,
                       cone=5,
                       barrier=5,
                       barrel=5),
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
        scale_ratio_range=[0.95, 1.05],
        translation_std=[1.0, 1.0, 0.333]),
    dict(type='PointsRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter',
         point_cloud_range=point_cloud_range),
    dict(type='FilterBoxWithMinimumPointsCount', num_points=1),
    dict(type='PointShuffle'),
    dict(type='LabelIDMap', map=class_names_mapping),
    dict(type='DefaultFormatBundle3D', class_names=mapped_class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True),
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
            dict(type='LabelIDMap',
                 map=class_names_mapping),
            dict(
                type='DefaultFormatBundle3D',
                class_names=mapped_class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        info_path='training/infos',
        datainfo_client_args=datainfo_client_args,
        sensor_type='ruby',
        sensor_filter=[0],
        pcd_limit_range=point_cloud_range,
        pipeline=train_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1),
    val=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        sensor_type='ruby',
        sensor_filter=[0],
        pcd_limit_range=point_cloud_range,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1),
    test=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        sensor_type='ruby',
        sensor_filter=[0],
        pcd_limit_range=point_cloud_range,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1))

evaluation = dict(metric='cowa', interval=1,
                  class_names_mapping=class_names_mapping)

