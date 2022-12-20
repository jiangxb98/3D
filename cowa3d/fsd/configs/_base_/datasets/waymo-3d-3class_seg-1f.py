dataset_type = 'WaymoDatasetDetSeg'

file_client_args = dict(
    backend='MINIO',
    bucket='waymo',
    endpoint='oss01-api.cowadns.com:30009',
    secure=False)

datainfo_client_args = dict(
    backend='MONGODB',
    database='waymo',
    host='mongodb://root:root@172.16.110.100:27017/')

class_names = ['Car', 'Pedestrian', 'Cyclist']
point_cloud_range = [-74.88, -74.88, -2, 74.88, 74.88, 4]
## seg: 23 classes; idx 0 ~ 22
semantic_class_names = ['TYPE_UNDEFINED', 'TYPE_CAR', 'TYPE_TRUCK', 'TYPE_BUS',
'TYPE_OTHER_VEHICLE', 'TYPE_MOTORCYCLIST', 'TYPE_BICYCLIST', 'TYPE_PEDESTRIAN',
'TYPE_SIGN', 'TYPE_TRAFFIC_LIGHT', 'TYPE_POLE', 'TYPE_CONSTRUCTION_CONE',
'TYPE_BICYCLE', 'TYPE_MOTORCYCLE', 'TYPE_BUILDING', 'TYPE_VEGETATION', 
'TYPE_TREE_TRUNK', 'TYPE_CURB', 'TYPE_ROAD', 'TYPE_LANE_MARKER', 
'TYPE_OTHER_GROUND', 'TYPE_WALKABLE', 'TYPE_SIDEWALK']
seg_label_mapping = {0:10, 1:5, 2:5, 3:5, 4:5, 5:5, 6:5, 7:5, 8:6, 9:6, 10:7,
    11:9, 12:5, 13:5, 14:4, 15:8, 16:7, 17:1, 18:0, 19:2, 20:3, 21:3, 22:3}
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
    sample_groups=dict(Car=5, Pedestrian=5, Cyclist=5),
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
train_pipeline_with_seg = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d=True,
        file_client_args=file_client_args),
    # dict(
    #     type='PointSegClassMapping',
    #     valid_cat_ids=tuple(range(1, len(semantic_class_names))),
    #     max_cat_id=len(semantic_class_names)-1),
    dict(
        type='PointSegClassMerge',
        seg_label_mapping=seg_label_mapping),
    # dict(type='ObjectSample', db_sampler=db_sampler),
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
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=file_client_args),
    # dict(
    #     type='LoadAnnos3D',
    #     with_bbox_3d=True,
    #     with_label_3d=True),
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
            # dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
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
    samples_per_gpu=3,
    workers_per_gpu=3,
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
        load_interval=1,
        with_semseg=True,
        semseg_classes=semantic_class_names,
        semseg_info_path='tools/data_converter/semseg_label_training_info.csv',
        semseg_pipeline=train_pipeline_with_seg,
        only_load_seg_frames=True),
    val=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        with_semseg=True,
        semseg_classes=semantic_class_names,
        semseg_info_path='tools/data_converter/semseg_label_validation_info.csv',
        semseg_pipeline=test_pipeline,
        only_load_seg_frames=False),
    test=dict(
        type=dataset_type,
        info_path='validation/infos',
        datainfo_client_args=datainfo_client_args,
        file_client_args=file_client_args,
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        with_semseg=True,
        semseg_classes=semantic_class_names,
        semseg_info_path='tools/data_converter/semseg_label_validation_info.csv',
        semseg_pipeline=test_pipeline,
        only_load_seg_frames=False))

evaluation = dict(interval=24, pipeline=eval_pipeline)
