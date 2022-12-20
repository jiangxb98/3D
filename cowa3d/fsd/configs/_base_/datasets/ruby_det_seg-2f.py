import sys
sys.path.insert(0, 'fsd/configs/_base_/datasets')
from ruby_split import train_parts, val_parts

dataset_type = 'COWADatasetDetSegFSD'

sensor_signals = ['x', 'y', 'z', 'intensity']
sensors = [0] #[0, 1, 2]
point_cloud_range = [-124.999, -124.999, -1.999, 124.999, 124.999, 3.999]
input_modality = dict(use_lidar=True, use_camera=False)

det_file_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-ruby-sync-70',
    endpoint='oss01-api.cowadns.com:30009',
    # proxy='socks5h://localhost:9999/',
    access_key='k3KWM6HxRRer1seb',
    secret_key='SqLxUWiIJauZ5PM8JSOXaoq36ucViPQK',
    secure=False)

det_vel_file_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-ruby-vel',
    endpoint='oss01-api.cowadns.com:30009',
    # proxy='socks5h://localhost:9999/',
    secure=False)


# det_datainfo_client_args = dict(
#     backend='MONGODB',
#     database='cowa3d-ruby-sync-70',
#     host='mongodb://root:root@172.16.110.100:27017/')
det_datainfo_client_args = dict(
    backend='MONGODB',
    database='cowa3d-ruby-vel',
    host='mongodb://root:root@172.16.110.100:27017/')


seg_datainfo_client_args = dict(
    backend='MONGODB',
    database='cowa3d-base',
    host='mongodb://root:root@172.16.110.100:27017/')

seg_pts_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-base',
    endpoint='oss01-api.cowadns.com:30009',
    # proxy='socks5h://localhost:9999/',
    secure=False)

seg_labels_client_args = dict(
    backend='MINIO',
    bucket='cowa3d-seg',
    endpoint='oss01-api.cowadns.com:30009',
    # proxy='socks5h://localhost:9999/',
    secure=False)


det_class_names = ('big_vehicle',
                    'pedestrian',
                    'vehicle',
                    'bicycle',
                    'huge_vehicle',
                    'tricycle',
                    'cone',
                    'barrier',
                    'barrel')

det_class_names_mapping = (('big_vehicle', 'big_vehicle'),
                       ('pedestrian', 'pedestrian'),
                       ('vehicle', 'vehicle'),
                       ('bicycle', 'motorcycle_bicycle'),
                       ('huge_vehicle', 'big_vehicle'),
                       ('tricycle', 'tricycle'),
                       ('cone', 'barrier'),
                       ('barrier', 'barrier'),
                       ('barrel', 'barrier'))

det_mapped_class_names = ('big_vehicle',
                      'pedestrian',
                      'vehicle',
                      'motorcycle_bicycle',
                      'tricycle',
                      'barrier')

seg_class_names = (
    'unlabeled', 'road-plane', 'curb', 'other-ground', 'terrain',
    'vegetation', 'pillars', 'framework', 'building', 'fence',
    'traffic-sign', 'other-structure', 'noise', 'road-users', 'road-block')
seg_map_labels = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 6, 10, 11, 12, 13, 14]

detcat2seglabel={'big_vehicle':13,
                 'pedestrian':13,
                 'vehicle':13,
                 'bicycle':13,
                 'huge_vehicle':13,
                 'tricycle':13,
                 'cone':14,
                 'barrier':14,
                 'barrel':14}


db_sampler = dict(
    type='DataBaseSamplerWithSeg',
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
    classes=det_class_names,
    sample_groups=dict(big_vehicle=10,
                       pedestrian=15,
                       vehicle=15,
                       bicycle=15,
                       huge_vehicle=0,
                       tricycle=15,
                       cone=5,
                       barrier=5,
                       barrel=5),
    datainfo_client_args=det_datainfo_client_args,
    cat2seglabel=detcat2seglabel,
    points_loader=[
        dict(
            type='LoadPoints',
            coord_type='LIDAR',
            file_client_args=det_vel_file_client_args),
        dict(
            type='LoadSweeps',
            sweeps_num=1,
            coord_type='LIDAR',
            random_choose=False,
            with_sweep_ind=True,
            file_client_args=det_vel_file_client_args)
        ])

train_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=seg_pts_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        random_choose=False,
        with_sweep_ind=True,
        file_client_args=seg_pts_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d=True,
        file_client_args=seg_labels_client_args),
    dict(
        type='LoadFakeSweepsAnnos3D',
        sweeps_num=1,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True),
    dict(type='ObjectSampleWithSeg', db_sampler=db_sampler),
    dict(
        type='PointSegClassMapping',
        valid_cat_ids=tuple(range(1, len(seg_class_names))),
        max_cat_id=len(seg_class_names)-1),
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
    dict(type='LabelIDMap', map=det_class_names_mapping),
    dict(type='DefaultFormatBundle3D', class_names=det_mapped_class_names),
    dict(type='Collect3D',
        # keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
         keys=['points', 'gt_bboxes_3d', 'gt_labels_3d', 'pts_semantic_mask'])
]
test_pipeline = [
    dict(
        type='LoadPoints',
        coord_type='LIDAR',
        file_client_args=seg_pts_client_args),
    dict(
        type='LoadSweeps',
        sweeps_num=1,
        coord_type='LIDAR',
        random_choose=False,
        with_sweep_ind=True,
        file_client_args=seg_pts_client_args),
    dict(
        type='LoadAnnos3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_seg_3d=True,
        file_client_args=seg_labels_client_args),
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
                 map=det_class_names_mapping),
            dict(
                type='DefaultFormatBundle3D',
                class_names=det_mapped_class_names,
                with_label=False),
            dict(type='Collect3D',
                 keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                       'pts_semantic_mask'])
            # dict(type='Collect3D',
            #      keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        det_info_path='training/infos',
        seg_info_path='clips',
        seg_clip_parts=train_parts,
        det_datainfo_client_args=det_datainfo_client_args,
        seg_datainfo_client_args=seg_datainfo_client_args,
        sensor_filter=sensors,
        sensor_signals=sensor_signals,
        pcd_limit_range=point_cloud_range,
        pipeline=train_pipeline,
        modality=input_modality,
        det_classes=det_class_names,
        seg_classes=seg_class_names,
        seg_map_labels=seg_map_labels,
        test_mode=False,
        box_type_3d='LiDAR',
        load_interval=1),
    val=dict(
        type=dataset_type,
        det_info_path='validation/infos',
        seg_info_path='clips',
        seg_clip_parts=val_parts,
        det_datainfo_client_args=det_datainfo_client_args,
        seg_datainfo_client_args=seg_datainfo_client_args,
        sensor_filter=sensors,
        sensor_signals=sensor_signals,
        pcd_limit_range=point_cloud_range,
        pipeline=test_pipeline,
        modality=input_modality,
        det_classes=det_class_names,
        seg_classes=seg_class_names,
        seg_map_labels=seg_map_labels,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1),
    test=dict(
        type=dataset_type,
        det_info_path='validation/infos',
        seg_info_path='clips',
        seg_clip_parts=val_parts,
        det_datainfo_client_args=det_datainfo_client_args,
        seg_datainfo_client_args=seg_datainfo_client_args,
        sensor_filter=sensors,
        sensor_signals=sensor_signals,
        pcd_limit_range=point_cloud_range,
        pipeline=test_pipeline,
        modality=input_modality,
        det_classes=det_class_names,
        seg_classes=seg_class_names,
        seg_map_labels=seg_map_labels,
        test_mode=True,
        box_type_3d='LiDAR',
        load_interval=1))

evaluation = dict(metric='cowa', interval=1,
                  class_names_mapping=det_class_names_mapping)

