import numpy as np
from io import BytesIO
import minio


minio_cfg = dict(
    endpoint='ossapi.cowarobot.cn:9000',
    access_key='abcdef',
    secret_key='12345678',
    region='shjd-oss',
    secure=False,
    )

bucket = 'ai-waymo-v1.4'
object_name = 'training/lidar/0000000'

def read_points_from_minio(bucket, object_name):
    client = minio.Minio(**minio_cfg)
    if not client.bucket_exists(bucket):
        return None
    pts_bytes = client.get_object(bucket, object_name).read()
    points = np.load(BytesIO(pts_bytes))
    return np.stack([points['x'].astype('f4'),
                        points['y'].astype('f4'),
                        points['z'].astype('f4'),
                        #np.tanh(points['intensity'].astype('f4')),
                        points['intensity'].astype('f4'),
                        points['elongation'].astype('f4'),
                        points['lidar_idx'].astype('i2'),
                        points['cam_idx_0'].astype('i2'),
                        points['cam_idx_1'].astype('i2'),
                        points['cam_column_0'].astype('i2'),
                        points['cam_column_1'].astype('i2'),
                        points['cam_row_0'].astype('i2'),
                        points['cam_row_1'].astype('i2'),
                        ], axis=-1)
# x,y,z,intensity,elongation,lidar_idx,cam_idx_0,cam_idx_1,c_0,c_1,r_0,r_1
# shape=(N,12)
points = read_points_from_minio(bucket=bucket,object_name=object_name)

# 1. 过滤掉没有idx相机投影的点，得到索引值
sample_idx = 0
mask = (points[:,6]==0) | (points[:,7]==0)  # 真值列表
mask_id = np.where(mask)[0]    # 索引值
in_img_points = points[mask]
# 2. 得到在2D bbox内的点
x1, y1, x2, y2 = 0,0,1000,1000
gt_bboxes = [[x1, y1, x2, y2]]
gt_mask = np.array([False for _ in range(mask_id.shape[0])])
# 针对所有的gt bbox进行筛选
for gt_bbox in gt_bboxes:
    # if 0 cam 8,10（列，行）
    gt_mask_0 = (((in_img_points[:,8] > gt_bbox[0]) & (in_img_points[:,8] < gt_bbox[2]))  & 
                ((in_img_points[:,10] > gt_bbox[1]) & (in_img_points[:,10] < gt_bbox[3])) &
                (in_img_points[:,6]==0))
    # if 1 cam 9,11
    gt_mask_1 = (((in_img_points[:,9] > gt_bbox[0]) & (in_img_points[:,9] < gt_bbox[2]))  & 
                ((in_img_points[:,11] > gt_bbox[1]) & (in_img_points[:,11] < gt_bbox[3])) &
                (in_img_points[:,7]==0))
    gt_mask = gt_mask_0 | gt_mask_1 | gt_mask
# 得到id索引值
gt_mask_id = mask_id[gt_mask]
# 得到所有的值
in_gt_bboxes_points = in_img_points[gt_mask]
# 3. 对上述的点进行球查询
from mmcv.ops.ball_query import ball_query
# to tensor
import torch
xyz = torch.from_numpy(in_gt_bboxes_points[np.newaxis,:,0:3]).contiguous()
center_xyz = torch.from_numpy(in_gt_bboxes_points[np.newaxis,:,0:3]).contiguous()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
xyz.to(device)
center_xyz.to(device)
# 返回基于in_gt_bboxes_points的id索引值
idx = ball_query(
                0.,
                0.1,
                10,
                xyz,
                center_xyz)
