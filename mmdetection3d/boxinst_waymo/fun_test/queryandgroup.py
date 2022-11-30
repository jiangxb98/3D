import numpy as np
from io import BytesIO
import minio
import torch
import mmcv
import pymongo

minio_cfg = dict(
    endpoint='ossapi.cowarobot.cn:9000',
    access_key='abcdef',
    secret_key='12345678',
    region='shjd-oss',
    secure=False,
    )
bucket = 'ai-waymo-v1.4'
object_name = 'training/lidar/0000000'
img_name = 'training/image/0000100/0'

mongo_cfg = dict(
    host="mongodb://root:root@172.16.110.100:27017/"
)
database='ai-waymo-v1_4'

_client = pymongo.MongoClient(**mongo_cfg)[database]
ret = _client.get_collection('training/infos').find_one(100)

def read_img_from_minio(bucket, img_name):
    client = minio.Minio(**minio_cfg)
    if not client.bucket_exists(bucket):
        return None
    pts_bytes = client.get_object(bucket, img_name).read()
    # not need loader, use mmcv.imfrombytes
    img = mmcv.imfrombytes(pts_bytes, flag='color', channel_order='rgb')
    import tensorflow as tf
    img_2 = tf.image.decode_jpeg(pts_bytes) # both two function have the same size (1280,1920,3) but have tiny distinction
    return img, img_2

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
img, img2 = read_img_from_minio(bucket=bucket, img_name=img_name)

"""
超参设置：
    查询半径: 0.1
    查询点数: 10
"""

# 1. 过滤掉没有idx相机投影的点，得到索引值
sample_idx = 0
mask = (points[:,6]==0) | (points[:,7]==0)  # 真值列表
mask_id = np.where(mask)[0]    # 全局索引值
in_img_points = points[mask]

# 2. 得到在2D bbox内的点

gt_bboxes = ret['annos']['bbox'][sample_idx]
gt_mask = np.array([False for _ in range(mask_id.shape[0])])
# 第一步过滤——针对所有的gt bbox进行筛选
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
# 得到id全局索引值
gt_mask_id = mask_id[gt_mask]
# 得到所有的值
in_gt_bboxes_points = in_img_points[gt_mask]

# 3. 对上述的点（在2d bboxes的点）进行球查询
from mmcv.ops.ball_query import ball_query
# to tensor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

xyz = torch.from_numpy(in_gt_bboxes_points[np.newaxis,:,0:3]).contiguous()
center_xyz = torch.from_numpy(in_gt_bboxes_points[np.newaxis,:,0:3]).contiguous()
xyz = xyz.to(device)
center_xyz = center_xyz.to(device)
min_radius = 0.
max_radius = 0.1
sample_num = 5
# 返回基于in_gt_bboxes_points的id索引值
# idx(1,7243,10)
idx = ball_query(
                min_radius,
                max_radius,
                sample_num,
                xyz,
                center_xyz).to(device)
# 转numpy
idx = idx.cpu().numpy().squeeze()
idx.astype('i2')  # int16

# 4. 第二步过滤——过滤掉查询不足5的点
idx_mask = np.array([True if len(np.unique(i))==sample_num else False for i in idx])
new_idx = idx[idx_mask]  # 过滤掉不足5之后的邻居点(N, 5), 这个5是指向in_gt_bboxes_points的局部索引
new_idx_neg = idx[~idx_mask]  # 小于5的点局部索引
ball_query_idx = np.zeros((new_idx.shape))  # 邻居点信息(N,5)，全局id
ball_query_idx_neg = np.zeros((new_idx_neg.shape))  # 负样本邻居点信息(N,5)，全局id
for i, ball_query_ in enumerate(new_idx):
    ball_query_idx[i] = gt_mask_id[ball_query_]
for i, ball_query_ in enumerate(new_idx_neg):
    ball_query_idx_neg[i] = gt_mask_id[ball_query_]
ball_query_id = gt_mask_id[idx_mask]  # 中心点的全局索引值(N, )
ball_query_points = in_gt_bboxes_points[idx_mask]  # 中心点坐标等信息(N, 12)
ball_query_id_neg = gt_mask_id[~idx_mask]
ball_query_points_neg = in_gt_bboxes_points[~idx_mask]

# 5. 将得到的点云映射到原始图片，如果维度886x1920，那么也是np.ones((886,1290))
ori_image_points = torch.zeros((1280, 1920, 3),dtype=torch.float)  # np.ones()*np.inf ???
ori_image_points_mask = torch.zeros((1280,1920))  # 是否有值的mask
for point in ball_query_points:
    if point[6] == sample_idx:
        x_0 = point[8]
        y_0 = point[10]
        ori_image_points[int(y_0), int(x_0)] = torch.tensor([point[0] ,point[1], point[2]])
        ori_image_points_mask[int(y_0), int(x_0)] = 1
    if point[7] == sample_idx:
        x_1 = point[9]
        y_1 = point[11]
        ori_image_points[int(y_1), int(x_1)] = torch.tensor([point[0] ,point[1], point[2]])
        ori_image_points_mask[int(y_1), int(x_1)] = 1
# 6.1 降采样，方法一直接降采样
# image_points = (320,480,3)
start = 2
stride = 4
image_points = ori_image_points[start::stride, start::stride, :]
image_points_mask = ori_image_points_mask[start::stride, start::stride]
# 6.2 avgpool，方法二使用平均池化
import torch.nn.functional as F
ori_image_points = np.expand_dims(ori_image_points.permute(2,0,1), axis=0)
ori_image_points = torch.from_numpy(ori_image_points)
# downsampled_images=(1,320,480)
downsampled_images = F.avg_pool2d(ori_image_points.float(), kernel_size=stride, stride=stride, padding=0)

# 6.3 当大小不一致时，需要pad到大小，参考其他代码

# 7. 对于pairwise loss，我们计算每一对的距离d(1,8,320,480)，然后用exp(d)来做损失
pairwise_size = 3
pairwise_dilation = 2


# 可视化
import cv2
img.astype('uint8')
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img_bbox = img
for gt_bbox in gt_bboxes:
    
    cv2.rectangle(img, (int(gt_bbox[0]),int(gt_bbox[1])), (int(gt_bbox[2]),int(gt_bbox[3])), (0,255,0), 2)

# cv2.imwrite('out_plot.jpeg', img)
# points, in_img_points, in_gt_bboxes_points, ball_query_points
for point in ball_query_points:
    if point[6] == sample_idx:
        x_0 = int(point[8])
        y_0 = int(point[10])
        cv2.circle(img, (x_0, y_0), 1, (0, 255, 0), 1)
    if point[7] == sample_idx:
        x_1 = int(point[9])
        y_1 = int(point[11])
        cv2.circle(img, (x_1, y_1), 1, (0, 255, 0), 1)
cv2.imwrite('out_plot_in_ball_query_points.jpeg', img)
print('finish!!!!!!!!')
