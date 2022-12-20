import glob
import os.path as osp
import os
import shutil
import numpy as np
import json
from mmdet3d.core.bbox.structures import *
from mmcv.utils.progressbar import track_iter_progress
import pickle
import torch


def read_pcd(pcd):
    with open(pcd, 'rb') as f:
        content = f.read()
    num_points = None
    data = None
    while True:
        head, content = content.split(b'\n', 1)
        if head.startswith(b'POINTS'):
            num_points = int(head.split()[-1])
        elif head == b'DATA binary':
            data = content
            break
    assert num_points is not None, "wrong pcd format"
    assert data is not None, "wrong pcd format"
    assert len(data) >= num_points * 4 * 4, "wrong pcd format"
    data = data[:num_points * 4 * 4]
    return np.frombuffer(data, dtype=np.float32).reshape(num_points, 4).copy()


def process_m1_single(idx, pcd, ann, root_path, bin_path, gt_bin_path=None):
    xyzi = read_pcd(pcd)
    xyzi[..., -1] /= 255.0
    valid = np.isfinite(xyzi).all(axis=-1)
    xyzi = xyzi[valid, :]
    frame_anno = {
        'sample_idx': idx,
        'point_cloud': {'num_features': 4,
                        'path': osp.join(bin_path, f'{idx:07d}.bin')},
        'calib': {},
        'annos': {'name': [],
                  'bboxes': [],
                  'visibility': [],
                  'index': [],
                  'type_confidence': []}}
    xyzi.tofile(osp.join(root_path, frame_anno['point_cloud']['path']))
    bboxes = []
    with open(ann) as f:
        label = json.load(f)
    for obj_idx, obj in enumerate(label['labels']):
        ctr = obj['center']
        ctr = [ctr['x'], ctr['y'], ctr['z']]
        typ = obj['type']
        rot = obj['rotation']
        assert rot['pitch'] == rot['roll'] == 0

        yaw = rot['yaw']

        size = obj['size']
        size = [size['x'], size['y'], size['z']]

        visibility = obj['visibility'] if 'visibility' in obj else 255
        typconf = obj['type_confidence'] if 'type_confidence' in obj else 255

        bboxes.append(ctr + size + [yaw])

        frame_anno['annos']['name'].append(typ)
        frame_anno['annos']['visibility'].append(visibility)
        frame_anno['annos']['type_confidence'].append(typconf)
        frame_anno['annos']['index'].append(obj_idx)

    bboxes = np.array(bboxes, dtype=np.float32)

    bboxes = LiDARInstance3DBoxes(bboxes, origin=(0.5, 0.5, 0.5))

    frame_anno['annos']['name'] = np.array(frame_anno['annos']['name'])
    frame_anno['annos']['bboxes'] = bboxes.tensor.numpy()
    frame_anno['annos']['visibility'] = np.array(
        frame_anno['annos']['visibility'], dtype=np.uint8)
    frame_anno['annos']['type_confidence'] = np.array(
        frame_anno['annos']['type_confidence'], dtype=np.uint8)
    frame_anno['annos']['index'] = np.array(
        frame_anno['annos']['index'], dtype=np.uint16)

    frame_gt = {}
    if gt_bin_path is not None:
        ind = bboxes.points_in_boxes_part(
            torch.from_numpy(xyzi)[:, :3].cuda()).cpu().numpy()
        for obj_idx in range(len(bboxes)):
            gt_points = xyzi[ind == obj_idx].copy()
            box3d_lidar = frame_anno['annos']['bboxes'][obj_idx]
            gt_points[:, :3] -= box3d_lidar[None, :3]
            if len(gt_points) == 0:
                continue
            obj_name = frame_anno['annos']['name'][obj_idx]
            db_info = {
                'name': obj_name,
                'path': osp.join(gt_bin_path,
                                 f'{idx}_{obj_name}_{obj_idx}.bin'),
                'image_idx': idx,
                'gt_idx': obj_idx,
                'box3d_lidar': box3d_lidar,
                'num_points_in_gt': len(gt_points),
                'difficulty': 0,
            }
            if obj_name not in frame_gt:
                frame_gt[obj_name] = []
            frame_gt[obj_name].append(db_info)
            gt_points.tofile(osp.join(root_path, db_info['path']))

    return frame_anno, frame_gt


if __name__ == '__main__':
    data_root = 'data/m1'

    isval = lambda x: x.startswith('jiananyilu') or x.startswith(
        'jinggangaogaosu') or x.startswith('beihuandadao')

    if osp.isdir(osp.join(data_root, 'kitti_format')):
        shutil.rmtree(osp.join(data_root, 'kitti_format'))
    os.makedirs(osp.join(data_root, 'kitti_format/training/m1'), exist_ok=True)
    os.makedirs(osp.join(data_root, 'kitti_format/ImageSets'), exist_ok=True)
    os.makedirs(osp.join(data_root, 'kitti_format/m1_gt_database'),
                exist_ok=True)
    pcd_files = glob.glob(f'{data_root}/**/*.pcd', recursive=True)
    json_files = glob.glob(f'{data_root}/**/*.json', recursive=True)
    pcd_ids = {osp.splitext(osp.split(p)[-1])[0]: p for p in pcd_files}
    json_ids = {osp.splitext(osp.split(j)[-1])[0]: j for j in json_files}
    ids = list(set(pcd_ids.keys()).intersection(set(json_ids.keys())))
    ids.sort()
    val_idx = [i for i in range(len(ids)) if isval(ids[i])]
    train_idx = [i for i in range(len(ids)) if not isval(ids[i])]

    trainval = '\n'.join(ids)
    with open(osp.join(data_root, 'kitti_format/ImageSets/', 'trainval.txt'),
              'w') as f:
        f.write(trainval)
    train = '\n'.join([ids[i] for i in train_idx])
    with open(osp.join(data_root, 'kitti_format/ImageSets/', 'train.txt'),
              'w') as f:
        f.write(train)
    val = '\n'.join([ids[i] for i in val_idx])
    with open(osp.join(data_root, 'kitti_format/ImageSets/', 'val.txt'),
              'w') as f:
        f.write(val)

    annos = []
    gts = {}
    for idx, d_id in enumerate(track_iter_progress(ids)):
        frame_anno, frame_gt = process_m1_single(
            idx, pcd_ids[d_id], json_ids[d_id], f'{data_root}/kitti_format',
            bin_path='training/m1', gt_bin_path='m1_gt_database')
        annos.append(frame_anno)
        for k in frame_gt:
            if k not in gts:
                gts[k] = []
            gts[k].extend(frame_gt[k])

    annos_val_only = [annos[i] for i in val_idx]
    annos_train_only = [annos[i] for i in train_idx]

    gts_train_only = {}
    for k in gts:
        gts_train_only[k] = [gt for gt in gts[k] if
                             gt['image_idx'] in train_idx]

    with open(osp.join(data_root, 'kitti_format/m1_infos_trainval.pkl'),
              'wb') as f:
        pickle.dump(annos, f)
    with open(osp.join(data_root, 'kitti_format/m1_infos_train.pkl'),
              'wb') as f:
        pickle.dump(annos_train_only, f)
    with open(osp.join(data_root, 'kitti_format/m1_infos_val.pkl'), 'wb') as f:
        pickle.dump(annos_val_only, f)

    with open(osp.join(data_root, 'kitti_format/m1_dbinfos_trainval.pkl'),
              'wb') as f:
        pickle.dump(gts, f)
    with open(osp.join(data_root, 'kitti_format/m1_dbinfos_train.pkl'),
              'wb') as f:
        pickle.dump(gts_train_only, f)
