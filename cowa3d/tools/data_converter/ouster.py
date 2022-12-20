import glob
import json
import numpy as np
import os
import os.path as osp
import shutil
import tqdm
import pickle as pkl
from mmdet3d.core.bbox.structures import *
from multiprocessing import Pool
import torch
from scipy.spatial.transform import Rotation as R


class TF:
    def __init__(self, r, t):
        self.r = r
        self.t = t

    @classmethod
    def fromXYZYPR(cls, xyzypr):
        return cls(R.from_euler('ZYX', xyzypr[3:], degrees=False),
                   np.array(xyzypr[:3], dtype=float))

    def toXYZYPR(self):
        return tuple(self.t) + tuple(self.r.as_euler('ZYX', degrees=False))

    @classmethod
    def fromCRPose(cls, pose):
        pose = pose[1]
        return cls(R.from_quat(pose[1]),
                   np.array(pose[0], dtype=float))

    def to(self, other):
        assert isinstance(other, TF)
        inv = other.r.inv()
        return TF(inv * self.r, inv.apply(self.t - other.t))

    def on(self, other):
        assert isinstance(other, TF)
        return TF(other.r * self.r, other.r.apply(self.t) + other.t)

    def dump_kitti(self):
        return np.concatenate((self.r.as_matrix(), self.t[:, None]),
                              axis=-1).astype(float).tolist()

    @classmethod
    def from_kitti(cls, mat_list):
        mat3x4 = np.array(mat_list).reshape((3, 4))
        r = R.from_matrix(mat3x4[:, :3])
        t = mat3x4[:, 3]
        return cls(r, t)

    def __getstate__(self):
        return dict(r=self.r.as_quat(), t=self.t.copy())

    def __setstate__(self, state):
        self.r = R.from_quat(state['r'])
        self.t = state['t']


class seedOriginDataMapper:
    def __init__(self, output_dir):
        seed_data_mappings = glob.glob(
            osp.join(output_dir, '*/for_seed/yg_inverse.json'))
        self.seed_data_mappings = dict()
        for mapping in seed_data_mappings:
            with open(mapping) as f:
                self.seed_data_mappings.update(json.load(f)['map'])

    def __getitem__(self, x):
        return self.seed_data_mappings[x]


class originDataSeqPrevMapper:
    def __init__(self, output_dir):
        if osp.exists('dataSeqPrevMapping.pkl'):
            with open('dataSeqPrevMapping.pkl', 'rb') as f:
                self.clip_pos_mapper = pkl.load(f)
        else:
            clip_mappings = glob.glob(
                osp.join(output_dir, '*/tag/*.clip'))
            self.clip_pos_mapper = dict()
            for mapping in clip_mappings:
                with open(mapping) as f:
                    clip = json.load(f)['sequence']
                    for idx, clip_frames in enumerate(clip):
                        for frame in clip_frames:
                            self.clip_pos_mapper[frame] = (clip, idx)
            with open('dataSeqPrevMapping.pkl', 'wb') as f:
                pkl.dump(self.clip_pos_mapper, f)

    def __getitem__(self, frame):
        n = 1
        if isinstance(frame, tuple):
            frame, n = frame
        clip, idx = self.clip_pos_mapper[frame]
        ret = []
        raw = clip[idx - n + 1:idx + 1][::-1]
        chn_idx = raw[0].index(frame)
        for r in raw:
            ret.append((r[chn_idx], r[:chn_idx] + r[chn_idx + 1:]))
        return ret


class originDataIndexMapper:
    def __init__(self, output_dir) -> None:
        self.output_dir = output_dir
        if osp.exists('dataIndexMapping.pkl'):
            with open('dataIndexMapping.pkl', 'rb') as f:
                self.data_index = pkl.load(f)
        else:
            self.data_index = dict()
            indices = glob.iglob(osp.join(output_dir, '*/tag/*.json'))
            indices = [*tqdm.tqdm(indices)]
            p = Pool()
            indices = p.map(self.load_tag_cowa, tqdm.tqdm(indices))
            for frame_idx, index in tqdm.tqdm(indices):
                self.data_index[frame_idx] = index
            with open('dataIndexMapping.pkl', 'wb') as f:
                pkl.dump(self.data_index, f)

    def load_tag_cowa(self, index_json):
        frame_idx = osp.splitext(osp.split(index_json)[-1])[0]
        with open(index_json) as f:
            tag = json.load(f)
        tag["tf"] = TF.fromXYZYPR(
            tag["tf"]["translation"] + tag["tf"]["rotation"])
        tag["pose"] = TF.fromXYZYPR([tag["pose"]["x"], tag["pose"]["y"], tag["pose"]
                                    ["z"], tag["pose"]["yaw"], tag["pose"]["pitch"], tag["pose"]["roll"]])
        return frame_idx, tag

    def __getitem__(self, frame):
        frame_idx = osp.splitext(osp.split(frame)[-1])[0]
        return self.data_index[frame_idx]


class originDataFullDirMapper:
    def __init__(self, output_dir):
        if osp.exists('fullDataMapping.pkl'):
            with open('fullDataMapping.pkl', 'rb') as f:
                self.full_dir = pkl.load(f)
        else:
            npys = glob.iglob(osp.join(output_dir, '*/data/*.npy'))
            jpgs = glob.iglob(osp.join(output_dir, '*/data/*.jpg'))
            self.full_dir = dict()
            for npy in tqdm.tqdm(npys):
                self.full_dir[osp.split(npy)[-1]] = npy
            for jpg in tqdm.tqdm(jpgs):
                self.full_dir[osp.split(jpg)[-1]] = jpg
            with open('fullDataMapping.pkl', 'wb') as f:
                pkl.dump(self.full_dir, f)

    def __getitem__(self, x):
        return self.full_dir[x]


class saveConfig:
    def __init__(self, label_map, root_path, point_cloud_path, gt_database_path):
        self.label_map = label_map
        self.root_path = root_path
        self.point_cloud_path = point_cloud_path
        self.gt_database_path = gt_database_path


class labelInfoMapper:
    NUM_PREV = 5

    def __init__(self, seed_data_mapper, clip_mapper, origin_data_mapper, origin_index_mapper, save_cfg):
        self.seed_data_mapper = seed_data_mapper
        self.clip_mapper = clip_mapper
        self.origin_data_mapper = origin_data_mapper
        self.origin_index_mapper = origin_index_mapper
        self.save_cfg = save_cfg

    def load_label(self, label):
        with open(label) as f:
            label = json.load(f)
        data_seq = int(label['taskJobContinuousNO'])
        dataset = label['datasetName']
        if (data_seq + 1) % 5 != 0:
            return None
        result = {'seq': data_seq, 'seed_label': label, 'dataset': dataset,
                  'merged': {}, 'frame_annos': {}}
        self.load_pointcloud(result)
        self.load_sweeps(result)
        self.load_objects(result)
        self.segment_objects(result)
        return result

    def load_sweeps(self, result):
        frame = result['merged']['npy']
        frame_index = result['merged']['index']
        frame_world_tf = frame_index['tf'].on(frame_index['pose'])
        prev_frames = [x[0]
                       for x in self.clip_mapper[frame, self.NUM_PREV][1:]]
        for s in result['merged']['sensors']:
            result['frame_annos'][s]['sweeps'] = []
        for prev_frame in prev_frames:
            prev_index = self.origin_index_mapper[prev_frame]
            prev_ts = prev_index['timestamp']
            prev_world_tf = prev_index['tf'].on(prev_index['pose'])
            prev_tf_to_current = prev_world_tf.to(frame_world_tf)
            sensor2lidar_rotation = prev_tf_to_current.r.as_matrix()
            sensor2lidar_translation = prev_tf_to_current.t
            lidar_fdir = self.origin_data_mapper[prev_frame]
            pc = np.load(lidar_fdir)
            for s in result['merged']['sensors']:
                pc_s = pc[pc['sensor'] == s]
                pc_s = np.stack([pc_s['x'], pc_s['y'], pc_s['z'], pc_s['intensity'] /
                                255, pc_s['value'] / 255], axis=-1).astype(np.float32)
                sample_idx = self.get_data_idx(prev_frame, s)
                sweep = {
                    'sample_idx': sample_idx,
                    'point_cloud': {'num_features': 5,
                                    'path': osp.join(self.save_cfg.point_cloud_path, f'{sample_idx}.bin')},
                    'calib': {},
                    'sensor': s,
                    'timestamp': prev_ts,
                    'dataset': result['dataset'],
                    'sensor2lidar_rotation': sensor2lidar_rotation,
                    'sensor2lidar_translation': sensor2lidar_translation
                }
                result['frame_annos'][s]['sweeps'].append(sweep)
                pc_s.tofile(osp.join(self.save_cfg.root_path,
                            sweep['point_cloud']['path']))
        return result

    def load_pointcloud(self, result):
        label = result['seed_label']
        lidar_seed_fname = osp.join(
            label['folderName'], label['originStorageName'])
        lidar_finfo = self.seed_data_mapper[lidar_seed_fname]
        result['merged']['npy'] = lidar_finfo['from'][0]
        result['merged']['sensors'] = lidar_finfo['sensors']
        result['merged']['index'] = self.origin_index_mapper[result['merged']['npy']]
        lidar_fdir = self.origin_data_mapper[result['merged']['npy']]
        pc = np.load(lidar_fdir)
        point_cloud = {}
        for s in result['merged']['sensors']:
            pc_s = pc[pc['sensor'] == s]
            point_cloud[s] = np.stack([pc_s['x'], pc_s['y'], pc_s['z'], pc_s['intensity'] / 255,
                                      pc_s['value'] / 255, np.full_like(pc_s['x'], s)], axis=-1).astype(np.float32)
        result['merged']['point_cloud'] = np.concatenate(
            [point_cloud[s] for s in result['merged']['sensors']], axis=0)
        for s in result['merged']['sensors']:
            sample_idx = self.get_data_idx(result['merged']['npy'], s)
            frame_anno = {
                'sample_idx': sample_idx,
                'point_cloud': {'num_features': 5,
                                'path': osp.join(self.save_cfg.point_cloud_path, f'{sample_idx}.bin')},
                'calib': {},
                'sensor': s,
                'timestamp': result['merged']['index']['timestamp'],
                'dataset': result['dataset'],
                'annos': {'name': [],
                          'bboxes': [],
                          'num_points': [],
                          'index': []}}
            result['frame_annos'][s] = frame_anno
            point_cloud[s][:, :-1].tofile(
                osp.join(self.save_cfg.root_path, frame_anno['point_cloud']['path']))

    def load_objects(self, result):
        label = result['seed_label']
        objects = label['LabelDetail']['objects']
        result['merged']['name'] = [self.save_cfg.label_map[o['labelName']]
                                    for o in objects]
        result['merged']['bboxes'] = []
        rot_dict = {'+x': 0, '-x': np.pi,
                    '+y': 0.5 * np.pi, '-y': -0.5 * np.pi}
        for obj in objects:
            obj = obj['feature']
            dir = obj['frontFaceDirection']
            if dir not in rot_dict:
                print(f'warning! box with dir = {dir} found!')
                dir = '+x'
            ctr = [obj['center']['x'], obj['center']['y'],
                   obj['center']['z'] - float(obj['depth']) / 2]
            yaw = [limit_period(obj['rotation']['z'] +
                                rot_dict[dir], period=2 * np.pi)]
            if 'x' in dir:
                size = [float(obj['width']), float(
                    obj['height']), min(float(obj['depth']), 4.0)]
            else:
                size = [float(obj['height']), float(
                    obj['width']), min(float(obj['depth']), 4.0)]
            result['merged']['bboxes'].append(ctr + size + yaw)
        result['merged']['bboxes'] = np.array(
            result['merged']['bboxes'], dtype=np.float32)

    @staticmethod
    def get_data_idx(npyfile, sensor=None):
        if sensor is None:
            return osp.splitext(osp.split(npyfile)[-1])[0]
        else:
            return f'{osp.splitext(osp.split(npyfile)[-1])[0]}_sensor{sensor}'

    def segment_objects(self, result):
        result['frame_gt'] = {}
        bboxes = LiDARInstance3DBoxes(result['merged']['bboxes'])
        point_cloud = result['merged']['point_cloud']
        ind = bboxes.points_in_boxes_part(
            torch.from_numpy(point_cloud)[:, :3].cuda()).cpu().numpy()
        sensors = result['merged']['sensors']
        for obj_idx in range(len(bboxes)):
            gt_points = point_cloud[ind == obj_idx].copy()
            box3d_lidar = result['merged']['bboxes'][obj_idx]
            box3d_lidar_s, gt_points_s = self.adjust_box_per_sensor(
                box3d_lidar, gt_points, sensors)
            for box3d_lidar, gt_points, s in zip(box3d_lidar_s, gt_points_s, sensors):
                gt_points[:, :3] -= box3d_lidar[None, :3]
                if len(gt_points) == 0:
                    continue
                obj_name = result['merged']['name'][obj_idx]
                db_info = {
                    'name': obj_name,
                    'path': osp.join(f'{self.save_cfg.gt_database_path}',
                                     f'{self.get_data_idx(result["merged"]["npy"], s)}_{obj_name}_{obj_idx}.bin'),
                    'image_idx': '{}',
                    'gt_idx': obj_idx,
                    'box3d_lidar': box3d_lidar,
                    'num_points_in_gt': len(gt_points),
                    'sensor': s,
                    'difficulty': 0,
                    'dataset': result['dataset']
                }
                if obj_name not in result['frame_gt']:
                    result['frame_gt'][obj_name] = []
                result['frame_gt'][obj_name].append(db_info)
                gt_points.tofile(
                    osp.join(self.save_cfg.root_path, db_info['path']))
                annos = result['frame_annos'][s]['annos']
                annos['name'].append(obj_name)
                annos['bboxes'].append(box3d_lidar)
                annos['num_points'].append(len(gt_points))
                annos['index'].append(obj_idx)
        for s in sensors:
            annos = result['frame_annos'][s]['annos']
            annos['name'] = np.array(annos['name'])
            annos['bboxes'] = np.stack(annos['bboxes'], axis=0)
            annos['num_points'] = np.array(
                annos['num_points'], dtype=np.uint32)
            annos['index'] = np.array(annos['index'], dtype=np.uint16)

    def adjust_box_per_sensor(self, bbox, points, sensors):
        ret_bboxes = []
        ret_points = []
        for s in sensors:
            points_s = points[points[..., -1] == s][..., :-1]
            bbox_s = bbox.copy()
            ret_bboxes.append(bbox_s)
            ret_points.append(points_s)
        return ret_bboxes, ret_points


if __name__ == '__main__':
    data_root = 'data/ouster'
    seed_data_mapper = seedOriginDataMapper('data/ouster/origin/output/')
    clip_mapper = originDataSeqPrevMapper('data/ouster/origin/output/')
    origin_data_mapper = originDataFullDirMapper('data/ouster/origin/output/')
    origin_index_mapper = originDataIndexMapper('data/ouster/origin/output/')
    def is_val(result): return result['dataset'] in ['ouster00016']
    save_cfg = saveConfig(
        label_map={
            '中大型车': 'big_vehicle',
            # '': 'motorcycle',
            '行人': 'pedestrian',
            '小型车': 'vehicle',
            '两轮骑行者': 'bicycle',
            '超大型车': 'huge_vehicle',
            # '': 'cone',
            '三轮骑行者': 'tricycle'},
        root_path=f'{data_root}/kitti_format',
        point_cloud_path='training/ouster',
        gt_database_path='ouster_gt_database'
    )
    label_info_mapper = labelInfoMapper(
        seed_data_mapper, clip_mapper, origin_data_mapper, origin_index_mapper, save_cfg)

    if osp.isdir(osp.join(data_root, 'kitti_format')):
        shutil.rmtree(osp.join(data_root, 'kitti_format'))
    os.makedirs(osp.join(data_root, 'kitti_format/training/ouster'),
                exist_ok=True)
    os.makedirs(osp.join(data_root, 'kitti_format/ImageSets'), exist_ok=True)
    os.makedirs(osp.join(data_root, 'kitti_format/ouster_gt_database'),
                exist_ok=True)

    trainval_annos = []
    trainval_gts = {}
    train_annos = []
    train_gts = {}
    val_annos = []
    labels = glob.glob(
        'data/ouster/origin/ousterlabel/**/*.json', recursive=True)
    for label in tqdm.tqdm(labels):
        result = label_info_mapper.load_label(label)
        if result is None:
            continue
        trainval_annos.extend([*result['frame_annos'].values()])
        if is_val(result):
            val_annos.extend([*result['frame_annos'].values()])
        else:
            train_annos.extend([*result['frame_annos'].values()])

        for k in result['frame_gt']:
            if k not in trainval_gts:
                trainval_gts[k] = []
            if k not in train_gts:
                train_gts[k] = []
            trainval_gts[k].extend(result['frame_gt'][k])
            if not is_val(result):
                train_gts[k].extend(result['frame_gt'][k])

    with open(osp.join(data_root, 'kitti_format/ouster_infos_trainval.pkl'), 'wb') as f:
        pkl.dump(trainval_annos, f)
    with open(osp.join(data_root, 'kitti_format/ouster_dbinfos_trainval.pkl'), 'wb') as f:
        pkl.dump(trainval_gts, f)

    with open(osp.join(data_root, 'kitti_format/ouster_infos_train.pkl'), 'wb') as f:
        pkl.dump(train_annos, f)
    with open(osp.join(data_root, 'kitti_format/ouster_dbinfos_train.pkl'), 'wb') as f:
        pkl.dump(train_gts, f)

    with open(osp.join(data_root, 'kitti_format/ouster_infos_val.pkl'), 'wb') as f:
        pkl.dump(val_annos, f)
