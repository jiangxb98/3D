import glob
import os.path as osp
import pickle


def get_lidar_index(p):
    set_idx = p.split('/')[-3]
    lidar_idx = osp.splitext(osp.split(p)[-1])[0]
    return f'{set_idx}/{lidar_idx}'


if __name__ == '__main__':
    data_root = 'data/semantickitti'

    trainset = {0, 1, 2, 3, 4, 5, 6, 7, 9, 10}
    valset = {8}
    testset = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21}
    pointclouds = glob.glob(osp.join(data_root, '*/velodyne/*.bin'))
    labels = glob.glob(osp.join(data_root, '*/labels/*.label'))

    pointclouds = sorted(['/'.join(p.split('/')[-3:]) for p in pointclouds])
    labels = {get_lidar_index(
        p): '/'.join(p.split('/')[-3:]) for p in labels}
    train_infos = []
    val_infos = []
    test_infos = []

    for p in pointclouds:
        set_idx = int(p.split('/')[-3])
        lidar_idx = get_lidar_index(p)
        info = dict(
            pts_path=p,
            point_cloud=dict(lidar_idx=lidar_idx))
        if lidar_idx in labels:
            info['pts_semantic_mask_path'] = labels[lidar_idx]
        if set_idx in trainset:
            assert 'pts_semantic_mask_path' in info
            train_infos.append(info)
        elif set_idx in valset:
            assert 'pts_semantic_mask_path' in info
            val_infos.append(info)
        elif set_idx in testset:
            test_infos.append(info)
        else:
            raise ValueError(
                f"Unknown data split encountered!! {set_idx}")
    with open(osp.join(data_root, 'kitti_infos_train.pkl'), 'wb') as f:
        pickle.dump(train_infos, f)
    with open(osp.join(data_root, 'kitti_infos_val.pkl'), 'wb') as f:
        pickle.dump(val_infos, f)
    with open(osp.join(data_root, 'kitti_infos_test.pkl'), 'wb') as f:
        pickle.dump(test_infos, f)
