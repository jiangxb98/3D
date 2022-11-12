from glob import glob
import os.path as osp
load_dir = '/disk/deepdata/dataset/waymo_v1.4/training'
ls = glob(osp.join(load_dir,'*.tfrecord'))
print(ls)