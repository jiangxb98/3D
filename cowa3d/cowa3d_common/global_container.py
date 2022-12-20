# coding: utf-8

from mmcv.runner.dist_utils import master_only


def _init():
    global _global_dict
    _global_dict = {}

def get_keys():
    return _global_dict.keys()
 
def set_value(key, value):
    _global_dict[key] = value
 
 
def get_value(key, def_value=None):
    try:
        return _global_dict[key]
    except KeyError:
        return def_value


@master_only
def set_tensorboard_value(key, value):
    _global_dict[key] = value


def get_tensorboard_value(key):
    return get_value(key)