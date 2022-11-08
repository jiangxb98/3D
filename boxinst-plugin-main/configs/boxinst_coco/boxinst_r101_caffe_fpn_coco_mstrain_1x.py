_base_ = './boxinst_r50_caffe_fpn_coco_mstrain_1x.py'
# model settings
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(
            checkpoint='open-mmlab://detectron2/resnet101_caffe')))
