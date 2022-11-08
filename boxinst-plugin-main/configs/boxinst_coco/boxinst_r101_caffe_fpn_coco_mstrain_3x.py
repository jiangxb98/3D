_base_ = './boxinst_r101_caffe_fpn_coco_mstrain_1x.py'

lr_config = dict(step=[28, 34])

runner = dict(type='EpochBasedRunner', max_epochs=36)
