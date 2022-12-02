_base_ = ['./cylinder3d.py', './semantic_kitti.py']

lr = 0.004
# The optimizer follows the setting in SECOND.Pytorch, but here we use
# the offcial AdamW optimizer implemented by PyTorch.
optimizer = dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
optimizer_config = dict(grad_clip=None)
# We use cyclic learning rate and momentum schedule following SECOND.Pytorch
# https://github.com/traveller59/second.pytorch/blob/3aba19c9688274f75ebb5e576f65cfe54773c021/torchplus/train/learning_schedules_fastai.py#L69  # noqa
# We implement them in mmcv, for more details, please refer to
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/lr_updater.py#L327  # noqa
# https://github.com/open-mmlab/mmcv/blob/f48241a65aebfe07db122e9db320c31b685dc674/mmcv/runner/hooks/momentum_updater.py#L130  # noqa
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    warmup_by_epoch=False)
# lr_config = dict(
#     policy='cyclic',
#     target_ratio=(10, 1e-4),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
# momentum_config = dict(
#     policy='cyclic',
#     target_ratio=(0.85 / 0.95, 1),
#     cyclic_times=1,
#     step_ratio_up=0.4,
# )
# Although the max_epochs is 40, this schedule is usually used we
# RepeatDataset with repeat ratio N, thus the actual max epoch
# number could be Nx40
runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=1)
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
