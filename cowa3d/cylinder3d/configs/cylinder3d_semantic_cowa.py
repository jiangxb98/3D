_base_ = ['./cylinder3d.py', './semantic_cowa.py']

cylinder_range=[-2, -50, -50, 4, 50, 50] # zyx's order
cylinder_partition=[30, 400, 400] # zyx's order
sparse_shape=[400, 400, 30] # xyz's order

# cylinder_range=[-2, -3.141592653589793, 0, 4, 3.141592653589793, 50]
# cylinder_partition=[32, 360, 480]
# sparse_shape=[480, 360, 32]

model = dict(
    pts_voxel_layer=dict(
        type='Cartesian', # ['Cylinder', 'Cartesian']
        cylinder_range=cylinder_range, # change pipline and in_channels simultaneously if type changed
        cylinder_partition=cylinder_partition),
    pts_voxel_encoder=dict(
        in_channels=5, # 4 if Cartesian, else 6, for sweep, both plus 1
        cylinder_range=cylinder_range,
        cylinder_partition=cylinder_partition),
    pts_middle_encoder=dict(num_classes=15,
                            sparse_shape=sparse_shape),
    pts_seg_head=dict(num_classes=15,
                      grid_size=sparse_shape))

lr = 0.0001  # 0.004
optimizer = dict(type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.1,
    warmup_by_epoch=False)

runner = dict(type='EpochBasedRunner', max_epochs=40)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

opencv_num_threads = 0
mp_start_method = 'fork'
fp16 = dict(loss_scale='dynamic')
