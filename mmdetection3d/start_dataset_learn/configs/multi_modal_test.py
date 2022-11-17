_base_ = [
    './_base_/models/hv_pointpillars_secfpn_waymo.py',
    './_base_/datasets/ai_waymo-3d-3class-1f.py',
    './_base_/schedules/schedule_2x.py',
    './_base_/default_runtime.py',
]

# data settings
# evaluation = dict(interval=24)
fp16 = dict(loss_scale='dynamic')
