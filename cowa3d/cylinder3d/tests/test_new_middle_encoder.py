import torch
from cylinder3d import AsymSparseUnet
from mmcv.cnn.utils import revert_sync_batchnorm

def test_middle():
    model = AsymSparseUnet(sparse_shape=[480, 360, 32],
                           in_channels=16,
                           feat_channels=32,
                           num_classes=17,
                           strides=(64, 64, 128),
                           height_poolings=[True, True, False],
                           conv_type='SubMConv3d',
                           norm_cfg=dict(type='SyncBN', eps=1e-05, momentum=0.1),
                           order=('conv', 'norm', 'act'))
    
    existed_state_dict = torch.load('./work_dirs/cylinder3d_semantic_cowa_debug/ddd/model.pth')
    
    state_dict = model.state_dict()
    for key in state_dict.keys():
        old_key = key
        if 'ReconNet' in key:
            if key[14] == '.' :
                old_key = key[:14] + key[16:]
            elif key[16] == '.' :
                old_key = key[:16] + key[18:]
        
        state_dict[key] = existed_state_dict[old_key]
       
    model.load_state_dict(state_dict)
    
    model.cuda()
    model.eval()
    
    
    bs = 1
    voxel_features, coors = torch.load('./work_dirs/cylinder3d_semantic_cowa_debug/ddd/in.pth')
    targets = torch.load('./work_dirs/cylinder3d_semantic_cowa_debug/ddd/out.pth')
    
    y = model(voxel_features, coors, bs)
    assert torch.allclose(y.float(), targets.float(), rtol=1e-1, atol=1e-1)
    