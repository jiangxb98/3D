import torch
import torch.nn.functional as F
from cylinder3d.head import Cylinder3dHead

def test_head():
    head = Cylinder3dHead(grid_size=[480, 360, 32],
                          ignore=0,
                          num_classes=20,
                          loss_func=dict(
                            type='CrossEntropyLoss',
                            reduction='none', # 'mean'
                            avg_non_ignore=False, # reduction & ignore, must be True
                            loss_weight=1.0),
                          loss_lova=dict(
                            type='LovaszLoss',
                            per_image=False,
                            reduction='none',
                            loss_weight=1.0))
    '''
    coors = torch.load('cylinder3d/tests/data/head_coors.pth')
    coors = torch.tensor(coors).cuda()
    coors = F.pad(coors, (1, 0), 'constant', value=0)
    assert coors.shape[1] == 4
    
    labels = torch.load('cylinder3d/tests/data/head_points_labels.pth')
    labels = [torch.from_numpy(labels).cuda().squeeze()]

    voxel_label_targets = torch.load('cylinder3d/tests/data/head_voxel_labels_target.pth')
    voxel_label_targets = torch.tensor(voxel_label_targets)
    voxel_labels = head.get_targets(coors, labels)
    assert (voxel_labels == voxel_label_targets.to(voxel_labels)).all()
    # assert torch.allclose(voxel_labels.to(torch.uint8).cpu(), voxel_label_targets.to(torch.uint8), rtol=1, atol=1)
    
    head_outputs = torch.load('cylinder3d/tests/data/head_outputs.pth').cuda()
    voxel_labels = torch.load('cylinder3d/tests/data/head_voxel_labels_inputforloss.pth').cuda()
    
    lossLova = head.loss_lova(torch.nn.functional.softmax(head_outputs), voxel_labels)
    lossCE = head.loss_func(head_outputs, voxel_labels)
    # assert (lossLova - torch.tensor(0.9417) < torch.tensor(0.1))
    # assert (lossCE - torch.tensor(2.8167) < torch.tensor(0.1))
    assert lossLova == 0.9417
    assert lossCE == 2.8167
    '''
    head_outputs = torch.load('cylinder3d/tests/data/head_outputs_none.pth').cuda()
    voxel_labels = torch.load('cylinder3d/tests/data/head_voxel_labels_inputforloss_none.pth').cuda()
    lossLova_none = torch.load('cylinder3d/tests/data/lossLova_none.pth')
    lossLova_none = torch.tensor(lossLova_none)
    lossCE_none = torch.load('cylinder3d/tests/data/lossCE_none.pth')

    lossLova = head.loss_lova(torch.nn.functional.softmax(head_outputs), voxel_labels)
    lossCE = head.loss_func(head_outputs, voxel_labels)
    # assert (lossLova - torch.tensor(0.9417) < torch.tensor(0.1))
    # assert (lossCE - torch.tensor(2.8167) < torch.tensor(0.1))
    assert (lossLova == lossLova_none).all()
    assert (lossCE == lossCE_none).all()

