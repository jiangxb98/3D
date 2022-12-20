import torch
from cylinder3d.middle_encoder import AsymSparseUnet


def load_checkpoint(model, cylinder_model_path):
    trans_dict = {'dr': {'conv1': 'conv1.0', 'bn0': 'conv1.2', 'conv1_2': 'conv1_2.0', 'bn0_2': 'conv1_2.2',
                         'conv2': 'conv2.0', 'bn1': 'conv2.2', 'conv3': 'conv3.0', 'bn2': 'conv3.2', 'pool': 'pool'},
                  'up': {'trans_dilao': 'trans_dilao.0', 'trans_bn': 'trans_dilao.2', 'conv1': 'conv1.0', 'bn1': 'conv1.2',
                         'conv2': 'conv2.0', 'bn2': 'conv2.2', 'conv3': 'conv3.0', 'bn3': 'conv3.2', 'up_subm': 'up_subm'},
                  'ReconNet': {'conv1': 'conv1.0', 'bn0': 'conv1.1', 'conv1_2': 'conv1_2.0', 'bn0_2': 'conv1_2.1',
                               'conv1_3': 'conv1_3.0', 'bn0_3': 'conv1_3.1'},
                  'logits': {'logits': 'logits'}}
    model_dict = model.state_dict()
    cylinder_dict = torch.load(cylinder_model_path)

    for key, value in cylinder_dict.items():  # e.g, key = 'downCntx.conv1.weight'
        if ('downCntx' in key) or ('resBlock' in key):
            t_dict = trans_dict['dr']
        elif ('upBlock' in key):
            t_dict = trans_dict['up']
        elif ('ReconNet' in key):
            t_dict = trans_dict['ReconNet']
        elif ('logits' in key):
            t_dict = trans_dict['logits']

        if (not ('logits' in key)):
            old_str = key.split('.')[1]  # e.g, 'conv1'
            key = key.replace(old_str, t_dict[old_str])

        assert (key in model_dict.keys())
        model_dict[key] = value

    model.load_state_dict(model_dict)
    return model


def test_middle_encoder():
    mid_enc = AsymSparseUnet(sparse_shape=[480, 360, 32],
                             in_channels=16,
                             feat_channels=32,
                             num_classes=20)
    mid_enc = load_checkpoint(
        mid_enc, 'cylinder3d/tests/data/midenc_params.pth')

    mid_enc.cuda()
    mid_enc.eval()

    tgt_outp = torch.load('cylinder3d/tests/data/midenc_output.pth')
    voxel_features, coors = torch.load(
        'cylinder3d/tests/data/midenc_input.pth')
    batch_size = 1
    outp = mid_enc(voxel_features, coors, batch_size)
    assert torch.allclose(outp, tgt_outp, rtol=1e-4, atol=1e-4)
