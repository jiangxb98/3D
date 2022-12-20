import torch
import numpy as np
import os.path as osp
from mmcv import Config
from mmcv.cnn.utils import revert_sync_batchnorm
from mmdet3d.apis import init_model
from deploy3d.symfun.ops.voxel_encoder import cylinder_encoder
from deploy3d.symfun.ops.spconv import spconv_mm, spconv_index
from deploy3d.symfun.ops.point_voxel_mapping import scatter_to, gather_back


class InferSpConvModule(torch.nn.Module):
    def __init__(self, conv, bn, act, index_name, max_num_act_out):
        super(InferSpConvModule, self).__init__()
        if bn:
            self.weight, self.bias = self.fused_weight_bias(conv, bn)
        else:
            self.weight, self.bias = self.only_weight(conv)
        self.conv = conv
        self.bn = bn
        self.act = act
        self.index_name = index_name
        self.max_num_act_out = max_num_act_out

    @torch.no_grad()
    def fused_weight_bias(self, conv, bn):
        w = conv.weight
        std = bn.running_var.sqrt()
        m = bn.running_mean
        nw = bn.weight
        w = w / std[:, None, None, None, None] * nw[:, None, None, None, None]
        b = bn.bias - m / std * nw
        o, i = w.shape[0], w.shape[4]
        kvol = w.shape[1] * w.shape[2] * w.shape[3]
        w = w.reshape(o, kvol, i)
        w = w.permute(1, 2, 0)
        w = w.reshape(-1).contiguous()
        return w.float().cpu().numpy(), b.float().cpu().numpy()

    @torch.no_grad()
    def only_weight(self, conv):
        b = torch.zeros((conv.weight.shape[0]))

        w = conv.weight
        o, i = w.shape[0], w.shape[4]
        kvol = w.shape[1] * w.shape[2] * w.shape[3]
        w = w.reshape(o, kvol, i)
        w = w.permute(1, 2, 0)
        w = w.reshape(-1).contiguous()

        return w.float().cpu().numpy(), b.float().cpu().numpy()

    def forward(self, inp):
        in_feats, in_coors, num_act_in, in_spatial_shape, index_dict = inp
        if self.index_name in index_dict:
            indices = index_dict[self.index_name]
        else:
            assert not self.conv.inverse
            indices = spconv_index(
                in_coors,
                num_act_in,
                in_spatial_shape,
                self.conv.kernel_size,
                self.conv.stride,
                self.conv.padding,
                self.conv.dilation,
                self.conv.output_padding,
                self.max_num_act_out,
                self.conv.subm,
                self.conv.transposed)
            if self.conv.subm:
                indices = indices + (in_coors, num_act_in, in_spatial_shape)
            indices = indices + (in_coors, num_act_in, in_spatial_shape)
            index_dict[self.index_name] = indices
        
        index, index_buf_len, out_coors, num_act_out, out_spatial_shape, in_coors, num_act_in, in_spatial_shape = indices
        
        out_feats = spconv_mm(
            in_feats,
            num_act_in,
            num_act_out,
            index,
            index_buf_len,
            self.conv.kernel_size,
            self.conv.in_channels,
            self.conv.out_channels,
            self.max_num_act_out,
            self.conv.subm,
            self.conv.inverse,
            self.weight,
            self.bias)
        if self.act:
            out_feats = self.act(out_feats)
        
        if self.conv.inverse:
            num_act_in, num_act_out = num_act_out, num_act_in
            out_coors, in_coors = in_coors, out_coors
            out_spatial_shape, in_spatial_shape = in_spatial_shape, out_spatial_shape
        
        return out_feats, out_coors, num_act_out, out_spatial_shape, index_dict


class ResContextBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out):
        super(ResContextBlock, self).__init__()
        self.conv1 = InferSpConvModule(*model.conv1, 'res2_1', max_num_act_out)
        self.conv1_2 = InferSpConvModule(
            *model.conv1_2, 'res1_1', max_num_act_out)
        self.conv2 = InferSpConvModule(*model.conv2, 'res1_1', max_num_act_out)
        self.conv3 = InferSpConvModule(*model.conv3, 'res2_1', max_num_act_out)

    def forward(self, x):
        shortcut = self.conv1(x)
        
        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        tmp = resA[0] + shortcut[0]

        return (tmp, resA[1], resA[2], resA[3], resA[4])
        

class ResBlock(torch.nn.Module):
    def __init__(self, model, res_act_out, pool_act_out, stride, indice_key):
        super(ResBlock, self).__init__()

        self.conv1 = InferSpConvModule(
            *model.conv1, 'res1_' + str(stride), res_act_out)
        self.conv1_2 = InferSpConvModule(
            *model.conv1_2, 'res2_' + str(stride), res_act_out)
        self.conv2 = InferSpConvModule(
            *model.conv2, 'res2_' + str(stride), res_act_out)
        self.conv3 = InferSpConvModule(
            *model.conv3, 'res1_' + str(stride), res_act_out)
        self.pool = InferSpConvModule(
            model.pool, None, None, indice_key, pool_act_out)

    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut = self.conv1_2(shortcut)

        resA = self.conv2(x)

        resA = self.conv3(resA)

        tmp = resA[0] + shortcut[0]
        resA = (tmp, resA[1], resA[2], resA[3], resA[4])

        resB = self.pool(resA)

        return resB, resA


class UpBlock(torch.nn.Module):
    def __init__(self, model, res_act_out, up_act_out, stride, indice_key):
        super(UpBlock, self).__init__()

        self.trans_dilao = InferSpConvModule(
            *model.trans_dilao, 'trans_dilao_' + str(stride), res_act_out)
        self.conv1 = InferSpConvModule(
            *model.conv1, 'res2_' + str(stride), up_act_out)
        self.conv2 = InferSpConvModule(
            *model.conv2, 'res1_' + str(stride), up_act_out)
        self.conv3 = InferSpConvModule(
            *model.conv3, 'up3_' + str(stride), up_act_out)
        self.up_subm = InferSpConvModule(
            model.up_subm, None, None, indice_key, up_act_out)

    def forward(self, x, skip):
        upA = self.trans_dilao(x)

        upA = self.up_subm(upA)
        
        tmp = upA[0] + skip[0]
        upA = (tmp, skip[1], skip[2], skip[3], upA[4])

        upE = self.conv1(upA)

        upE = self.conv2(upE)

        upE = self.conv3(upE)

        return upE        


class ReconBlock(torch.nn.Module):
    def __init__(self, model, max_num_act_out):
        super(ReconBlock, self).__init__()

        self.conv1 = InferSpConvModule(
            model.conv1[0][0], model.conv1[0][1], model.conv1[1], 'rec1', max_num_act_out)
        self.conv1_2 = InferSpConvModule(
            model.conv1_2[0][0], model.conv1_2[0][1], model.conv1_2[1], 'rec1_2', max_num_act_out)
        self.conv1_3 = InferSpConvModule(
            model.conv1_3[0][0], model.conv1_3[0][1], model.conv1_3[1], 'rec1_3', max_num_act_out)

    def forward(self, x):
        shortcut = self.conv1(x)

        shortcut2 = self.conv1_2(x)

        shortcut3 = self.conv1_3(x)

        tmp = shortcut[0] + shortcut2[0] + shortcut3[0]
        shortcut = (tmp, shortcut[1], shortcut[2], shortcut[3], shortcut3[4])
        
        tmp = shortcut[0] * x[0]

        return (tmp, shortcut[1], shortcut[2], shortcut[3], shortcut[4])


class MiddleEncoder(torch.nn.Module):
    def __init__(self, middle_encoder, max_num_act_out, num_layers): # max_num_act_out = [131072, 65536, 32768, 16384]
        super(MiddleEncoder, self).__init__()

        self.downCntx = ResContextBlock(
            middle_encoder.downCntx, max_num_act_out[0])

        indice_keys = []
        resBlocks = torch.nn.ModuleList()
        for i in range(num_layers):  # i = 0, 1, 2; stride = 1, 2, 4
            stride = pow(2, i)
            indice_key = 'res_pool_' + str(stride)
            indice_keys.append(indice_key)
            resBlocks.append(
                ResBlock(middle_encoder.resBlocks[i], max_num_act_out[i], max_num_act_out[i+1], stride, indice_key))
        self.resBlocks = resBlocks

        upBlocks = torch.nn.ModuleList()
        for i in range(num_layers):  # i = 0, 1, 2; stride = 4, 2, 1
            indice_key = indice_keys.pop()
            stride = pow(2, num_layers - i - 1)
            upBlocks.append(
                UpBlock(middle_encoder.upBlocks[i], max_num_act_out[num_layers - i], max_num_act_out[num_layers - i - 1], stride, indice_key))
        self.upBlocks = upBlocks

        self.ReconNet = ReconBlock(middle_encoder.ReconNet, max_num_act_out[0])

        self.logits = InferSpConvModule(
            middle_encoder.logits, None, None, 'logits', max_num_act_out[0])

    def forward(self, in_feats, in_coors, num_act_in, in_spatial_shape):
        index_dict = dict()

        # downcntx
        downc = self.downCntx(
            (in_feats, in_coors, num_act_in, in_spatial_shape, index_dict))
        
        # resblocks
        res = []
        for resBlock in self.resBlocks:
            downc, downb = resBlock(downc)
            res.append((downc, downb))
        
        # upblocks
        upe = res[-1][0]
        for upBlock in self.upBlocks:
            _, downb = res.pop()
            upe = upBlock(upe, downb)
        
        # reconnet
        up0e = self.ReconNet(upe)
        
        tmp = torch.cat((up0e[0], upe[0]), 1)
        up0e = (tmp, up0e[1], up0e[2], up0e[3], up0e[4])
        
        # logits
        logits = self.logits(up0e)
        return logits[0]


class PointFeatureNet(torch.nn.Module):
    def __init__(self, model, max_num_act_out, reduce_type=0):
        super(PointFeatureNet, self).__init__()
        self.reduce_type = reduce_type
        self.max_num_act_out = max_num_act_out
        self.pfn_layers = model.pfn_layers
        self.post_reduce_layers = model.post_reduce_layers
    
    def forward(self, batch_point_feats, batch_indices, cylinder_config, in_spatial_shape):
        pts_feats, scatter_index, scatter_count, out_coors, num_act_out = cylinder_encoder(batch_point_feats,
                                                                                           batch_indices,
                                                                                           cylinder_config,
                                                                                           in_spatial_shape,
                                                                                           self.max_num_act_out)
        for i, pfn in enumerate(self.pfn_layers):
            point_feats = pfn(pts_feats)
            voxel_feats = scatter_to(
                point_feats, scatter_index, scatter_count, self.reduce_type)
            if i != len(self.pfn_layers) - 1:
                feat_per_point = gather_back(voxel_feats, scatter_index, 0.0)
                pts_feats = torch.cat([point_feats, feat_per_point], dim=1)

        voxel_feats = self.post_reduce_layers(voxel_feats)
        return voxel_feats, out_coors, num_act_out, scatter_index


class InferModel(torch.nn.Module):
    def __init__(self, model, num_layers, max_num_act_out, reduce_type):
        super(InferModel, self).__init__() 
        self.reduce_type = reduce_type
        self.pfn = PointFeatureNet(model.pts_voxel_encoder, max_num_act_out[0], reduce_type)
        self.pts_middle_encoder = MiddleEncoder(model.pts_middle_encoder, max_num_act_out, num_layers)
    
    def forward(self, batch_point_feats, batch_indices, cylinder_config, in_spatial_shape):       
        voxel_feats, out_coors, num_act_out, scatter_index = self.pfn(batch_point_feats,
                                                                      batch_indices,
                                                                      cylinder_config,
                                                                      in_spatial_shape)     
        
        
        logits = self.pts_middle_encoder(voxel_feats,
                                         out_coors,
                                         num_act_out,
                                         in_spatial_shape)      
        
        
        logits = gather_back(logits, scatter_index, 0.0)
        logits = torch.argmax(logits, dim=1).int()
        
        return logits # (N,)


def export_onnx(config_file, model_file, onnx_file='cylinder3d.onnx'):
    cfg = Config.fromfile(config_file)
    model = init_model(cfg, model_file)
    model = revert_sync_batchnorm(model)
    
    num_layers = 3
    reduce_type = 0 # reduce_max
    max_num_act_out = [131072, 65536, 32768, 16384] # [102124, 64857, 21810, 11491]
    cylinder3d = InferModel(model, num_layers, max_num_act_out, reduce_type).cuda()
    cylinder3d.eval() # model.eval()中已经set bn.training=True, 此处可不加.eval(), 但为了规范, 还是加上了

    batch_point_feats = torch.zeros((480000, 4), dtype=torch.float32).cuda()
    batch_indices = torch.zeros((480000,), dtype=torch.int32).cuda()
    in_spatial_shape = torch.empty((1, 0, 480, 360, 32), dtype=torch.int32).cuda()
    cylinder_config = torch.tensor([-2, -np.pi, 0, 4, np.pi, 50]).cuda()
    
    torch.onnx.export(
            cylinder3d,
            (batch_point_feats, 
             batch_indices,
             cylinder_config,
             in_spatial_shape),
            onnx_file,
            verbose=False,
            opset_version=9,
            enable_onnx_checker=False,
            keep_initializers_as_inputs=True,
            input_names=['batch_point_feats', 'batch_indices', 'cylinder_config', 'in_spatial_shape'],
            output_names=['batch_point_labels'],
            dynamic_axes={
            'batch_point_feats': {0: 'n'},
            'batch_indices': {0: 'n'},
            'in_spatial_shape': {0: 'b'},
            'batch_point_labels': {0:'n'},
            }
        )


if __name__ == '__main__':
    model_path = 'work_dirs/cylinder3d_semantic_cowa_debug/0926_corolla_and_x3_voted_box3d_concatedLabels_p3_234'
    config_file = osp.join(model_path, 'cylinder3d_semantic_cowa.py')
    model_file = osp.join(model_path, 'epoch_35.pth')
    export_onnx(config_file, model_file)