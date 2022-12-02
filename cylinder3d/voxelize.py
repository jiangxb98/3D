# -*- coding:utf-8 -*-
import torch
from mmcv.ops import Voxelization


class Cylinderization(Voxelization):
    EPS = 1e-3

    def __init__(self,
                 cylinder_partition,
                 cylinder_range,
                 max_num_points,
                 max_voxels=20000,
                 clamp_input=True,
                 deterministic=True):

        cylinder_range = torch.tensor(
            cylinder_range, dtype=torch.float32)
        cylinder_partition = torch.tensor(
            cylinder_partition, dtype=torch.float32)
        voxel_size = (cylinder_range[3:] -
                      cylinder_range[:3]) / cylinder_partition

        super(Cylinderization, self).__init__(
            voxel_size=voxel_size.tolist(),
            point_cloud_range=cylinder_range.tolist(),
            max_num_points=max_num_points,
            max_voxels=max_voxels,
            deterministic=deterministic)

        self.grid_size = cylinder_partition
        self.clamp_input = clamp_input

    def forward(self, input):
        cart, other = input[:, :3], input[:, 3:]
        polar = self.cart2polar(cart)
        if self.clamp_input:
            polar_clamp = []
            for i in range(3):
                polar_clamp.append(polar[:, i].clamp(
                    self.point_cloud_range[i], self.point_cloud_range[i+3]-self.EPS))
            polar_clamp = torch.stack(polar_clamp, dim=-1)
            input = torch.cat([polar_clamp, polar, cart[:, :2], other], dim=-1)
            output = super(Cylinderization, self).forward(input)
            if isinstance(output, tuple):  # hard voxelization
                voxels_out, coors_out, num_points_per_voxel_out = output
                return voxels_out[..., 3:], coors_out, num_points_per_voxel_out
            else:
                return input[:, 3:], output
        else:
            input = torch.cat([polar, cart[:, :2], other], dim=-1)
            output = super(Cylinderization, self).forward(input)
            if isinstance(output, tuple):  # hard voxelization
                return output
            else:
                return input, output

    @classmethod
    def cart2polar(cls, input):
        rho = input[:, :2].norm(dim=-1, p=2)
        phi = torch.atan2(input[:, 1], input[:, 0])
        return torch.stack((input[:, 2], phi, rho), dim=1)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += 'cylinder_partition=' + str(self.grid_size)
        s += ', cylinder_range=' + str(self.point_cloud_range)
        s += ', max_num_points=' + str(self.max_num_points)
        s += ', max_voxels=' + str(self.max_voxels)
        s += ', clamp_input=' + str(self.clamp_input)
        s += ', deterministic=' + str(self.deterministic)
        s += ')'
        return s
