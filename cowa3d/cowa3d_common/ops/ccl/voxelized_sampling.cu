#include "hash32.h"
#include "macros.h"
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cmath>
#include <vector>

__global__ void voxel_kernel(float *seg_points,
                             int32_t *batch_id,
                             float *voxel_config,
                             int32_t *serialized,
                             int32_t *points_to_coors,
                             int32_t *num_points_of_coors,
                             int32_t *is_coors,
                             int num_points,
                             hash::LinearHashMap<int32_t, int32_t> coor2id)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_points; i += blockDim.x * gridDim.x)
    {
        float x = seg_points[4 * i];
        float y = seg_points[4 * i + 1];
        float z = seg_points[4 * i + 2];
        int b = batch_id[i];
        float x_min = voxel_config[0];
        float y_min = voxel_config[1];
        float z_min = voxel_config[2];
        float dx = voxel_config[3];
        float dy = voxel_config[4];
        float dz = voxel_config[5];

        int coor_x = (x - x_min) / dx;
        int coor_y = (y - y_min) / dy;
        int coor_z = (z - z_min) / dz;

        // coor_x:0~2499, 2^12, coor_y:0~2499, 2^12, coor_z:0~59, 2^6, bs:2^2
        serialized[i] = b * (1 << 30) + coor_z * (1 << 24) + coor_y * (1 << 12) + coor_x;
        int hash_id = coor2id.insert_if_empty(serialized[i], i);
        assert(hash_id != coor2id.EMPTY);
        points_to_coors[i] = hash_id;
        is_coors[hash_id] = 1;
        atomicAdd(&num_points_of_coors[hash_id], 1);
    }
}

__global__ void feats_avg_kernel(float *seg_points,
                                 float *seg_logits,
                                 float *seg_vote_preds,
                                 float *seg_feats,
                                 int32_t *batch_idx,
                                 float *vote_offsets,
                                 int32_t *points_to_coors,
                                 int32_t *num_points_of_coors,
                                 int32_t *sum,
                                 int num_threads,
                                 float *seg_points_reduce,
                                 float *seg_logits_reduce,
                                 float *seg_vote_preds_reduce,
                                 float *seg_feats_reduce,
                                 int32_t *batch_idx_reduce,
                                 float *vote_offsets_reduce)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_threads; i += blockDim.x * gridDim.x)
    {
        int points_id = i / 114;
        int feats_id = i % 114;

        int hash_id = points_to_coors[points_id];
        int out_id = sum[hash_id] - 1;
        float num_points = num_points_of_coors[hash_id];

        if (feats_id < 4)
        {
            atomicAdd(&seg_points_reduce[out_id * 4 + feats_id], seg_points[points_id * 4 + feats_id] / num_points);
        }
        else if (feats_id < 10)
        {
            atomicAdd(&seg_logits_reduce[out_id * 6 + feats_id - 4], seg_logits[points_id * 6 + feats_id - 4] / num_points);
        }
        else if (feats_id < 28)
        {
            atomicAdd(&seg_vote_preds_reduce[out_id * 18 + feats_id - 10], seg_vote_preds[points_id * 18 + feats_id - 10] / num_points);
        }
        else if (feats_id < 95)
        {
            atomicAdd(&seg_feats_reduce[out_id * 67 + feats_id - 28], seg_feats[points_id * 67 + feats_id - 28] / num_points);
        }
        else if (feats_id < 113)
        {
            atomicAdd(&vote_offsets_reduce[out_id * 18 + feats_id - 95], vote_offsets[points_id * 18 + feats_id - 95] / num_points);
        }
        else
        {
            batch_idx_reduce[out_id] = batch_idx[points_id];
        }
    }
}

std::vector<torch::Tensor> voxelized_sampling(const torch::Tensor &seg_points,
                                              const torch::Tensor &seg_logits,
                                              const torch::Tensor &seg_vote_preds,
                                              const torch::Tensor &seg_feats,
                                              const torch::Tensor &batch_idx,
                                              const torch::Tensor &vote_offsets,
                                              const torch::Tensor &voxel_config)
{
    using hash = hash::LinearHashMap<int32_t, int32_t>;
    int num_points = seg_points.size(0);
    int num_hash = 2 * num_points;
    torch::Tensor serialized_coors = torch::zeros({num_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor points_to_coors = torch::empty_like(serialized_coors);

    torch::Tensor coors_hash_key = torch::full({num_hash}, hash::EMPTY, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor coors_hash_val = torch::empty_like(coors_hash_key);
    torch::Tensor num_points_of_coors = torch::zeros_like(coors_hash_key, {torch::kInt32});
    torch::Tensor is_coors = torch::zeros_like(coors_hash_key, {torch::kInt32});

    hash coors_hash(coors_hash_key.data_ptr<int32_t>(), coors_hash_val.data_ptr<int32_t>(), num_hash);

    voxel_kernel<<<(num_points + 255) / 256, 256>>>(seg_points.data_ptr<float>(),
                                                    batch_idx.data_ptr<int32_t>(),
                                                    voxel_config.data_ptr<float>(),
                                                    serialized_coors.data_ptr<int32_t>(),
                                                    points_to_coors.data_ptr<int32_t>(),
                                                    num_points_of_coors.data_ptr<int32_t>(),
                                                    is_coors.data_ptr<int32_t>(),
                                                    num_points, coors_hash);

    torch::Tensor sum = is_coors.cumsum(0).to(torch::kInt32);
    
    int num_coors = sum[num_hash - 1].item<int>();
    torch::Tensor seg_points_reduce = torch::zeros({num_coors, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_logits_reduce = torch::zeros({num_coors, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_vote_preds_reduce = torch::zeros({num_coors, 18}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_feats_reduce = torch::zeros({num_coors, 67}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor batch_idx_reduce = torch::zeros({num_coors}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor vote_offsets_reduce = torch::zeros({num_coors, 18}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    int num_threads = num_points * 114; // 4+6+18+67+18+1
    feats_avg_kernel<<<(num_threads + 255) / 256, 256>>>(seg_points.data_ptr<float>(),
                                                         seg_logits.data_ptr<float>(),
                                                         seg_vote_preds.data_ptr<float>(),
                                                         seg_feats.data_ptr<float>(),
                                                         batch_idx.data_ptr<int32_t>(),
                                                         vote_offsets.data_ptr<float>(),
                                                         points_to_coors.data_ptr<int32_t>(),
                                                         num_points_of_coors.data_ptr<int32_t>(),
                                                         sum.data_ptr<int32_t>(),
                                                         num_threads,
                                                         seg_points_reduce.data_ptr<float>(),
                                                         seg_logits_reduce.data_ptr<float>(),
                                                         seg_vote_preds_reduce.data_ptr<float>(),
                                                         seg_feats_reduce.data_ptr<float>(),
                                                         batch_idx_reduce.data_ptr<int32_t>(),
                                                         vote_offsets_reduce.data_ptr<float>());

    std::vector<torch::Tensor> out;
    out.push_back(seg_points_reduce);
    out.push_back(seg_logits_reduce);
    out.push_back(seg_vote_preds_reduce);
    out.push_back(seg_feats_reduce);
    out.push_back(batch_idx_reduce);
    out.push_back(vote_offsets_reduce);

    return out;
}
