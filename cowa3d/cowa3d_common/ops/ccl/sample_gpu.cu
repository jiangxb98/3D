#include "hash32.h"
#include "macros.h"
#include <torch/extension.h>
#include <cub/cub.cuh>
#include <cmath>
#include <vector>

__global__ void filter_mask_kernel(float *seg_logits,
                                   float *score_thresh,
                                   int8_t *fg_mask,
                                   int num_mask)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_mask; i += blockDim.x * gridDim.x)
    {
        int class_id = i % 6;
        float seg_scores = (1 / (1 + exp(-seg_logits[i])));
        float s = score_thresh[class_id];
        if (seg_scores > s)
        {
            fg_mask[i] = 1;
        }
    }
}

__global__ void filter_feats_kernel(float *seg_points,
                                    float *seg_logits,
                                    float *seg_vote_preds,
                                    float *seg_feats,
                                    int32_t *batch_idx,
                                    float *vote_offsets,
                                    int8_t *fg_mask,
                                    int32_t *mask,
                                    int num_threads,
                                    float *seg_points_reduce,
                                    float *seg_logits_reduce,
                                    float *seg_vote_preds_reduce,
                                    float *seg_feats_reduce,
                                    float *vote_offsets_reduce,
                                    float *center_preds_reduce,
                                    int32_t *batch_idx_reduce,
                                    int32_t *class_idx_reduce)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_threads; i += blockDim.x * gridDim.x)
    {
        int points_id = i / 118;
        int feats_id = i % 118;

        for (int class_id = 0; class_id < 6; class_id++)
            if (fg_mask[points_id * 6 + class_id])
            {
                int out_id = mask[points_id * 6 + class_id] - 1;
                if (feats_id < 4)
                {
                    seg_points_reduce[out_id * 4 + feats_id] = seg_points[points_id * 4 + feats_id];
                }
                else if (feats_id < 10)
                {
                    seg_logits_reduce[out_id * 6 + feats_id - 4] = seg_logits[points_id * 6 + feats_id - 4];
                }
                else if (feats_id < 28)
                {
                    seg_vote_preds_reduce[out_id * 18 + feats_id - 10] = seg_vote_preds[points_id * 18 + feats_id - 10];
                }
                else if (feats_id < 95)
                {
                    seg_feats_reduce[out_id * 67 + feats_id - 28] = seg_feats[points_id * 67 + feats_id - 28];
                }
                else if (feats_id < 113)
                {
                    vote_offsets_reduce[out_id * 18 + feats_id - 95] = vote_offsets[points_id * 18 + feats_id - 95];
                }
                else if (feats_id < 116)
                {
                    center_preds_reduce[out_id * 3 + feats_id - 113] = seg_points[points_id * 4 + feats_id - 113] + vote_offsets[points_id * 18 + class_id * 3 + feats_id - 113];
                }
                else if (feats_id < 117)
                {
                    batch_idx_reduce[out_id] = batch_idx[points_id];
                }
                else
                {
                    class_idx_reduce[out_id] = class_id;
                }
            }
    }
}

std::vector<torch::Tensor> sample(const torch::Tensor &seg_points,
                                  const torch::Tensor &seg_logits,
                                  const torch::Tensor &seg_vote_preds,
                                  const torch::Tensor &seg_feats,
                                  const torch::Tensor &batch_idx,
                                  const torch::Tensor &vote_offsets,
                                  const torch::Tensor &score_thresh)
{
    using hash = hash::LinearHashMap<int32_t, int32_t>;
    int num_points = seg_points.size(0);
    int num_mask = num_points * 6;
    torch::Tensor fg_mask = torch::zeros({num_mask}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));

    filter_mask_kernel<<<(num_mask + 255) / 256, 256>>>(seg_logits.data_ptr<float>(),
                                                        score_thresh.data_ptr<float>(),
                                                        fg_mask.data_ptr<int8_t>(),
                                                        num_mask);

    torch::Tensor mask = fg_mask.cumsum(0).to(torch::kInt32);
    // torch::Tensor mask = torch::zeros({num_mask}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    // size_t temp_storage_bytes = size_t(4 * num_mask);
    // torch::Tensor temp = torch::empty({4 * num_mask}, torch::TensorOptions().dtype(torch::kInt8).device(torch::kCUDA));
    // cub::DeviceScan::InclusiveSum((void*)temp.data_ptr<int8_t>(), temp_storage_bytes, fg_mask.data_ptr<int8_t>(), mask.data_ptr<int32_t>(), num_mask);
    int num_out_points;
    cudaMemcpy(&num_out_points, mask.data_ptr<int32_t>() + num_mask - 1, 4, cudaMemcpyDeviceToHost);
    torch::Tensor seg_points_reduce = torch::zeros({num_out_points, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_logits_reduce = torch::zeros({num_out_points, 6}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_vote_preds_reduce = torch::zeros({num_out_points, 18}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor seg_feats_reduce = torch::zeros({num_out_points, 67}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor vote_offsets_reduce = torch::zeros({num_out_points, 18}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor center_preds_reduce = torch::zeros({num_out_points, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor batch_idx_reduce = torch::zeros({num_out_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor class_idx_reduce = torch::zeros({num_out_points}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    int num_threads = num_points * 118; // 4+6+18+67+18+3+1+1
    filter_feats_kernel<<<(num_threads + 255) / 256, 256>>>(seg_points.data_ptr<float>(),
                                                            seg_logits.data_ptr<float>(),
                                                            seg_vote_preds.data_ptr<float>(),
                                                            seg_feats.data_ptr<float>(),
                                                            batch_idx.data_ptr<int32_t>(),
                                                            vote_offsets.data_ptr<float>(),
                                                            fg_mask.data_ptr<int8_t>(),
                                                            mask.data_ptr<int32_t>(),
                                                            num_threads,
                                                            seg_points_reduce.data_ptr<float>(),
                                                            seg_logits_reduce.data_ptr<float>(),
                                                            seg_vote_preds_reduce.data_ptr<float>(),
                                                            seg_feats_reduce.data_ptr<float>(),
                                                            vote_offsets_reduce.data_ptr<float>(),
                                                            center_preds_reduce.data_ptr<float>(),
                                                            batch_idx_reduce.data_ptr<int32_t>(),
                                                            class_idx_reduce.data_ptr<int32_t>());

    std::vector<torch::Tensor> out;
    out.push_back(seg_points_reduce);
    out.push_back(seg_logits_reduce);
    out.push_back(seg_vote_preds_reduce);
    out.push_back(seg_feats_reduce);
    out.push_back(batch_idx_reduce);
    out.push_back(vote_offsets_reduce);
    out.push_back(center_preds_reduce);
    out.push_back(fg_mask);
    out.push_back(class_idx_reduce);

    return out;
}
