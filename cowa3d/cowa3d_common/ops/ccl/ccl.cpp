#include <torch/extension.h>

torch::Tensor spccl(const torch::Tensor &coors);
torch::Tensor voxel_spccl(const torch::Tensor &points,
                          const torch::Tensor &batch_id,
                          const torch::Tensor &class_id,
                          const torch::Tensor &voxel_config,
                          const int min_points,
                          const int num_batch_size);
std::vector<torch::Tensor> voxelized_sampling(const torch::Tensor &seg_points,
                                              const torch::Tensor &seg_logits,
                                              const torch::Tensor &seg_vote_preds,
                                              const torch::Tensor &seg_feats,
                                              const torch::Tensor &batch_idx,
                                              const torch::Tensor &vote_offsets,
                                              const torch::Tensor &voxel_config);
std::vector<torch::Tensor> sample(const torch::Tensor &seg_points,
                                  const torch::Tensor &seg_logits,
                                  const torch::Tensor &seg_vote_preds,
                                  const torch::Tensor &seg_feats,
                                  const torch::Tensor &batch_idx,
                                  const torch::Tensor &vote_offsets,
                                  const torch::Tensor &score_thresh);
void ha4_launcher(torch::Tensor &I, torch::Tensor &L);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  m.def("spccl", &spccl, "spccl")
      .def("voxel_spccl", &voxel_spccl, "voxel_spccl")
      .def("voxelized_sampling", &voxelized_sampling, "voxelized_sampling")
      .def("sample", &sample, "sample")
      .def("ccl", &ha4_launcher, "ccl");
}
