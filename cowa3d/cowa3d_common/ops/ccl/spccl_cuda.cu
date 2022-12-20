#include "hash32.h"
#include "macros.h"
#include <torch/extension.h>

DEVICE_INLINE void merge(int* L, int label1, int label2)
{
  // find the root node of label1
  while (label1 != label2 && label1 != L[label1]) {
    int Llabel1 = L[label1];
    if (label1 == Llabel1) break;
    label1 = Llabel1;
  }

  // find the root node of label2
  while (label1 != label2 && label2 != L[label2]) {
    int Llabel2 = L[label2];
    if (label2 == Llabel2) break;
    label2 = Llabel2;
  }
  // assign the root node of label2 as the parent node of the root node
  // of label1, assuming that label2 is smaller than label1
  while (label1 != label2) {
    if (label1 < label2) {
      int tmp = label1;
      label1 = label2;
      label2 = tmp;
    }
    int label3 = atomicMin(&L[label1], label2);
    if (label1 == label3) { label1 = label2; }
    else {
      // if the the root node of label1 changes then
      // incorporate this change and continue the loop
      label1 = label3;
    }
  }
}

__global__ void fill_hash_kernel(int32_t* serialized,
                                 int32_t* equal_table,
                                 int num_coors,
                                 hash::LinearHashMap<int32_t, int32_t> coor2id)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x) {
    coor2id.insert_if_empty(serialized[i], i);
    equal_table[i] = i;
  }
}

__global__ void build_equal_tree(int32_t* serialized,
                                 int32_t* equal_table,
                                 int num_coors,
                                 hash::LinearHashMap<int32_t, int32_t> coor2id)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x) {
    int serialized_coor = serialized[i];
    int self_i;
    coor2id.lookup(serialized_coor, self_i);
    if (i != self_i) {
      equal_table[i] = self_i;
      continue;
    }

    int x = serialized_coor & 0x3fff;
    int y = (serialized_coor >> 14) & 0x3fff;
    int nb;
    if ((x >= 2 && coor2id.lookup(serialized_coor - 0x0002, nb)) ||
        (x >= 1 && coor2id.lookup(serialized_coor - 0x0001, nb)))
      merge(equal_table, i, nb);
    if ((x >= 1 && y >= 1 && coor2id.lookup(serialized_coor - 0x4001, nb)) ||
        (y >= 1 && coor2id.lookup(serialized_coor - 0x4000, nb)) ||
        (y >= 1 && coor2id.lookup(serialized_coor - 0x3fff, nb)))
      merge(equal_table, i, nb);
    if ((y >= 2 && coor2id.lookup(serialized_coor - 0x8000, nb))) merge(equal_table, i, nb);
  }
}

__global__ void uniquify_label(int32_t* serialized, int32_t* equal_table, int32_t* count, int num_coors)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x) {
    int serialized_coor = serialized[i];
    int batch_id = serialized_coor >> 28;
    if (i == equal_table[i]) { 
      equal_table[i] = atomicAdd(count + batch_id, 1);
    }
    else{
      equal_table[i] = ~equal_table[i];
    }
  }
}

__global__ void relabel(int32_t* equal_table, int num_coors)
{
  for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x) {
    int label = ~i;
    while (label < 0)
      label = equal_table[~label];
    
    equal_table[i] = label;
  }
}

torch::Tensor spccl(const torch::Tensor& coors)
{
  using hash = hash::LinearHashMap<int32_t, int32_t>;
  auto b = coors.index({"...", 0});
  auto y = coors.index({"...", 1});
  auto x = coors.index({"...", 2});
  auto serialized_coors = b * (1 << 28) + y * (1 << 14) + x;
  serialized_coors = serialized_coors.to({torch::kInt32});
  int num_coors = serialized_coors.size(0);
  int num_hash = 2 * num_coors;
  torch::Tensor coors_hash_key =
      torch::full({num_hash}, hash::EMPTY, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
  torch::Tensor coors_hash_val = torch::empty_like(coors_hash_key);
  torch::Tensor equal_table = torch::full_like(serialized_coors, -1);
  torch::Tensor count = torch::zeros({16}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

  hash coors_hash(coors_hash_key.data_ptr<int32_t>(), coors_hash_val.data_ptr<int32_t>(), num_hash);

  fill_hash_kernel<<<(num_coors + 255) / 256, 256>>>(serialized_coors.data_ptr<int32_t>(),
                                                     equal_table.data_ptr<int32_t>(), num_coors, coors_hash);

  build_equal_tree<<<(num_coors + 255) / 256, 256>>>(serialized_coors.data_ptr<int32_t>(),
                                                     equal_table.data_ptr<int32_t>(), num_coors, coors_hash);

  uniquify_label<<<(num_coors + 255) / 256, 256>>>(
      serialized_coors.data_ptr<int32_t>(), equal_table.data_ptr<int32_t>(), count.data_ptr<int32_t>(), num_coors);
  
  relabel<<<(num_coors + 255) / 256, 256>>>(equal_table.data_ptr<int32_t>(), num_coors);

  return equal_table;
}
