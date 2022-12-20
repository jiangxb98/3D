#include "hash32.h"
#include "macros.h"
#include <torch/extension.h>

DEVICE_INLINE void merge(int *L, int label1, int label2)
{
    // find the root node of label1
    while (label1 != label2 && label1 != L[label1])
    {
        int Llabel1 = L[label1];
        if (label1 == Llabel1)
            break;
        label1 = Llabel1;
    }

    // find the root node of label2
    while (label1 != label2 && label2 != L[label2])
    {
        int Llabel2 = L[label2];
        if (label2 == Llabel2)
            break;
        label2 = Llabel2;
    }
    // assign the root node of label2 as the parent node of the root node
    // of label1, assuming that label2 is smaller than label1
    while (label1 != label2)
    {
        if (label1 < label2)
        {
            int tmp = label1;
            label1 = label2;
            label2 = tmp;
        }
        int label3 = atomicMin(&L[label1], label2);
        if (label1 == label3)
        {
            label1 = label2;
        }
        else
        {
            // if the the root node of label1 changes then
            // incorporate this change and continue the loop
            label1 = label3;
        }
    }
}

__global__ void fill_hash_kernel(float *point_ptr,
                                 int32_t *batch_id,
                                 int32_t *class_id,
                                 float *voxel_config,
                                 bool *valid_mask,
                                 int32_t *num_points,
                                 int32_t *points_to_coors,
                                 int32_t *serialized,
                                 int32_t *equal_table,
                                 int num_coors,
                                 hash::LinearHashMap<int32_t, int32_t> coor2id)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x)
    {
        float x = point_ptr[3 * i];
        float y = point_ptr[3 * i + 1];
        int b = batch_id[i];
        int c = class_id[i];
        float x_min = voxel_config[6 * c];
        float y_min = voxel_config[6 * c + 1];
        float dx = voxel_config[6 * c + 3];
        float dy = voxel_config[6 * c + 4];

        int coor_x = (x - x_min) / dx;
        int coor_y = (y - y_min) / dy;
        if (coor_x < 0 || coor_y < 0)
        {
            valid_mask[i] = false;
            continue;
        }
        serialized[i] = c * (1 << 28) + b * (1 << 24) + coor_y * (1 << 12) + coor_x;
        int j = coor2id.insert_if_empty(serialized[i], i);
        assert(j != coor2id.EMPTY);
        atomicAdd(&num_points[j], 1);
        points_to_coors[i] = j;

        equal_table[i] = i;
    }
}

__global__ void filter_almost_empty(bool *valid_mask,
                                    int32_t *num_points,
                                    int32_t *points_to_coors,
                                    int32_t *serialized,
                                    int32_t *equal_table,
                                    int min_points,
                                    int num_coors)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x)
    {
        if (!valid_mask[i])
            continue;
        int coors_id = points_to_coors[i];
        if (num_points[coors_id] < min_points)
        {
            valid_mask[i] = false;
        }
    }
}

__global__ void build_equal_tree(int32_t *serialized,
                                 int32_t *equal_table,
                                 bool *valid_mask,
                                 int num_coors,
                                 hash::LinearHashMap<int32_t, int32_t> coor2id)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x)
    {
        if (!valid_mask[i])
            continue;
        int serialized_coor = serialized[i];
        int self_i;
        coor2id.lookup(serialized_coor, self_i);
        if (i != self_i)
        {
            equal_table[i] = self_i;
            continue;
        }

        int x = serialized_coor & 0x0fff;
        int y = (serialized_coor >> 12) & 0x0fff;
        int nb;
        if ((x >= 2 && coor2id.lookup(serialized_coor - 0x0002, nb) && valid_mask[nb]) ||
            (x >= 1 && coor2id.lookup(serialized_coor - 0x0001, nb) && valid_mask[nb]))
        {
            merge(equal_table, i, nb);
        }

        if ((x >= 1 && y >= 1 && coor2id.lookup(serialized_coor - 0x1001, nb) && valid_mask[nb]) ||
            (y >= 1 && coor2id.lookup(serialized_coor - 0x1000, nb) && valid_mask[nb]) ||
            (y >= 1 && coor2id.lookup(serialized_coor - 0x0fff, nb) && valid_mask[nb]))
        {
            merge(equal_table, i, nb);
        }

        if ((y >= 2 && coor2id.lookup(serialized_coor - 0x2000, nb) && valid_mask[nb]))
        {
            merge(equal_table, i, nb);
        }
    }
}

__global__ void uniquify_label(int32_t *serialized, int32_t *equal_table, int32_t *equal_table_unique,
                               int32_t *count, bool *valid_mask, int num_coors, int num_batch_size)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x)
    {
        if (!valid_mask[i])
            continue;
        // int serialized_coor = serialized[i];
        // int batch_id = (serialized_coor >> 24) & 0xf;
        // int class_id = serialized_coor >> 28;
        if (i == equal_table[i])
        {
            // equal_table_unique[i] = atomicAdd(&count[batch_id + num_batch_size * class_id], 1);
            equal_table_unique[i] = atomicAdd(&count[0], 1);
        }
    }
}

__global__ void relabel(int32_t *equal_table, int32_t *equal_table_unique, bool *valid_mask, int num_coors)
{
    for (int i = threadIdx.x + blockDim.x * blockIdx.x; i < num_coors; i += blockDim.x * gridDim.x)
    {
        if (!valid_mask[i])
        {
            equal_table_unique[i] = -1;
            continue;
        }
        int label = i;
        while (label != equal_table[label])
        {
            label = equal_table[label];
        }
        equal_table_unique[i] = equal_table_unique[label];
    }
}

torch::Tensor voxel_spccl(const torch::Tensor &points,
                          const torch::Tensor &batch_id,
                          const torch::Tensor &class_id,
                          const torch::Tensor &voxel_config,
                          const int min_points,
                          const int num_batch_size)
{
    using hash = hash::LinearHashMap<int32_t, int32_t>;
    int num_class = voxel_config.size(0);
    int num_coors = points.size(0);
    int num_hash = 2 * num_coors;
    torch::Tensor serialized_coors =
        torch::zeros({num_coors}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor valid_mask = torch::ones_like(serialized_coors, {torch::kBool});

    torch::Tensor coors_hash_key =
        torch::full({num_hash}, hash::EMPTY, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor coors_hash_val = torch::empty_like(coors_hash_key);
    torch::Tensor num_points = torch::zeros_like(coors_hash_key, {torch::kInt32});
    torch::Tensor points_to_coors = torch::empty_like(serialized_coors);

    torch::Tensor equal_table = torch::empty_like(serialized_coors);
    torch::Tensor equal_table_unique = torch::empty_like(serialized_coors);
    // torch::Tensor count = torch::zeros({num_class, num_batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    torch::Tensor count = torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    hash coors_hash(coors_hash_key.data_ptr<int32_t>(), coors_hash_val.data_ptr<int32_t>(), num_hash);

    fill_hash_kernel<<<(num_coors + 255) / 256, 256>>>(
        points.data_ptr<float>(), batch_id.data_ptr<int32_t>(), class_id.data_ptr<int32_t>(),
        voxel_config.data_ptr<float>(), valid_mask.data_ptr<bool>(), num_points.data_ptr<int32_t>(),
        points_to_coors.data_ptr<int32_t>(), serialized_coors.data_ptr<int32_t>(), equal_table.data_ptr<int32_t>(),
        num_coors, coors_hash);

    filter_almost_empty<<<(num_coors + 255) / 256, 256>>>(
        valid_mask.data_ptr<bool>(), num_points.data_ptr<int32_t>(), points_to_coors.data_ptr<int32_t>(),
        serialized_coors.data_ptr<int32_t>(), equal_table.data_ptr<int32_t>(), min_points, num_coors);

    build_equal_tree<<<(num_coors + 255) / 256, 256>>>(serialized_coors.data_ptr<int32_t>(),
                                                       equal_table.data_ptr<int32_t>(),
                                                       valid_mask.data_ptr<bool>(),
                                                       num_coors, coors_hash);

    uniquify_label<<<(num_coors + 255) / 256, 256>>>(serialized_coors.data_ptr<int32_t>(),
                                                     equal_table.data_ptr<int32_t>(),
                                                     equal_table_unique.data_ptr<int32_t>(),
                                                     count.data_ptr<int32_t>(),
                                                     valid_mask.data_ptr<bool>(),
                                                     num_coors, num_batch_size);

    relabel<<<(num_coors + 255) / 256, 256>>>(equal_table.data_ptr<int32_t>(), equal_table_unique.data_ptr<int32_t>(),
                                              valid_mask.data_ptr<bool>(), num_coors);
    // cudaDeviceSynchronize();
    // printf("finish\n");
    return equal_table_unique;
}
