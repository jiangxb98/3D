#include <torch/extension.h>

#define DEVICE_INLINE __device__ __forceinline__
constexpr int warp_size = 32;
constexpr int block_warps = 4;

DEVICE_INLINE int warp_idx()
{
  return threadIdx.x & (warp_size - 1);
}

DEVICE_INLINE int start_distance(uint32_t pixels_masks, int pixels_idx)
{
  return __clz(~(pixels_masks << (32 - pixels_idx)));
}

DEVICE_INLINE int end_distance(uint32_t pixels_masks, int pixels_idx)
{
  return __ffs(~(pixels_masks >> (pixels_idx + 1)));
}

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

__global__ void ha4_strip_labeling(uint8_t* I, int* L, int W, int H)
{
  __shared__ uint32_t shared_pixels[block_warps];
  for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < H; y += gridDim.x * blockDim.y) {
    int distance_y = 0;
    int distance_ym1 = 0;
    for (int x = threadIdx.x; x < W; x += blockDim.x) {
      int k_yx = y * W + x;
      uint8_t P_yx = I[k_yx];
      uint32_t pixels_y = __ballot_sync(0xffffffff, P_yx);
      int s_dist_y = start_distance(pixels_y, threadIdx.x);
      if (P_yx && s_dist_y == 0) {
        if (threadIdx.x == 0) { L[k_yx] = k_yx - distance_y; }
        else {
          L[k_yx] = k_yx;
        }
      }
      if (threadIdx.x == 0) { shared_pixels[threadIdx.y] = pixels_y; }
      __syncthreads();
      uint32_t pixels_ym1 = threadIdx.y > 0 ? shared_pixels[threadIdx.y - 1] : 0U;
      uint8_t P_ym1x = (pixels_ym1 & (1 << threadIdx.x)) ? 1 : 0;
      int s_dist_ym1 = start_distance(pixels_ym1, threadIdx.x);
      if (threadIdx.x == 0) {
        s_dist_y = distance_y;
        s_dist_ym1 = distance_ym1;
      }
      if (P_yx && P_ym1x && (s_dist_y == 0 || s_dist_ym1 == 0)) {
        int label1 = k_yx - s_dist_y;
        int label2 = k_yx - W - s_dist_ym1;
        merge(L, label1, label2);
      }
      int d = start_distance(pixels_ym1, warp_size);
      distance_ym1 = d == warp_size ? warp_size + distance_ym1 : d;
      d = start_distance(pixels_y, warp_size);
      distance_y = d == warp_size ? warp_size + distance_y : d;
    }
  }
}

__global__ void ha4_strip_merge(uint8_t* I, int* L, int W, int H)
{
  for (int y = block_warps * (1 + blockIdx.x); y < H; y += block_warps * gridDim.x) {
    int distance_y = 0;
    int distance_ym1 = 0;
    for (int x = threadIdx.x; x < W; x += blockDim.x) {
      int k_yx = y * W + x;
      int k_ym1x = k_yx - W;
      uint8_t P_yx = I[k_yx];
      uint8_t P_ym1x = I[k_ym1x];
      uint32_t pixels_y = __ballot_sync(0xffffffff, P_yx);
      uint32_t pixels_ym1 = __ballot_sync(0xffffffff, P_ym1x);
      if (P_yx && P_ym1x) {
        int s_dist_y = start_distance(pixels_y, threadIdx.x);
        int s_dist_ym1 = start_distance(pixels_ym1, threadIdx.x);
        if (threadIdx.x == 0) {
          s_dist_y = distance_y;
          s_dist_ym1 = distance_ym1;
        }
        if (s_dist_y == 0 || s_dist_ym1 == 0) { merge(L, k_yx - s_dist_y, k_ym1x - s_dist_ym1); }
        int d = start_distance(pixels_ym1, warp_size);
        distance_ym1 = d == warp_size ? warp_size + distance_ym1 : d;
        d = start_distance(pixels_y, warp_size);
        distance_y = d == warp_size ? warp_size + distance_y : d;
      }
    }
  }
}

__global__ void ha4_relabeling(uint8_t* I, int* L, int W, int H)
{
  for (int y = threadIdx.y + blockIdx.x * blockDim.y; y < H; y += gridDim.x * blockDim.y) {
    int distance_y = 0;
    for (int x = threadIdx.x; x < W; x += blockDim.x) {
      int k_yx = y * W + x;
      uint8_t P_yx = I[k_yx];
      uint32_t pixels_y = __ballot_sync(0xffffffff, P_yx);
      int s_dist_y = start_distance(pixels_y, threadIdx.x);
      if (threadIdx.x == 0) { s_dist_y = distance_y; }
      int label = 0;
      if (P_yx && s_dist_y == 0) {
        label = L[k_yx];
        while (label != L[label]) { label = L[label]; }
      }
      label = __shfl_sync(0xffffffff, label, threadIdx.x - s_dist_y);
      if (P_yx) { L[k_yx] = label; }
      int d = start_distance(pixels_y, warp_size);
      distance_y = d == warp_size ? warp_size + distance_y : d;
    }
  }
}

void ha4_launcher(torch::Tensor& I, torch::Tensor& L)
{
  assert(I.scalar_type() == torch::kUInt8);
  assert(L.scalar_type() == torch::kInt32);
  assert(I.size(0) == L.size(0) && I.size(1) == L.size(1) && I.dim() == L.dim() && I.dim() == 2);
  int W = L.size(1);
  int H = L.size(0);
  assert(W % warp_size == 0 && H % block_warps == 0);
  ha4_strip_labeling<<<H / block_warps, dim3{warp_size, block_warps}>>>(I.data_ptr<uint8_t>(), L.data_ptr<int>(), W, H);
  ha4_strip_merge<<<H / block_warps - 1, warp_size>>>(I.data_ptr<uint8_t>(), L.data_ptr<int>(), W, H);
  ha4_relabeling<<<H / block_warps, dim3{warp_size, block_warps}>>>(I.data_ptr<uint8_t>(), L.data_ptr<int>(), W, H);
}
