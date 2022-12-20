#pragma once
#include <assert.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>
#define CRERROR std::cout
#if defined(__CUDACC__)
#  define HOST_DEVICE_INLINE __forceinline__ __device__ __host__
#  define DEVICE_INLINE __forceinline__ __device__
#  define HOST_DEVICE __device__ __host__
#  define ASSERT(expr) assert(expr)
#  define CHECK_CUDA_ERR()                                                                                             \
    {                                                                                                                  \
      auto err = cudaGetLastError();                                                                                   \
      if (err != cudaSuccess) {                                                                                        \
        const char* errstr = cudaGetErrorString(err);                                                                  \
        std::stringstream __macro_s;                                                                                   \
        __macro_s << __FILE__ << " " << __LINE__ << "\n";                                                              \
        __macro_s << "cuda execution failed with error: " << int(err) << ": " << errstr << "\n";                       \
        CRERROR << __macro_s.str();                                                                                    \
      }                                                                                                                \
    }
#  define CHECK_RETURN_STATUS(err)                                                                                     \
    {                                                                                                                  \
      if (err != cudaSuccess) {                                                                                        \
        const char* errstr = cudaGetErrorString(err);                                                                  \
        std::stringstream __macro_s;                                                                                   \
        __macro_s << __FILE__ << " " << __LINE__ << "\n";                                                              \
        __macro_s << "cuda execution failed with error: " << int(err) << ": " << errstr << "\n";                       \
        CRERROR << __macro_s.str();                                                                                    \
      }                                                                                                                \
    }
#else
#  define ASSERT(x) assert(x)
#  define HOST_DEVICE_INLINE inline
#  define HOST_DEVICE
#endif
namespace utils {
template <class SStream, class T> void sstream_print(SStream& ss, T val) {
  ss << val;
}

template <class SStream, class T, class... TArgs> void sstream_print(SStream& ss, T val, TArgs... args) {
  ss << val << " ";
  sstream_print(ss, args...);
}

struct GPU {
  GPU(cudaStream_t stream = nullptr, cublasHandle_t cublas = nullptr) : mStream(stream), mCublasH(cublas) {}
  virtual cudaStream_t getStream() const { return mStream; }
  virtual cublasHandle_t getCublasHandle() const { return mCublasH; }
  cudaStream_t mStream = nullptr;
  cublasHandle_t mCublasH = nullptr;
};

struct CPU {};
};  // namespace utils

#define ASSERT_RT_ERR(expr, ...)                                                                                       \
  {                                                                                                                    \
    if (!(expr)) {                                                                                                     \
      std::stringstream __macro_s;                                                                                     \
      __macro_s << __FILE__ << " " << __LINE__ << "\n";                                                                \
      __macro_s << #expr << " assert faild. ";                                                                         \
      utils::sstream_print(__macro_s, __VA_ARGS__);                                                                    \
      __macro_s << "\n";                                                                                               \
      CRERROR << __macro_s.str();                                                                                      \
    }                                                                                                                  \
  }
