#pragma once
#include <cuda_runtime.h>

static inline void ladder_cuda_check(cudaError_t err) {
  if (err != cudaSuccess) {
    cudaGetLastError();
  }
}
