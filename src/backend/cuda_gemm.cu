#include "ladder/cuda_gemm.h"
#include "cuda_gemm_utils.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <vector>
#include <cstdlib>
#include <cstring>

namespace ladder {

static void cublasCheck(cublasStatus_t st) {
  if (st != CUBLAS_STATUS_SUCCESS) {
    // swallow in prototype
  }
}

__global__ void bias_relu_kernel(float *C, const float *bias, int M, int N, int relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = M * N;
  if (idx >= total) return;
  int j = idx % N;
  float v = C[idx];
  if (bias) v += bias[j];
  if (relu && v < 0.0f) v = 0.0f;
  C[idx] = v;
}

// Simple double-buffered shared-memory FP32 kernel (16x16x16)
__global__ void gemm_tiled_fp32_db(const float *A, const float *B, float *C,
                                  int M, int N, int K) {
  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;
  __shared__ float sA[2][BM][BK];
  __shared__ float sB[2][BK][BN];

  int row = blockIdx.y * BM + threadIdx.y;
  int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0.0f;

  int tiles = (K + BK - 1) / BK;
  for (int t = 0; t < tiles; ++t) {
    int buf = t & 1;
    int k0 = t * BK + threadIdx.x;
    int k1 = t * BK + threadIdx.y;

    if (row < M && k0 < K)
      sA[buf][threadIdx.y][threadIdx.x] = A[row * K + k0];
    else
      sA[buf][threadIdx.y][threadIdx.x] = 0.0f;

    if (k1 < K && col < N)
      sB[buf][threadIdx.y][threadIdx.x] = B[k1 * N + col];
    else
      sB[buf][threadIdx.y][threadIdx.x] = 0.0f;

    __syncthreads();
    for (int k = 0; k < BK; ++k)
      acc += sA[buf][threadIdx.y][k] * sB[buf][k][threadIdx.x];
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// FP32 tiled kernel using cp.async (SM80+). Falls back to regular loads if unavailable.
__device__ __forceinline__ void cp_async_4(void *smem_ptr, const void *gmem_ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(smem_ptr), "l"(gmem_ptr));
#else
  *reinterpret_cast<float *>(smem_ptr) = *reinterpret_cast<const float *>(gmem_ptr);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void cp_async_wait(int n = 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  if (n <= 0) asm volatile("cp.async.wait_group 0;");
  else if (n == 1) asm volatile("cp.async.wait_group 1;");
  else if (n == 2) asm volatile("cp.async.wait_group 2;");
  else asm volatile("cp.async.wait_group 3;");
#endif
}

__global__ void gemm_tiled_fp32_async(const float *A, const float *B, float *C,
                                      int M, int N, int K) {
  constexpr int BM = 16;
  constexpr int BN = 16;
  constexpr int BK = 16;
  __shared__ float sA[2][BM][BK];
  __shared__ float sB[2][BK][BN];

  int row = blockIdx.y * BM + threadIdx.y;
  int col = blockIdx.x * BN + threadIdx.x;
  float acc = 0.0f;

  int tiles = (K + BK - 1) / BK;

  auto load_tile = [&](int t, int buf) {
    int k0 = t * BK + threadIdx.x;
    int k1 = t * BK + threadIdx.y;

    if (row < M && k0 < K) {
      const float *gA = A + row * K + k0;
      cp_async_4(&sA[buf][threadIdx.y][threadIdx.x], gA);
    } else {
      sA[buf][threadIdx.y][threadIdx.x] = 0.0f;
    }

    if (k1 < K && col < N) {
      const float *gB = B + k1 * N + col;
      cp_async_4(&sB[buf][threadIdx.y][threadIdx.x], gB);
    } else {
      sB[buf][threadIdx.y][threadIdx.x] = 0.0f;
    }
  };

  // prefetch tile 0 and 1
  load_tile(0, 0);
  if (tiles > 1) load_tile(1, 1);
  cp_async_commit();
  cp_async_wait(0);
  __syncthreads();

  for (int t = 0; t < tiles; ++t) {
    int buf = t & 1;
    for (int k = 0; k < BK; ++k)
      acc += sA[buf][threadIdx.y][k] * sB[buf][k][threadIdx.x];

    int nextTile = t + 2;
    if (nextTile < tiles) {
      int nextBuf = buf;
      load_tile(nextTile, nextBuf);
      cp_async_commit();
      cp_async_wait(1);
    }
    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// WMMA kernel (16x16x16) using Tensor Cores, FP16 inputs, FP32 accumulate
__global__ void gemm_wmma_f16(const half *A, const half *B, float *C,
                              int M, int N, int K, int tileK) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda::wmma;
  int tileRow = blockIdx.y;
  int tileCol = blockIdx.x;
  int row = tileRow * 16;
  int col = tileCol * 16;

  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  int step = tileK > 0 ? tileK : 16;
  for (int k0 = 0; k0 < K; k0 += step) {
    int kend = min(k0 + step, K);
    for (int k = k0; k < kend; k += 16) {
    if (row < M && col < N) {
      const half *aTile = A + row * K + k;
      const half *bTile = B + k * N + col;
      load_matrix_sync(a_frag, aTile, K);
      load_matrix_sync(b_frag, bTile, N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    }
  }

  if (row < M && col < N) {
    float *cTile = C + row * N + col;
    store_matrix_sync(cTile, c_frag, N, mem_row_major);
  }
#endif
}

__global__ void gemm_wmma_f16_bias_relu(const half *A, const half *B, float *C,
                                        const float *bias, int M, int N, int K, int tileK, int relu) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda::wmma;
  int tileRow = blockIdx.y;
  int tileCol = blockIdx.x;
  int row = tileRow * 16;
  int col = tileCol * 16;

  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  int step = tileK > 0 ? tileK : 16;
  for (int k0 = 0; k0 < K; k0 += step) {
    int kend = min(k0 + step, K);
    for (int k = k0; k < kend; k += 16) {
    if (row < M && col < N) {
      const half *aTile = A + row * K + k;
      const half *bTile = B + k * N + col;
      load_matrix_sync(a_frag, aTile, K);
      load_matrix_sync(b_frag, bTile, N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    }
  }

  __shared__ float cTile[16 * 16];
  store_matrix_sync(cTile, c_frag, 16, mem_row_major);
  __syncthreads();

  int tid = threadIdx.x;
  for (int idx = tid; idx < 256; idx += 32) {
    int i = idx / 16;
    int j = idx % 16;
    int gr = row + i;
    int gc = col + j;
    if (gr < M && gc < N) {
      float v = cTile[idx];
      if (bias) v += bias[gc];
      if (relu && v < 0.0f) v = 0.0f;
      C[gr * N + gc] = v;
    }
  }
#endif
}

// 2x2 WMMA tiles per block (4 warps)
__global__ void gemm_wmma_f16_bias_relu_2x2(const half *A, const half *B, float *C,
                                           const float *bias, int M, int N, int K, int tileK, int relu) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  using namespace nvcuda::wmma;
  int warpId = threadIdx.x / 32;
  if (warpId >= 4) return;

  int tileRow = blockIdx.y * 2 + (warpId / 2);
  int tileCol = blockIdx.x * 2 + (warpId % 2);
  int row = tileRow * 16;
  int col = tileCol * 16;

  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);

  int step = tileK > 0 ? tileK : 16;
  for (int k0 = 0; k0 < K; k0 += step) {
    int kend = min(k0 + step, K);
    for (int k = k0; k < kend; k += 16) {
      if (row < M && col < N) {
        const half *aTile = A + row * K + k;
        const half *bTile = B + k * N + col;
        load_matrix_sync(a_frag, aTile, K);
        load_matrix_sync(b_frag, bTile, N);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
  }

  __shared__ float cTile[4][16 * 16];
  store_matrix_sync(cTile[warpId], c_frag, 16, mem_row_major);
  __syncthreads();

  int tid = threadIdx.x % 32;
  for (int idx = tid; idx < 256; idx += 32) {
    int i = idx / 16;
    int j = idx % 16;
    int gr = row + i;
    int gc = col + j;
    if (gr < M && gc < N) {
      float v = cTile[warpId][idx];
      if (bias) v += bias[gc];
      if (relu && v < 0.0f) v = 0.0f;
      C[gr * N + gc] = v;
    }
  }
#endif
}

static void launch_bias_relu(float *C, const float *bias, int M, int N, bool relu, cudaStream_t stream) {
  int total = M * N;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  bias_relu_kernel<<<blocks, threads, 0, stream>>>(C, bias, M, N, relu ? 1 : 0);
}

bool gemm_tiled_fused_tiles_cuda(size_t M, size_t N, size_t K,
                                const std::vector<float>& A,
                                const std::vector<float>& B,
                                std::vector<float>& C,
                                const std::vector<float>* bias,
                                bool apply_relu,
                                tType::Precision precision,
                                size_t tileM,
                                size_t tileN,
                                size_t tileK) {
  if (A.empty() || B.empty() || C.empty()) return false;

  cudaStream_t stream = nullptr;
  ladder_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cublasHandle_t handle;
  cublasCheck(cublasCreate(&handle));
  cublasCheck(cublasSetStream(handle, stream));

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dBias = nullptr;
  size_t bytesA = A.size() * sizeof(float);
  size_t bytesB = B.size() * sizeof(float);
  size_t bytesC = C.size() * sizeof(float);

  ladder_cuda_check(cudaMalloc(&dA, bytesA));
  ladder_cuda_check(cudaMalloc(&dB, bytesB));
  ladder_cuda_check(cudaMalloc(&dC, bytesC));
  ladder_cuda_check(cudaMemcpyAsync(dA, A.data(), bytesA, cudaMemcpyHostToDevice, stream));
  ladder_cuda_check(cudaMemcpyAsync(dB, B.data(), bytesB, cudaMemcpyHostToDevice, stream));

  if (bias) {
    ladder_cuda_check(cudaMalloc(&dBias, bias->size() * sizeof(float)));
    ladder_cuda_check(cudaMemcpyAsync(dBias, bias->data(), bias->size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  const char *tf32Env = std::getenv("LADDER_USE_TF32");
  const char *tileEnv = std::getenv("LADDER_USE_TILED_KERNEL");
  const char *asyncEnv = std::getenv("LADDER_USE_ASYNC_COPY");
  const char *wmmaEnv = std::getenv("LADDER_USE_WMMA");
  const char *wmmaTilesEnv = std::getenv("LADDER_WMMA_TILES_PER_BLOCK");
  const char *presetEnv = std::getenv("LADDER_PRESET");

  bool useTf32 = tf32Env ? std::atoi(tf32Env) != 0 : true;
  bool useTiledKernel = tileEnv ? std::atoi(tileEnv) != 0 : false;
  bool useAsyncCopy = asyncEnv ? std::atoi(asyncEnv) != 0 : false;
  bool useWmma = wmmaEnv ? std::atoi(wmmaEnv) != 0 : false;
  int wmmaTilesPerBlock = wmmaTilesEnv ? std::atoi(wmmaTilesEnv) : 0;

  // Preset overrides (if provided)
  if (presetEnv) {
    if (std::strcmp(presetEnv, "throughput") == 0) {
      useWmma = true;
      useAsyncCopy = true;
      useTf32 = true;
      wmmaTilesPerBlock = 4;
    } else if (std::strcmp(presetEnv, "accuracy") == 0) {
      useWmma = false;
      useAsyncCopy = true;
      useTf32 = false;
      wmmaTilesPerBlock = 1;
    } else if (std::strcmp(presetEnv, "small") == 0) {
      useWmma = true;
      useAsyncCopy = false;
      useTf32 = true;
      wmmaTilesPerBlock = 1;
    } else if (std::strcmp(presetEnv, "large") == 0) {
      useWmma = true;
      useAsyncCopy = true;
      useTf32 = true;
      wmmaTilesPerBlock = 4;
    }
  } else {
    // Heuristic defaults when env not set
    bool large = (M >= 256 && N >= 256 && K >= 256);
    bool small = (M <= 128 || N <= 128 || K <= 128);
    if (!wmmaEnv) useWmma = (precision == tType::Precision::FP16 || precision == tType::Precision::FP8);
    if (!asyncEnv) useAsyncCopy = large;
    if (!tf32Env) useTf32 = true;
    if (!wmmaTilesEnv) wmmaTilesPerBlock = large ? 4 : 1;
    if (small) {
      useAsyncCopy = false;
      if (!wmmaTilesEnv) wmmaTilesPerBlock = 1;
    }
  }
  if (tileK > 0 && tileK % 16 != 0)
    useWmma = false;
  if (tileM > 0 && tileN > 0 && tileK > 0) {
    // Allow tile parameters to drive kernel selection
    useTiledKernel = useTiledKernel || (tileM <= 32 && tileN <= 32 && tileK <= 32);
  }

  int bm = (tileM >= 16 && tileM <= 32) ? (int)tileM : 16;
  int bn = (tileN >= 16 && tileN <= 32) ? (int)tileN : 16;
  if (precision == tType::Precision::FP32 && useTiledKernel && (bm % 16 == 0) && (bn % 16 == 0)) {
    dim3 block(bn, bm, 1);
    dim3 grid((N + bn - 1) / bn, (M + bm - 1) / bm, 1);
    if (useAsyncCopy) {
      gemm_tiled_fp32_async<<<grid, block, 0, stream>>>(dA, dB, dC, (int)M, (int)N, (int)K);
    } else {
      gemm_tiled_fp32_db<<<grid, block, 0, stream>>>(dA, dB, dC, (int)M, (int)N, (int)K);
    }
  } else if (precision == tType::Precision::FP16 || precision == tType::Precision::FP8) {
    std::vector<__half> Ah(A.size());
    std::vector<__half> Bh(B.size());
    for (size_t i = 0; i < A.size(); ++i) Ah[i] = __float2half(A[i]);
    for (size_t i = 0; i < B.size(); ++i) Bh[i] = __float2half(B[i]);

    __half *dAh = nullptr, *dBh = nullptr;
    ladder_cuda_check(cudaMalloc(&dAh, Ah.size() * sizeof(__half)));
    ladder_cuda_check(cudaMalloc(&dBh, Bh.size() * sizeof(__half)));
    ladder_cuda_check(cudaMemcpyAsync(dAh, Ah.data(), Ah.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
    ladder_cuda_check(cudaMemcpyAsync(dBh, Bh.data(), Bh.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));

    if (useWmma && (M % 16 == 0) && (N % 16 == 0) && (K % 16 == 0)) {
      if (wmmaTilesPerBlock <= 0) {
        // Heuristic: prefer 2x2 tiles if tileM/N >= 32 and shared memory fits.
        int sharedPerWarpTile = 16 * 16 * sizeof(half) * 2; // A + B
        int shared2x2 = sharedPerWarpTile * 4;
        wmmaTilesPerBlock = (tileM >= 32 && tileN >= 32 && shared2x2 <= 64 * 1024) ? 4 : 1;
      }
      if (wmmaTilesPerBlock >= 4 && tileM >= 32 && tileN >= 32) {
        dim3 block(128, 1, 1);
        dim3 grid((N + 31) / 32, (M + 31) / 32, 1);
        gemm_wmma_f16_bias_relu_2x2<<<grid, block, 0, stream>>>(dAh, dBh, dC, dBias, (int)M, (int)N, (int)K, (int)tileK, apply_relu ? 1 : 0);
      } else {
        dim3 block(32, 1, 1);
        dim3 grid((N + 15) / 16, (M + 15) / 16, 1);
        gemm_wmma_f16_bias_relu<<<grid, block, 0, stream>>>(dAh, dBh, dC, dBias, (int)M, (int)N, (int)K, (int)tileK, apply_relu ? 1 : 0);
      }
    } else {
      cublasCheck(cublasGemmEx(handle,
                               CUBLAS_OP_N, CUBLAS_OP_N,
                               (int)N, (int)M, (int)K,
                               &alpha,
                               dBh, CUDA_R_16F, (int)N,
                               dAh, CUDA_R_16F, (int)K,
                               &beta,
                               dC, CUDA_R_32F, (int)N,
                               CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    ladder_cuda_check(cudaFree(dAh));
    ladder_cuda_check(cudaFree(dBh));
  } else {
    if (useTf32) {
      cublasCheck(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
      cublasCheck(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              (int)N, (int)M, (int)K,
                              &alpha,
                              dB, CUDA_R_32F, (int)N,
                              dA, CUDA_R_32F, (int)K,
                              &beta,
                              dC, CUDA_R_32F, (int)N,
                              CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      cublasCheck(cublasSgemm(handle,
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            (int)N, (int)M, (int)K,
                            &alpha,
                            dB, (int)N,
                            dA, (int)K,
                            &beta,
                            dC, (int)N));
    }
  }

  if (!(useWmma && (precision == tType::Precision::FP16 || precision == tType::Precision::FP8))) {
    launch_bias_relu(dC, dBias, (int)M, (int)N, apply_relu, stream);
  }
  ladder_cuda_check(cudaMemcpyAsync(C.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream));
  ladder_cuda_check(cudaStreamSynchronize(stream));

  if (dBias) ladder_cuda_check(cudaFree(dBias));
  ladder_cuda_check(cudaFree(dA));
  ladder_cuda_check(cudaFree(dB));
  ladder_cuda_check(cudaFree(dC));
  cublasCheck(cublasDestroy(handle));
  ladder_cuda_check(cudaStreamDestroy(stream));
  return true;
}

__global__ void scale_output_kernel(float *C, int total, float scale, int relu) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = C[idx] * scale;
  if (relu && v < 0.0f) v = 0.0f;
  C[idx] = v;
}

bool gemm_tiled_fused_tiles_quantized_cuda(size_t M, size_t N, size_t K,
                                           const std::vector<float>& A,
                                           const std::vector<float>& B,
                                           std::vector<float>& C,
                                           const QuantParams& q,
                                           const std::vector<float>* bias,
                                           bool apply_relu) {
  if (A.empty() || B.empty() || C.empty()) return false;

  cudaStream_t stream = nullptr;
  ladder_cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  cublasHandle_t handle;
  cublasCheck(cublasCreate(&handle));
  cublasCheck(cublasSetStream(handle, stream));

  float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dBias = nullptr;
  size_t bytesA = A.size() * sizeof(float);
  size_t bytesB = B.size() * sizeof(float);
  size_t bytesC = C.size() * sizeof(float);

  ladder_cuda_check(cudaMalloc(&dA, bytesA));
  ladder_cuda_check(cudaMalloc(&dB, bytesB));
  ladder_cuda_check(cudaMalloc(&dC, bytesC));
  ladder_cuda_check(cudaMemcpyAsync(dA, A.data(), bytesA, cudaMemcpyHostToDevice, stream));
  ladder_cuda_check(cudaMemcpyAsync(dB, B.data(), bytesB, cudaMemcpyHostToDevice, stream));

  if (bias) {
    ladder_cuda_check(cudaMalloc(&dBias, bias->size() * sizeof(float)));
    ladder_cuda_check(cudaMemcpyAsync(dBias, bias->data(), bias->size() * sizeof(float), cudaMemcpyHostToDevice, stream));
  }

  const float alpha = q.input_scale * q.input_scale;
  const float beta = 0.0f;

  cublasCheck(cublasSgemm(handle,
                          CUBLAS_OP_N, CUBLAS_OP_N,
                          (int)N, (int)M, (int)K,
                          &alpha,
                          dB, (int)N,
                          dA, (int)K,
                          &beta,
                          dC, (int)N));

  if (bias) {
    launch_bias_relu(dC, dBias, (int)M, (int)N, apply_relu, stream);
  } else if (apply_relu || q.output_scale != 1.0f) {
    int total = (int)(M * N);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    scale_output_kernel<<<blocks, threads, 0, stream>>>(dC, total, q.output_scale, apply_relu ? 1 : 0);
  }

  ladder_cuda_check(cudaMemcpyAsync(C.data(), dC, bytesC, cudaMemcpyDeviceToHost, stream));
  ladder_cuda_check(cudaStreamSynchronize(stream));

  if (dBias) ladder_cuda_check(cudaFree(dBias));
  ladder_cuda_check(cudaFree(dA));
  ladder_cuda_check(cudaFree(dB));
  ladder_cuda_check(cudaFree(dC));
  cublasCheck(cublasDestroy(handle));
  ladder_cuda_check(cudaStreamDestroy(stream));
  return true;
}

} // namespace ladder
