#include "ladder/cuda_gemm.h"
#include "cuda_gemm_utils.cuh"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <vector>

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
                                tType::Precision precision) {
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

  if (precision == tType::Precision::FP16 || precision == tType::Precision::FP8) {
    std::vector<__half> Ah(A.size());
    std::vector<__half> Bh(B.size());
    for (size_t i = 0; i < A.size(); ++i) Ah[i] = __float2half(A[i]);
    for (size_t i = 0; i < B.size(); ++i) Bh[i] = __float2half(B[i]);

    __half *dAh = nullptr, *dBh = nullptr;
    ladder_cuda_check(cudaMalloc(&dAh, Ah.size() * sizeof(__half)));
    ladder_cuda_check(cudaMalloc(&dBh, Bh.size() * sizeof(__half)));
    ladder_cuda_check(cudaMemcpyAsync(dAh, Ah.data(), Ah.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));
    ladder_cuda_check(cudaMemcpyAsync(dBh, Bh.data(), Bh.size() * sizeof(__half), cudaMemcpyHostToDevice, stream));

    cublasCheck(cublasGemmEx(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             (int)N, (int)M, (int)K,
                             &alpha,
                             dBh, CUDA_R_16F, (int)N,
                             dAh, CUDA_R_16F, (int)K,
                             &beta,
                             dC, CUDA_R_32F, (int)N,
                             CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    ladder_cuda_check(cudaFree(dAh));
    ladder_cuda_check(cudaFree(dBh));
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

  launch_bias_relu(dC, dBias, (int)M, (int)N, apply_relu, stream);
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
