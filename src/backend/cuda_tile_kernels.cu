#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void ladder_extract_2d_kernel(const float *in, float *out,
                                                    int H, int W,
                                                    int off0, int off1,
                                                    int size0, int size1,
                                                    int stride0, int stride1) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = size0 * size1;
  if (idx >= total) return;
  int i = idx / size1;
  int j = idx % size1;
  int src_i = off0 + i * stride0;
  int src_j = off1 + j * stride1;
  if (src_i < 0 || src_i >= H || src_j < 0 || src_j >= W) return;
  out[idx] = in[src_i * W + src_j];
}

extern "C" __global__ void ladder_map_relu_kernel(float *data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float v = data[idx];
  data[idx] = v > 0.0f ? v : 0.0f;
}

extern "C" __global__ void ladder_pad_2d_kernel(const float *in, float *out,
                                                int H, int W,
                                                int pad_low0, int pad_low1,
                                                int outH, int outW,
                                                float pad_value) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = outH * outW;
  if (idx >= total) return;
  int i = idx / outW;
  int j = idx % outW;
  int src_i = i - pad_low0;
  int src_j = j - pad_low1;
  if (src_i >= 0 && src_i < H && src_j >= 0 && src_j < W) {
    out[idx] = in[src_i * W + src_j];
  } else {
    out[idx] = pad_value;
  }
}

extern "C" __global__ void ladder_transpose_2d_kernel(const float *in, float *out,
                                                      int H, int W) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = H * W;
  if (idx >= total) return;
  int i = idx / W;
  int j = idx % W;
  out[j * H + i] = in[i * W + j];
}

extern "C" void ladder_cuda_launch_extract_2d(const float *in, float *out,
                                              int H, int W,
                                              int off0, int off1,
                                              int size0, int size1,
                                              int stride0, int stride1,
                                              cudaStream_t stream) {
  int total = size0 * size1;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ladder_extract_2d_kernel<<<blocks, threads, 0, stream>>>(in, out, H, W, off0, off1,
                                                           size0, size1, stride0, stride1);
}

extern "C" void ladder_cuda_launch_map_relu(float *data, int n, cudaStream_t stream) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  ladder_map_relu_kernel<<<blocks, threads, 0, stream>>>(data, n);
}

extern "C" void ladder_cuda_launch_pad_2d(const float *in, float *out,
                                          int H, int W,
                                          int pad_low0, int pad_low1,
                                          int pad_high0, int pad_high1,
                                          float pad_value,
                                          cudaStream_t stream) {
  int outH = H + pad_low0 + pad_high0;
  int outW = W + pad_low1 + pad_high1;
  int total = outH * outW;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ladder_pad_2d_kernel<<<blocks, threads, 0, stream>>>(in, out, H, W, pad_low0, pad_low1, outH, outW, pad_value);
}

extern "C" void ladder_cuda_launch_transpose_2d(const float *in, float *out,
                                                int H, int W,
                                                cudaStream_t stream) {
  int total = H * W;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  ladder_transpose_2d_kernel<<<blocks, threads, 0, stream>>>(in, out, H, W);
}
