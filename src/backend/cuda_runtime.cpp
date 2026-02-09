#include "ladder/runtime_cuda.h"

#include <cstring>
#include <cstdint>

namespace {

static bool isGlobalToLocal(int64_t direction) {
  return direction == 0;
}

} // namespace

extern "C" ladder::tTile *ladder_cuda_tile_copy(ladder::tTile *input,
                                                int64_t direction,
                                                int64_t level,
                                                int64_t stream,
                                                int64_t event,
                                                int64_t layout_kind,
                                                int64_t layout_param0,
                                                int64_t layout_param1) {
  (void)level;
  if (!input) return nullptr;
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  if (isGlobalToLocal(direction)) {
    auto out = input->copyToLocal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                  reinterpret_cast<ladder::tTile::CudaEvent>(event));
    out.setLayout(layoutKind, layoutParams);
    return new ladder::tTile(std::move(out));
  }
  auto out = input->copyToGlobal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                 reinterpret_cast<ladder::tTile::CudaEvent>(event));
  out.setLayout(layoutKind, layoutParams);
  return new ladder::tTile(std::move(out));
}

extern "C" ladder::tTile *ladder_cuda_tile_prefetch(ladder::tTile *input,
                                                    int64_t direction,
                                                    int64_t level,
                                                    int64_t stream,
                                                    int64_t event,
                                                    int64_t layout_kind,
                                                    int64_t layout_param0,
                                                    int64_t layout_param1) {
  (void)direction;
  if (!input) return nullptr;
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  input->prefetch(static_cast<int>(level),
                  reinterpret_cast<ladder::tTile::CudaStream>(stream),
                  reinterpret_cast<ladder::tTile::CudaEvent>(event));
  auto out = new ladder::tTile(*input);
  out->setLayout(layoutKind, layoutParams);
  return out;
}

extern "C" ladder::tTile *ladder_cuda_tile_async_copy(ladder::tTile *input,
                                                      int64_t direction,
                                                      int64_t level,
                                                      int64_t stream,
                                                      int64_t event,
                                                      int64_t layout_kind,
                                                      int64_t layout_param0,
                                                      int64_t layout_param1) {
  (void)level;
  if (!input) return nullptr;
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  if (isGlobalToLocal(direction)) {
    auto out = input->asyncCopyToLocal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                       reinterpret_cast<ladder::tTile::CudaEvent>(event));
    out.setLayout(layoutKind, layoutParams);
    return new ladder::tTile(std::move(out));
  }
  auto out = input->asyncCopyToGlobal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                      reinterpret_cast<ladder::tTile::CudaEvent>(event));
  out.setLayout(layoutKind, layoutParams);
  return new ladder::tTile(std::move(out));
}

// CUDA kernel launchers (defined in cuda_tile_kernels.cu)
#ifdef LADDER_ENABLE_CUDA
extern "C" void ladder_cuda_launch_extract_2d(const float *in, float *out,
                                              int H, int W,
                                              int off0, int off1,
                                              int size0, int size1,
                                              int stride0, int stride1,
                                              cudaStream_t stream);
extern "C" void ladder_cuda_launch_map_relu(float *data, int n, cudaStream_t stream);
extern "C" void ladder_cuda_launch_pad_2d(const float *in, float *out,
                                          int H, int W,
                                          int pad_low0, int pad_low1,
                                          int pad_high0, int pad_high1,
                                          float pad_value,
                                          cudaStream_t stream);
extern "C" void ladder_cuda_launch_transpose_2d(const float *in, float *out,
                                                int H, int W,
                                                cudaStream_t stream);
#endif

extern "C" ladder::tTile *ladder_cuda_tile_extract(ladder::tTile *input,
                                                   int64_t off0,
                                                   int64_t off1,
                                                   int64_t size0,
                                                   int64_t size1,
                                                   int64_t stride0,
                                                   int64_t stride1,
                                                   int64_t stream,
                                                   int64_t event,
                                                   int64_t layout_kind,
                                                   int64_t layout_param0,
                                                   int64_t layout_param1) {
  if (!input) return nullptr;
  std::vector<size_t> shape = input->shape();
  int H = shape.size() > 0 ? static_cast<int>(shape[0]) : 0;
  int W = shape.size() > 1 ? static_cast<int>(shape[1]) : 0;
  int outH = static_cast<int>(size0);
  int outW = static_cast<int>(size1);
  auto out = new ladder::tTile({static_cast<size_t>(outH), static_cast<size_t>(outW)});
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  out->setLayout(layoutKind, layoutParams);

#ifdef LADDER_ENABLE_CUDA
  input->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                      reinterpret_cast<ladder::tTile::CudaEvent>(event));
  out->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                    reinterpret_cast<ladder::tTile::CudaEvent>(event));
  ladder_cuda_launch_extract_2d(reinterpret_cast<const float *>(input->devicePtr()),
                                reinterpret_cast<float *>(out->devicePtr()),
                                H, W,
                                static_cast<int>(off0), static_cast<int>(off1),
                                outH, outW,
                                static_cast<int>(stride0), static_cast<int>(stride1),
                                reinterpret_cast<ladder::tTile::CudaStream>(stream));
#else
  *out = input->slice(0, static_cast<size_t>(off0), static_cast<size_t>(size0));
#endif
  return out;
}

extern "C" ladder::tTile *ladder_cuda_tile_map(ladder::tTile *input,
                                               int64_t map_fn,
                                               int64_t axis0,
                                               int64_t axis1,
                                               int64_t stream,
                                               int64_t event,
                                               int64_t layout_kind,
                                               int64_t layout_param0,
                                               int64_t layout_param1) {
  (void)axis0;
  (void)axis1;
  if (!input) return nullptr;
  auto out = new ladder::tTile(*input);
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  out->setLayout(layoutKind, layoutParams);

#ifdef LADDER_ENABLE_CUDA
  out->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                    reinterpret_cast<ladder::tTile::CudaEvent>(event));
  if (map_fn == 1) {
    ladder_cuda_launch_map_relu(reinterpret_cast<float *>(out->devicePtr()),
                                static_cast<int>(out->size()),
                                reinterpret_cast<ladder::tTile::CudaStream>(stream));
  }
#else
  if (map_fn == 1) {
    out->map([](float v) { return v > 0.0f ? v : 0.0f; });
  }
#endif
  return out;
}

extern "C" ladder::tTile *ladder_cuda_tile_pad(ladder::tTile *input,
                                               int64_t pad_low0,
                                               int64_t pad_low1,
                                               int64_t pad_high0,
                                               int64_t pad_high1,
                                               int64_t pad_inner0,
                                               int64_t pad_inner1,
                                               int64_t pad_value_bits,
                                               int64_t stream,
                                               int64_t event,
                                               int64_t layout_kind,
                                               int64_t layout_param0,
                                               int64_t layout_param1) {
  (void)pad_inner0;
  (void)pad_inner1;
  if (!input) return nullptr;
  std::vector<size_t> shape = input->shape();
  int H = shape.size() > 0 ? static_cast<int>(shape[0]) : 0;
  int W = shape.size() > 1 ? static_cast<int>(shape[1]) : 0;
  int outH = H + static_cast<int>(pad_low0) + static_cast<int>(pad_high0);
  int outW = W + static_cast<int>(pad_low1) + static_cast<int>(pad_high1);
  auto out = new ladder::tTile({static_cast<size_t>(outH), static_cast<size_t>(outW)});
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));
  out->setLayout(layoutKind, layoutParams);

  float padValue = *reinterpret_cast<float *>(&pad_value_bits);
#ifdef LADDER_ENABLE_CUDA
  input->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                      reinterpret_cast<ladder::tTile::CudaEvent>(event));
  out->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                    reinterpret_cast<ladder::tTile::CudaEvent>(event));
  ladder_cuda_launch_pad_2d(reinterpret_cast<const float *>(input->devicePtr()),
                            reinterpret_cast<float *>(out->devicePtr()),
                            H, W,
                            static_cast<int>(pad_low0), static_cast<int>(pad_low1),
                            static_cast<int>(pad_high0), static_cast<int>(pad_high1),
                            padValue,
                            reinterpret_cast<ladder::tTile::CudaStream>(stream));
#else
  *out = *input;
  out->pad(0, static_cast<size_t>(pad_low0), static_cast<size_t>(pad_high0), padValue);
#endif
  return out;
}

extern "C" ladder::tTile *ladder_cuda_tile_transform(ladder::tTile *input,
                                                     int64_t transform_kind,
                                                     int64_t param0,
                                                     int64_t param1,
                                                     int64_t stream,
                                                     int64_t event,
                                                     int64_t layout_kind,
                                                     int64_t layout_param0,
                                                     int64_t layout_param1) {
  (void)param0;
  (void)param1;
  if (!input) return nullptr;
  auto layoutKind = static_cast<ladder::tTile::LayoutKind>(layout_kind);
  std::vector<size_t> layoutParams;
  if (layout_param0 > 0) layoutParams.push_back(static_cast<size_t>(layout_param0));
  if (layout_param1 > 0) layoutParams.push_back(static_cast<size_t>(layout_param1));

  std::vector<size_t> shape = input->shape();
  if (shape.size() < 2 || transform_kind == 0) {
    auto out = new ladder::tTile(*input);
    out->setLayout(layoutKind, layoutParams);
    return out;
  }

  int H = static_cast<int>(shape[0]);
  int W = static_cast<int>(shape[1]);
  auto out = new ladder::tTile({static_cast<size_t>(W), static_cast<size_t>(H)});
  out->setLayout(layoutKind, layoutParams);

#ifdef LADDER_ENABLE_CUDA
  input->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                      reinterpret_cast<ladder::tTile::CudaEvent>(event));
  out->ensureDevice(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                    reinterpret_cast<ladder::tTile::CudaEvent>(event));
  ladder_cuda_launch_transpose_2d(reinterpret_cast<const float *>(input->devicePtr()),
                                  reinterpret_cast<float *>(out->devicePtr()),
                                  H, W,
                                  reinterpret_cast<ladder::tTile::CudaStream>(stream));
#else
  *out = *input;
#endif
  return out;
}

extern "C" void ladder_cuda_tile_free(ladder::tTile *tile) { delete tile; }
