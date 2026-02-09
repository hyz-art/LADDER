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
                                                int64_t stream,
                                                int64_t event) {
  if (!input) return nullptr;
  if (isGlobalToLocal(direction)) {
    auto out = input->copyToLocal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                  reinterpret_cast<ladder::tTile::CudaEvent>(event));
    return new ladder::tTile(std::move(out));
  }
  auto out = input->copyToGlobal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                 reinterpret_cast<ladder::tTile::CudaEvent>(event));
  return new ladder::tTile(std::move(out));
}

extern "C" ladder::tTile *ladder_cuda_tile_prefetch(ladder::tTile *input,
                                                    int64_t level,
                                                    int64_t stream,
                                                    int64_t event) {
  if (!input) return nullptr;
  input->prefetch(static_cast<int>(level),
                  reinterpret_cast<ladder::tTile::CudaStream>(stream),
                  reinterpret_cast<ladder::tTile::CudaEvent>(event));
  return new ladder::tTile(*input);
}

extern "C" ladder::tTile *ladder_cuda_tile_async_copy(ladder::tTile *input,
                                                      int64_t direction,
                                                      int64_t stream,
                                                      int64_t event) {
  if (!input) return nullptr;
  if (isGlobalToLocal(direction)) {
    auto out = input->asyncCopyToLocal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                       reinterpret_cast<ladder::tTile::CudaEvent>(event));
    return new ladder::tTile(std::move(out));
  }
  auto out = input->asyncCopyToGlobal(reinterpret_cast<ladder::tTile::CudaStream>(stream),
                                      reinterpret_cast<ladder::tTile::CudaEvent>(event));
  return new ladder::tTile(std::move(out));
}

extern "C" void ladder_cuda_tile_free(ladder::tTile *tile) { delete tile; }
