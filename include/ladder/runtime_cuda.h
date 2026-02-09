#pragma once

#include "ttile.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// C-style runtime APIs for tile ops (CUDA-enabled if LADDER_ENABLE_CUDA)
// direction: 0=global_to_local, 1=local_to_global
// stream/event are CUDA handles when CUDA enabled; otherwise ignored (pass 0).
ladder::tTile *ladder_cuda_tile_copy(ladder::tTile *input,
                                                                        int64_t direction,
                                                                        int64_t stream,
                                                                        int64_t event);

ladder::tTile *ladder_cuda_tile_prefetch(ladder::tTile *input,
                                                                                int64_t level,
                                                                                int64_t stream,
                                                                                int64_t event);

ladder::tTile *ladder_cuda_tile_async_copy(ladder::tTile *input,
                                                                                    int64_t direction,
                                                                                    int64_t stream,
                                                                                    int64_t event);

void ladder_cuda_tile_free(ladder::tTile *tile);

#ifdef __cplusplus
}
#endif
