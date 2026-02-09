#pragma once

#include "ttile.h"
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// C-style runtime APIs for tile ops (CUDA-enabled if LADDER_ENABLE_CUDA)
// direction: 0=global_to_local, 1=local_to_global
// layout_kind: 0=row_major, 1=blocked, 2=packed
// layout_param0/1: layout-specific parameters (e.g. block sizes)
// stream/event are CUDA handles when CUDA enabled; otherwise ignored (pass 0).
ladder::tTile *ladder_cuda_tile_copy(ladder::tTile *input,
                                                                        int64_t direction,
                                    int64_t level,
                                                                        int64_t stream,
                                                                        int64_t event,
                                                                        int64_t layout_kind,
                                                                        int64_t layout_param0,
                                                                        int64_t layout_param1);

ladder::tTile *ladder_cuda_tile_prefetch(ladder::tTile *input,
                                        int64_t direction,
                                                                                int64_t level,
                                                                                int64_t stream,
                                                                                int64_t event,
                                                                                int64_t layout_kind,
                                                                                int64_t layout_param0,
                                                                                int64_t layout_param1);

ladder::tTile *ladder_cuda_tile_async_copy(ladder::tTile *input,
                                                                                    int64_t direction,
                                          int64_t level,
                                                                                    int64_t stream,
                                                                                    int64_t event,
                                                                                    int64_t layout_kind,
                                                                                    int64_t layout_param0,
                                                                                    int64_t layout_param1);

ladder::tTile *ladder_cuda_tile_extract(ladder::tTile *input,
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
                                       int64_t layout_param1);

// map_fn: 0=identity, 1=relu
ladder::tTile *ladder_cuda_tile_map(ladder::tTile *input,
                                   int64_t map_fn,
                                   int64_t axis0,
                                   int64_t axis1,
                                   int64_t stream,
                                   int64_t event,
                                   int64_t layout_kind,
                                   int64_t layout_param0,
                                   int64_t layout_param1);

// pad_value_bits: bitcast float bits in int64
ladder::tTile *ladder_cuda_tile_pad(ladder::tTile *input,
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
                                   int64_t layout_param1);

// transform_kind: 0=identity, 1=transpose(2D)
ladder::tTile *ladder_cuda_tile_transform(ladder::tTile *input,
                                         int64_t transform_kind,
                                         int64_t param0,
                                         int64_t param1,
                                         int64_t stream,
                                         int64_t event,
                                         int64_t layout_kind,
                                         int64_t layout_param0,
                                         int64_t layout_param1);

void ladder_cuda_tile_free(ladder::tTile *tile);

#ifdef __cplusplus
}
#endif
