#include "gemm.h"
#include "ttile.h"
#ifdef LADDER_ENABLE_CUDA
#include "ladder/cuda_gemm.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace ladder {

void gemm_naive(size_t M, size_t N, size_t K,
                const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C) {
    std::fill(C.begin(), C.end(), 0.0f);
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            float a = A[i*K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }
}

static inline float quantize_sim(float v, tType::Precision p) {
    // crude simulation of lower precision by clamping/rounding
    switch (p) {
        case tType::Precision::FP32: return v;
        case tType::Precision::FP16: return std::roundf(v * 128.0f) / 128.0f;
        case tType::Precision::FP8:  return std::roundf(v * 16.0f) / 16.0f;
        case tType::Precision::INT8: return std::roundf(v);
        case tType::Precision::INT4: { float q = std::roundf(v*2.0f); return q/2.0f; }
    }
    return v;
}

// Existing loop-based tiled fused implementation (kept as separate function)
void gemm_tiled_fused_loop(size_t M, size_t N, size_t K,
                           const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                           size_t tileM, size_t tileN, size_t tileK,
                           const std::vector<float>* bias,
                           bool apply_relu,
                           tType::Precision precision) {
    std::fill(C.begin(), C.end(), 0.0f);
    // iterate over tiles
    for (size_t ii = 0; ii < M; ii += tileM) {
        size_t mm = std::min(tileM, M - ii);
        for (size_t jj = 0; jj < N; jj += tileN) {
            size_t nn = std::min(tileN, N - jj);
            // local accumulation
            for (size_t kk = 0; kk < K; kk += tileK) {
                size_t kk_len = std::min(tileK, K - kk);
                for (size_t i = 0; i < mm; ++i) {
                    for (size_t k = 0; k < kk_len; ++k) {
                        float a = A[(ii + i)*K + (kk + k)];
                        a = quantize_sim(a, precision);
                        for (size_t j = 0; j < nn; ++j) {
                            float b = B[(kk + k)*N + (jj + j)];
                            b = quantize_sim(b, precision);
                            C[(ii + i)*N + (jj + j)] += a * b;
                        }
                    }
                }
            }
            // apply fusion: bias add + relu per tile
            for (size_t i = 0; i < mm; ++i) {
                for (size_t j = 0; j < nn; ++j) {
                    size_t off = (ii + i)*N + (jj + j);
                    if (bias) C[off] += (*bias)[jj + j];
                    if (apply_relu && C[off] < 0.0f) C[off] = 0.0f;
                    // simulate output precision
                    C[off] = quantize_sim(C[off], precision);
                }
            }
        }
    }
}

// Tile-level implementation using tTile to make tile ops explicit
void gemm_tiled_fused_tiles(size_t M, size_t N, size_t K,
                           const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                           size_t tileM, size_t tileN, size_t tileK,
                           const std::vector<float>* bias,
                           bool apply_relu,
                           tType::Precision precision) {
#ifdef LADDER_ENABLE_CUDA
    if (gemm_tiled_fused_tiles_cuda(M, N, K, A, B, C, bias, apply_relu, precision)) {
        return;
    }
#endif
    std::fill(C.begin(), C.end(), 0.0f);
    // iterate over tiles and use tTile for local operations
    for (size_t ii = 0; ii < M; ii += tileM) {
        size_t mm = std::min(tileM, M - ii);
        for (size_t jj = 0; jj < N; jj += tileN) {
            size_t nn = std::min(tileN, N - jj);
            // initialize C_tile
            tTile C_tile({mm, nn});
            for (size_t kk = 0; kk < K; kk += tileK) {
                size_t kk_len = std::min(tileK, K - kk);
                // extract tiles A_tile (mm x kk_len) and B_tile (kk_len x nn)
                tTile A_tile({mm, kk_len});
                tTile B_tile({kk_len, nn});
                // copy data into tiles
                for (size_t i = 0; i < mm; ++i) {
                    for (size_t k = 0; k < kk_len; ++k) {
                        A_tile.data()[i*kk_len + k] = A[(ii + i)*K + (kk + k)];
                    }
                }
                for (size_t k = 0; k < kk_len; ++k) {
                    for (size_t j = 0; j < nn; ++j) {
                        B_tile.data()[k*nn + j] = B[(kk + k)*N + (jj + j)];
                    }
                }

                // compute local matmul on tile and accumulate into C_tile
                for (size_t i = 0; i < mm; ++i) {
                    for (size_t k = 0; k < kk_len; ++k) {
                        float a = quantize_sim(A_tile.data()[i*kk_len + k], precision);
                        for (size_t j = 0; j < nn; ++j) {
                            float b = quantize_sim(B_tile.data()[k*nn + j], precision);
                            C_tile.data()[i*nn + j] += a * b;
                        }
                    }
                }
            }
            // apply fusion and write back
            for (size_t i = 0; i < mm; ++i) {
                for (size_t j = 0; j < nn; ++j) {
                    size_t off = (ii + i)*N + (jj + j);
                    if (bias) C_tile.data()[i*nn + j] += (*bias)[jj + j];
                    if (apply_relu && C_tile.data()[i*nn + j] < 0.0f) C_tile.data()[i*nn + j] = 0.0f;
                    C[off] = quantize_sim(C_tile.data()[i*nn + j], precision);
                }
            }
        }
    }
}

void gemm_tiled_fused_tiles_quantized(size_t M, size_t N, size_t K,
                                      const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                                      size_t tileM, size_t tileN, size_t tileK,
                                      const QuantParams& q,
                                      const std::vector<float>* bias,
                                      bool apply_relu) {
#ifdef LADDER_ENABLE_CUDA
    if (gemm_tiled_fused_tiles_quantized_cuda(M, N, K, A, B, C, q, bias, apply_relu)) {
        return;
    }
#endif
    std::fill(C.begin(), C.end(), 0.0f);
    for (size_t ii = 0; ii < M; ii += tileM) {
        size_t mm = std::min(tileM, M - ii);
        for (size_t jj = 0; jj < N; jj += tileN) {
            size_t nn = std::min(tileN, N - jj);
            tTile C_tile({mm, nn});
            for (size_t kk = 0; kk < K; kk += tileK) {
                size_t kk_len = std::min(tileK, K - kk);
                tTile A_tile({mm, kk_len});
                tTile B_tile({kk_len, nn});
                for (size_t i = 0; i < mm; ++i) {
                    for (size_t k = 0; k < kk_len; ++k) {
                        float v = A[(ii + i)*K + (kk + k)] * q.input_scale;
                        A_tile.data()[i*kk_len + k] = quantize_sim(v, q.compute);
                    }
                }
                for (size_t k = 0; k < kk_len; ++k) {
                    for (size_t j = 0; j < nn; ++j) {
                        float v = B[(kk + k)*N + (jj + j)] * q.input_scale;
                        B_tile.data()[k*nn + j] = quantize_sim(v, q.compute);
                    }
                }

                for (size_t i = 0; i < mm; ++i) {
                    for (size_t k = 0; k < kk_len; ++k) {
                        float a = A_tile.data()[i*kk_len + k];
                        for (size_t j = 0; j < nn; ++j) {
                            float b = B_tile.data()[k*nn + j];
                            float acc = C_tile.data()[i*nn + j] + a * b;
                            if (q.accumulate != tType::Precision::FP32) {
                                acc = quantize_sim(acc, q.accumulate);
                            }
                            C_tile.data()[i*nn + j] = acc;
                        }
                    }
                }
            }
            for (size_t i = 0; i < mm; ++i) {
                for (size_t j = 0; j < nn; ++j) {
                    size_t off = (ii + i)*N + (jj + j);
                    float v = C_tile.data()[i*nn + j];
                    if (bias) v += (*bias)[jj + j];
                    if (apply_relu && v < 0.0f) v = 0.0f;
                    v *= q.output_scale;
                    C[off] = quantize_sim(v, q.compute);
                }
            }
        }
    }
}

// Dispatcher that selects implementation
void gemm_tiled_fused(size_t M, size_t N, size_t K,
                      const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                      size_t tileM, size_t tileN, size_t tileK,
                      GemmImpl impl,
                      const std::vector<float>* bias,
                      bool apply_relu,
                      tType::Precision precision) {
    if (impl == GemmImpl::TTile) {
        gemm_tiled_fused_tiles(M,N,K,A,B,C,tileM,tileN,tileK,bias,apply_relu,precision);
    } else {
        gemm_tiled_fused_loop(M,N,K,A,B,C,tileM,tileN,tileK,bias,apply_relu,precision);
    }
}

// Backwards-compatible overload (default to loop impl)
void gemm_tiled_fused(size_t M, size_t N, size_t K,
                      const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                      size_t tileM, size_t tileN, size_t tileK,
                      const std::vector<float>* bias,
                      bool apply_relu,
                      tType::Precision precision) {
    gemm_tiled_fused(M,N,K,A,B,C,tileM,tileN,tileK,GemmImpl::Loop,bias,apply_relu,precision);
}

} // namespace ladder
