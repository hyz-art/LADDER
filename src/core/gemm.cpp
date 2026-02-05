#include "gemm.h"
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

void gemm_tiled_fused(size_t M, size_t N, size_t K,
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

} // namespace ladder
