#include "autotuner.h"
#include "gemm.h"
#include "profiling.h"
#include <iostream>
#include <limits>

namespace ladder {

TuningResult AutoTuner::tuneGEMM(size_t M, size_t N, size_t K,
                                 const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                                 const std::vector<size_t>& tileMs,
                                 const std::vector<size_t>& tileNs,
                                 const std::vector<size_t>& tileKs,
                                 const std::vector<float>* bias,
                                 bool apply_relu,
                                 int repeats) {
    TuningResult best{0,0,0,std::numeric_limits<double>::infinity()};
    Profiler p;
    for (size_t tm : tileMs) for (size_t tn : tileNs) for (size_t tk : tileKs) {
        // skip invalid tile sizes
        if (tm == 0 || tn == 0 || tk == 0) continue;
        double total = 0.0;
        for (int r = 0; r < repeats; ++r) {
            p.start();
            gemm_tiled_fused(M,N,K,A,B,C,tm,tn,tk,bias,apply_relu);
            p.stop();
            total += p.ms();
        }
        double avg = total / repeats;
        std::cout << "tile ("<<tm<<","<<tn<<","<<tk<<") avg ms: "<<avg<<"\n";
        if (avg < best.ms) {
            best = TuningResult{tm,tn,tk,avg};
        }
    }
    std::cout << "Best tile: ("<<best.tileM<<","<<best.tileN<<","<<best.tileK<<") ms="<<best.ms<<"\n";
    return best;
}

} // namespace ladder
