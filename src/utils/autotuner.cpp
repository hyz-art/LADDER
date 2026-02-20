#include "autotuner.h"
#include "gemm.h"
#include "profiling.h"
#include <iostream>
#include <limits>
#include <fstream>
#include <iomanip>
#include "scheduler.h"
#include <set>

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
    std::ofstream out("autotune_results.csv", std::ios::app);
    if (out) out << "prec,fuse,tm,tn,tk,ms\n";
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
        if (out) out << "FP32,"<< (apply_relu?1:0) <<","<<tm<<","<<tn<<","<<tk<<","<<std::fixed<<std::setprecision(6)<<avg<<"\n";
    }
    std::cout << "Best tile: ("<<best.tileM<<","<<best.tileN<<","<<best.tileK<<") ms="<<best.ms<<"\n";
    return best;
}

TuningResult AutoTuner::tuneGEMMAdvanced(size_t M, size_t N, size_t K,
                                         const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                                         const std::vector<size_t>& tileMs,
                                         const std::vector<size_t>& tileNs,
                                         const std::vector<size_t>& tileKs,
                                         const std::vector<tType::Precision>& precisions,
                                         const std::vector<bool>& fuse_relu_options,
                                         const std::vector<float>* bias,
                                         int repeats) {
    TuningResult best{0,0,0,std::numeric_limits<double>::infinity()};
    Profiler p;
    std::ofstream out("autotune_results.csv", std::ios::app);
    if (out) out << "prec,fuse,tm,tn,tk,ms\n";
    for (auto prec : precisions) {
        for (bool fuse_relu : fuse_relu_options) {
            for (size_t tm : tileMs) for (size_t tn : tileNs) for (size_t tk : tileKs) {
                if (tm == 0 || tn == 0 || tk == 0) continue;
                double total = 0.0;
                for (int r = 0; r < repeats; ++r) {
                    p.start();
                    gemm_tiled_fused(M,N,K,A,B,C,tm,tn,tk,bias,fuse_relu,prec);
                    p.stop();
                    total += p.ms();
                }
                double avg = total / repeats;
                std::cout << "prec("<< (int)prec <<") fuse("<<fuse_relu<<") tile ("<<tm<<","<<tn<<","<<tk<<") avg ms: "<<avg<<"\n";
                if (avg < best.ms) {
                    best = TuningResult{tm,tn,tk,avg};
                }
                if (out) out << (int)prec << "," << (fuse_relu?1:0) << ","<<tm<<","<<tn<<","<<tk<<","<<std::fixed<<std::setprecision(6)<<avg<<"\n";
            }
        }
    }
    std::cout << "Best tile: ("<<best.tileM<<","<<best.tileN<<","<<best.tileK<<") ms="<<best.ms<<"\n";
    return best;
}

TuningResult AutoTuner::tuneGEMMWithScheduler(size_t M, size_t N, size_t K,
                                              const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C,
                                              const DeviceInfo& dev,
                                              const std::vector<tType::Precision>& precisions,
                                              const std::vector<bool>& fuse_relu_options,
                                              size_t numCandidates,
                                              const std::vector<float>* bias,
                                              int repeats) {
    // Generate and rank candidates with cost model
    auto ranked = Scheduler::rankCandidates(M, N, K, dev, std::max<size_t>(numCandidates, 8));
    std::vector<size_t> tileMs, tileNs, tileKs;
    for (auto &c : ranked) {
        tileMs.push_back(c.tileM);
        tileNs.push_back(c.tileN);
        tileKs.push_back(c.tileK);
    }

    return tuneGEMMAdvanced(M,N,K,A,B,C,tileMs,tileNs,tileKs,precisions,fuse_relu_options,bias,repeats);
}

} // namespace ladder
