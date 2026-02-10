#include <iostream>
#include <random>
#include "autotuner.h"
#include "scheduler.h"

int main() {
    using namespace ladder;
    size_t M = 128, N = 128, K = 64;
    std::vector<float> A(M*K), B(K*N), C(M*N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);

    AutoTuner tuner;
    DeviceInfo dev; // defaults
    std::cout << "DeviceInfo: L1="<<dev.l1_size<<" L2="<<dev.l2_size
              <<" SM="<<dev.sm_count<<" warp="<<dev.warp_size
              <<" bw(GB/s)="<<dev.mem_bandwidth_gbs
              <<" peak(TF)="<<dev.peak_flops_tflops<<"\n";
    std::vector<tType::Precision> precisions = { tType::Precision::FP32, tType::Precision::FP16, tType::Precision::FP8 };
    std::vector<bool> fuse_opts = { false, true };

    std::cout << "Running autotune (small search)...\n";
    auto res = tuner.tuneGEMMWithScheduler(M,N,K,A,B,C,dev,precisions,fuse_opts,12,nullptr,2);
    std::cout << "Autotune best ms="<<res.ms<<" tile=("<<res.tileM<<","<<res.tileN<<","<<res.tileK<<")\n";

    // try to plot (if python script available)
    std::cout << "You can run: python3 tools/plot_results.py to generate autotune_summary.png\n";
    return 0;
}
