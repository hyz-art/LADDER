#include <iostream>
#include <vector>
#include <random>
#include "gemm.h"
#include "autotuner.h"

int main() {
    size_t M = 256, N = 256, K = 256;
    std::vector<float> A(M*K), B(K*N), C(M*N);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);

    ladder::AutoTuner tuner;
    std::vector<size_t> tileMs = {8,16,32};
    std::vector<size_t> tileNs = {8,16,32};
    std::vector<size_t> tileKs = {8,16,32};

    auto best = tuner.tuneGEMM(M,N,K,A,B,C,tileMs,tileNs,tileKs,nullptr,true,2);
    std::cout << "Selected best tile: ("<<best.tileM<<","<<best.tileN<<","<<best.tileK<<") ms="<<best.ms<<"\n";
    return 0;
}
