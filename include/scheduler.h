#pragma once
#include <cstddef>
#include <tuple>
#include <vector>

namespace ladder {

struct DeviceInfo {
    size_t l1_size = 32 * 1024; // bytes
    size_t l2_size = 256 * 1024;
    size_t shared_mem_per_sm = 64 * 1024;
    size_t max_threads_per_sm = 2048;
    size_t max_threads_per_block = 1024;
    size_t sm_count = 1;
    size_t warp_size = 32;
    double mem_bandwidth_gbs = 300.0; // GB/s
    double peak_flops_tflops = 10.0;  // TFLOPS
};

struct TileCandidate {
    size_t tileM;
    size_t tileN;
    size_t tileK;
    double score;
};

class Scheduler {
public:
    // Recommend (tileM, tileN, tileK) based on matrix dims and device info
    static std::tuple<size_t,size_t,size_t> recommendTiles(size_t M, size_t N, size_t K, const DeviceInfo& dev);

    // Generate a pool of candidates and rank by cost model
    static std::vector<TileCandidate> rankCandidates(size_t M, size_t N, size_t K,
                                                     const DeviceInfo& dev,
                                                     size_t maxCandidates = 32);

    // Cost model for a tile triple (lower is better)
    static double estimateGemmCost(size_t M, size_t N, size_t K,
                                   size_t tileM, size_t tileN, size_t tileK,
                                   const DeviceInfo& dev);
};

} // namespace ladder
