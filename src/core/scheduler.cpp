#include "scheduler.h"
#include <algorithm>
#include <cmath>
#include <set>

namespace ladder {

std::tuple<size_t,size_t,size_t> Scheduler::recommendTiles(size_t M, size_t N, size_t K, const DeviceInfo& dev) {
    // Very simple heuristic:
    // aim for tile bytes footprint ~ L1/4, assuming float32 (4 bytes)
    size_t target_bytes = dev.l1_size / 4;
    size_t elem_bytes = 4; // float
    // naive choice: tileM = tileN = sqrt(target_bytes / elem_bytes)
    size_t tile_side = std::max((size_t)8, (size_t)std::sqrt((double)target_bytes / elem_bytes));
    size_t tileM = std::min(tile_side, M);
    size_t tileN = std::min(tile_side, N);
    size_t tileK = std::min((size_t)32, K);
    return {tileM, tileN, tileK};
}

double Scheduler::estimateGemmCost(size_t M, size_t N, size_t K,
                                   size_t tileM, size_t tileN, size_t tileK,
                                   const DeviceInfo& dev) {
    if (tileM == 0 || tileN == 0 || tileK == 0) return 1e30;

    const double bytes = (double)(tileM * tileK + tileK * tileN + tileM * tileN) * 4.0;
    const double flops = 2.0 * (double)tileM * (double)tileN * (double)tileK;

    double mem_time = bytes / (dev.mem_bandwidth_gbs * 1e9);
    double compute_time = flops / (dev.peak_flops_tflops * 1e12);
    double cost = std::max(mem_time, compute_time);

    // Penalty if tile footprint exceeds shared memory per SM or L1 budget
    double l1_penalty = bytes > dev.l1_size ? (bytes / (double)dev.l1_size) : 1.0;
    double smem_penalty = bytes > dev.shared_mem_per_sm ? (bytes / (double)dev.shared_mem_per_sm) : 1.0;
    cost *= std::max(l1_penalty, smem_penalty);

    // Rough occupancy penalty if tile looks too large for a block
    if (tileM * tileN > dev.max_threads_per_block)
        cost *= 1.5;

    // Scale by global size to prefer larger tiles for large problems
    double scale = std::sqrt((double)(M * N) / (double)(tileM * tileN));
    return cost * scale;
}

std::vector<TileCandidate> Scheduler::rankCandidates(size_t M, size_t N, size_t K,
                                                     const DeviceInfo& dev,
                                                     size_t maxCandidates) {
    auto base = recommendTiles(M, N, K, dev);
    size_t baseM, baseN, baseK;
    std::tie(baseM, baseN, baseK) = base;

    std::vector<TileCandidate> candidates;
    std::set<std::tuple<size_t,size_t,size_t>> seen;
    std::vector<double> scales = {0.5, 0.75, 1.0, 1.25, 1.5, 2.0};
    std::vector<double> kscales = {0.5, 1.0, 2.0};

    for (double sM : scales) {
        for (double sN : scales) {
            for (double sK : kscales) {
                size_t tm = std::max((size_t)1, (size_t)(baseM * sM));
                size_t tn = std::max((size_t)1, (size_t)(baseN * sN));
                size_t tk = std::max((size_t)1, (size_t)(baseK * sK));
                tm = std::min(tm, M);
                tn = std::min(tn, N);
                tk = std::min(tk, K);
                auto tpl = std::make_tuple(tm, tn, tk);
                if (!seen.insert(tpl).second)
                    continue;

                double score = estimateGemmCost(M, N, K, tm, tn, tk, dev);
                candidates.push_back({tm, tn, tk, score});
            }
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const TileCandidate& a, const TileCandidate& b) { return a.score < b.score; });

    if (candidates.size() > maxCandidates)
        candidates.resize(maxCandidates);
    return candidates;
}

} // namespace ladder
