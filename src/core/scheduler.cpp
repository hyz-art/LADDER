#include "scheduler.h"
#include <algorithm>

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

} // namespace ladder
#include "scheduler.h"
#include <iostream>

namespace ladder {

class GenericScheduler : public Scheduler {
public:
    std::string name() const override { return "GenericScheduler"; }
    void schedule() override { std::cout << "Running generic schedule...\n"; }
};

// factory for convenience
std::unique_ptr<Scheduler> makeDefaultScheduler() {
    return std::unique_ptr<Scheduler>(new GenericScheduler());
}

} // namespace ladder
