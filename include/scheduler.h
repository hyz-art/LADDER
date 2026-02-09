#pragma once
#include <cstddef>
#include <tuple>

namespace ladder {

struct DeviceInfo {
    size_t l1_size = 32 * 1024; // bytes
    size_t l2_size = 256 * 1024;
    size_t sm_count = 1;
    size_t warp_size = 32;
};

class Scheduler {
public:
    // Recommend (tileM, tileN, tileK) based on matrix dims and device info
    static std::tuple<size_t,size_t,size_t> recommendTiles(size_t M, size_t N, size_t K, const DeviceInfo& dev);
};

} // namespace ladder
