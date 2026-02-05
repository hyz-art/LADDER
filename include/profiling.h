#pragma once
#include <chrono>

namespace ladder {

class Profiler {
public:
    void start();
    void stop();
    double ms() const;
private:
    std::chrono::high_resolution_clock::time_point start_, end_;
};

} // namespace ladder
