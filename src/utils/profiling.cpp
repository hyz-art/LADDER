#include "profiling.h"
#include <chrono>

namespace ladder {

void Profiler::start() { start_ = std::chrono::high_resolution_clock::now(); }

void Profiler::stop() { end_ = std::chrono::high_resolution_clock::now(); }

double Profiler::ms() const { return std::chrono::duration<double, std::milli>(end_ - start_).count(); }

} // namespace ladder
