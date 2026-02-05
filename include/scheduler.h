#pragma once
#include <string>

#include <memory>

namespace ladder {

class Scheduler {
public:
    virtual ~Scheduler() = default;
    virtual std::string name() const = 0;
    // schedule hook: in real system would accept tasks/graphs
    virtual void schedule() = 0;
};

// factory to create a default scheduler
std::unique_ptr<Scheduler> makeDefaultScheduler();

} // namespace ladder
