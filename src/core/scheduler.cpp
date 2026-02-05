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
