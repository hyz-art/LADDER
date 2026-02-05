#include <iostream>
#include "scheduler.h"

int main(){
    auto sched = ladder::makeDefaultScheduler();
    std::cout << "Scheduler: " << sched->name() << std::endl;
    sched->schedule();
    return 0;
}
