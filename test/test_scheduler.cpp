#include <iostream>
#include "scheduler.h"

int main(){
    ladder::DeviceInfo dev;
    auto t = ladder::Scheduler::recommendTiles(128,128,64, dev);
    std::cout << "recommended tiles: ("<< std::get<0>(t) <<","<< std::get<1>(t) <<","<< std::get<2>(t) <<")\n";
    return 0;
}
