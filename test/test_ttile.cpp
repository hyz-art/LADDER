#include <iostream>
#include "ttile.h"

int main(){
    ladder::tTile tile({4,4});
    std::cout << "tile size: " << tile.size() << std::endl;
    tile.pad(1,1,1,0.5f);
    std::cout << "tile new size after pad: " << tile.size() << std::endl;
    return 0;
}
