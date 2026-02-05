#include <iostream>
#include "ttype.h"

int main(){
    ladder::tType t(ladder::Precision::FP8);
    std::cout << "tType precision: " << t.toString() << std::endl;
    float v = ladder::tType::convertToFloat(8, ladder::Precision::FP8);
    std::cout << "convert example: " << v << std::endl;
    return 0;
}
