#include <iostream>
#include <vector>
#include <cstddef>

namespace ladder {

void printShape(const std::vector<size_t>& s) {
    std::cout << "shape: (";
    for (size_t i = 0; i < s.size(); ++i) {
        std::cout << s[i] << (i+1==s.size()?"":" , ");
    }
    std::cout << ")\n";
}

} // namespace ladder
