#include <iostream>

namespace ladder {

class Parser {
public:
    void parse(const std::string &desc) { std::cout << "Parsing graph: " << desc << "\n"; }
};

} // namespace ladder
