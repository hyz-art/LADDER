#include <iostream>

namespace ladder {

class Device {
public:
    virtual ~Device() = default;
    virtual void info() = 0;
};

class CpuDevice : public Device {
public:
    void info() override { std::cout << "CPU device (host)" << std::endl; }
};

} // namespace ladder
