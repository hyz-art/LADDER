#pragma once
#include <vector>
#include <cstddef>
#include <string>
#include <memory>

#ifdef LADDER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace ladder {

class tTile {
public:
    enum class LayoutKind {
        RowMajor,
        Blocked,
        Packed
    };

#ifdef LADDER_ENABLE_CUDA
    using CudaStream = cudaStream_t;
    using CudaEvent = cudaEvent_t;
#else
    using CudaStream = void*;
    using CudaEvent = void*;
#endif

    tTile() = default;
    tTile(const std::vector<size_t>& shape);
    tTile(const tTile& other);
    tTile& operator=(const tTile& other);
    tTile(tTile&& other) noexcept = default;
    tTile& operator=(tTile&& other) noexcept = default;
    ~tTile();

    const std::vector<size_t>& shape() const;
    size_t size() const;

    // plain float storage for prototype
    std::vector<float>& data();
    const std::vector<float>& data() const;

    // basic operations: slice (returns new tile), pad (inplace), map (apply func)
    tTile slice(size_t dim, size_t start, size_t len) const;
    void pad(size_t dim, size_t pad_before, size_t pad_after, float value = 0.0f);
    template<typename Func>
    void map(Func f) {
        for (auto &v : data_) v = f(v);
    }

    // layout metadata
    void setLayout(LayoutKind kind, const std::vector<size_t>& params = {});
    LayoutKind layoutKind() const;
    const std::vector<size_t>& layoutParams() const;

    // explicit copy and prefetch/async copy (prototype interfaces)
    tTile copyToLocal(CudaStream stream = nullptr, CudaEvent event = nullptr) const;
    tTile copyToGlobal(CudaStream stream = nullptr, CudaEvent event = nullptr) const;
    void prefetch(int level = 0, CudaStream stream = nullptr, CudaEvent event = nullptr) const;
    tTile asyncCopyToLocal(CudaStream stream = nullptr, CudaEvent event = nullptr) const;
    tTile asyncCopyToGlobal(CudaStream stream = nullptr, CudaEvent event = nullptr) const;

    // device accessors and helpers
    bool hasDevice() const;
    void *devicePtr() const;
    size_t deviceBytes() const;
    void ensureDevice(CudaStream stream = nullptr, CudaEvent event = nullptr);
    void ensureHost(CudaStream stream = nullptr, CudaEvent event = nullptr);

private:
    struct DeviceBuffer {
#ifdef LADDER_ENABLE_CUDA
        void *ptr = nullptr;
        size_t bytes = 0;
        DeviceBuffer() = default;
        explicit DeviceBuffer(size_t b) : bytes(b) {
            cudaMalloc(&ptr, bytes);
        }
        ~DeviceBuffer() {
            if (ptr) cudaFree(ptr);
        }
        DeviceBuffer(const DeviceBuffer&) = delete;
        DeviceBuffer& operator=(const DeviceBuffer&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept {
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
            if (this != &other) {
                if (ptr) cudaFree(ptr);
                ptr = other.ptr;
                bytes = other.bytes;
                other.ptr = nullptr;
                other.bytes = 0;
            }
            return *this;
        }
#else
        size_t bytes = 0;
        DeviceBuffer() = default;
        explicit DeviceBuffer(size_t b) : bytes(b) {}
#endif
    };
    std::vector<size_t> shape_;
    std::vector<float> data_;
    LayoutKind layout_ = LayoutKind::RowMajor;
    std::vector<size_t> layout_params_;
    std::unique_ptr<DeviceBuffer> device_;
};

} // namespace ladder
