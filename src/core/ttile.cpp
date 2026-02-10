#include "ttile.h"
#include <numeric>
#include <algorithm>
#include <functional>

#ifdef LADDER_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

namespace ladder {

#ifdef LADDER_ENABLE_CUDA
static void cudaCheck(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        (void)msg;
        // In prototype, avoid throwing; consume error for robustness.
        cudaGetLastError();
    }
}

static cudaStream_t getOrCreateStream(cudaStream_t stream, bool &owned) {
    if (stream) {
        owned = false;
        return stream;
    }
    cudaStream_t created = nullptr;
    cudaCheck(cudaStreamCreateWithFlags(&created, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    owned = true;
    return created;
}

static void recordEventIfNeeded(cudaEvent_t event, cudaStream_t stream) {
    if (event && stream) {
        cudaCheck(cudaEventRecord(event, stream), "cudaEventRecord");
    }
}
#endif


static size_t product(const std::vector<size_t>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), (size_t)1, std::multiplies<size_t>());
}

tTile::tTile(const std::vector<size_t>& shape) : shape_(shape) {
    size_t s = product(shape_);
    data_.assign(s, 0.0f);
}

tTile::tTile(const tTile& other)
    : shape_(other.shape_),
      data_(other.data_),
      layout_(other.layout_),
      layout_params_(other.layout_params_),
      device_(nullptr) {}

tTile& tTile::operator=(const tTile& other) {
    if (this != &other) {
        shape_ = other.shape_;
        data_ = other.data_;
        layout_ = other.layout_;
        layout_params_ = other.layout_params_;
        device_.reset();
    }
    return *this;
}

tTile::~tTile() = default;

const std::vector<size_t>& tTile::shape() const { return shape_; }

size_t tTile::size() const { return data_.size(); }

std::vector<float>& tTile::data() { return data_; }
const std::vector<float>& tTile::data() const { return data_; }

tTile tTile::slice(size_t dim, size_t start, size_t len) const {
    // naive: only supports slicing first dimension for prototype
    if (dim >= shape_.size()) return tTile();
    std::vector<size_t> new_shape = shape_;
    new_shape[dim] = std::min(len, shape_[dim] - start);
    tTile out(new_shape);
    // copy linear prefix for prototype
    size_t copy_elems = product(new_shape);
    std::copy_n(data_.begin() + start * (copy_elems / new_shape[dim]), copy_elems, out.data().begin());
    return out;
}

void tTile::pad(size_t dim, size_t pad_before, size_t pad_after, float value) {
    if (dim >= shape_.size()) return;
    // primitive implementation: only supports pad on last dim by resizing data
    shape_[dim] += pad_before + pad_after;
    size_t new_size = product(shape_);
    data_.resize(new_size, value);
}


void tTile::setLayout(LayoutKind kind, const std::vector<size_t>& params) {
    layout_ = kind;
    layout_params_ = params;
}

tTile::LayoutKind tTile::layoutKind() const { return layout_; }

const std::vector<size_t>& tTile::layoutParams() const { return layout_params_; }

bool tTile::hasDevice() const {
#ifdef LADDER_ENABLE_CUDA
    return device_ && device_->ptr;
#else
    return false;
#endif
}

void *tTile::devicePtr() const {
#ifdef LADDER_ENABLE_CUDA
    return device_ ? device_->ptr : nullptr;
#else
    return nullptr;
#endif
}

size_t tTile::deviceBytes() const {
#ifdef LADDER_ENABLE_CUDA
    return device_ ? device_->bytes : 0;
#else
    return 0;
#endif
}

void tTile::ensureDevice(CudaStream stream, CudaEvent event) {
#ifdef LADDER_ENABLE_CUDA
    if (device_ && device_->ptr)
        return;
    size_t bytes = data_.size() * sizeof(float);
    if (bytes == 0) return;
    device_ = std::make_unique<DeviceBuffer>(bytes);
    if (device_ && device_->ptr) {
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemcpyAsync(device_->ptr, data_.data(), bytes, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync H2D");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#else
    (void)stream;
    (void)event;
#endif
}

void tTile::ensureHost(CudaStream stream, CudaEvent event) {
#ifdef LADDER_ENABLE_CUDA
    if (!(device_ && device_->ptr))
        return;
    size_t bytes = data_.size() * sizeof(float);
    bool owned = false;
    auto s = getOrCreateStream(stream, owned);
    cudaCheck(cudaMemcpyAsync(data_.data(), device_->ptr, bytes, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync D2H");
    recordEventIfNeeded(event, s);
    if (owned) {
        cudaCheck(cudaStreamSynchronize(s), "cudaStreamSynchronize");
        cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
    }
#else
    (void)stream;
    (void)event;
#endif
}

tTile tTile::copyToLocal(CudaStream stream, CudaEvent event) const {
    tTile out(*this);
#ifdef LADDER_ENABLE_CUDA
    size_t bytes = out.data_.size() * sizeof(float);
    if (bytes == 0) return out;
    out.device_ = std::make_unique<DeviceBuffer>(bytes);
    if (out.device_ && out.device_->ptr) {
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemcpyAsync(out.device_->ptr, out.data_.data(), bytes, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync H2D");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#endif
    return out;
}

tTile tTile::copyToGlobal(CudaStream stream, CudaEvent event) const {
    tTile out(*this);
#ifdef LADDER_ENABLE_CUDA
    if (device_ && device_->ptr) {
        size_t bytes = out.data_.size() * sizeof(float);
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemcpyAsync(out.data_.data(), device_->ptr, bytes, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync D2H");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#endif
    return out;
}

void tTile::prefetch(int level, CudaStream stream, CudaEvent event) const {
    (void)level;
#ifdef LADDER_ENABLE_CUDA
    if (device_ && device_->ptr) {
        int dev = 0;
        cudaCheck(cudaGetDevice(&dev), "cudaGetDevice");
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemPrefetchAsync(device_->ptr, device_->bytes, dev, s), "cudaMemPrefetchAsync");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamSynchronize(s), "cudaStreamSynchronize");
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#endif
}

tTile tTile::asyncCopyToLocal(CudaStream stream, CudaEvent event) const {
    tTile out(*this);
#ifdef LADDER_ENABLE_CUDA
    size_t bytes = out.data_.size() * sizeof(float);
    if (bytes == 0) return out;
    out.device_ = std::make_unique<DeviceBuffer>(bytes);
    if (out.device_ && out.device_->ptr) {
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemcpyAsync(out.device_->ptr, out.data_.data(), bytes, cudaMemcpyHostToDevice, s), "cudaMemcpyAsync H2D");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#endif
    return out;
}

tTile tTile::asyncCopyToGlobal(CudaStream stream, CudaEvent event) const {
    tTile out(*this);
#ifdef LADDER_ENABLE_CUDA
    if (device_ && device_->ptr) {
        size_t bytes = out.data_.size() * sizeof(float);
        bool owned = false;
        auto s = getOrCreateStream(stream, owned);
        cudaCheck(cudaMemcpyAsync(out.data_.data(), device_->ptr, bytes, cudaMemcpyDeviceToHost, s), "cudaMemcpyAsync D2H");
        recordEventIfNeeded(event, s);
        if (owned) {
            cudaCheck(cudaStreamDestroy(s), "cudaStreamDestroy");
        }
    }
#endif
    return out;
}

} // namespace ladder
