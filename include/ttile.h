#pragma once
#include <vector>
#include <cstddef>
#include <string>
#include <memory>

namespace ladder {

class tTile {
public:
    enum class LayoutKind {
        RowMajor,
        Blocked,
        Packed
    };

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
    void map(Func f);

    // layout metadata
    void setLayout(LayoutKind kind, const std::vector<size_t>& params = {});
    LayoutKind layoutKind() const;
    const std::vector<size_t>& layoutParams() const;

    // explicit copy and prefetch/async copy (prototype interfaces)
    tTile copyToLocal() const;
    tTile copyToGlobal() const;
    void prefetch(int level = 0) const;
    tTile asyncCopyToLocal() const;
    tTile asyncCopyToGlobal() const;

private:
    struct DeviceBuffer;
    std::vector<size_t> shape_;
    std::vector<float> data_;
    LayoutKind layout_ = LayoutKind::RowMajor;
    std::vector<size_t> layout_params_;
    std::unique_ptr<DeviceBuffer> device_;
};

} // namespace ladder
