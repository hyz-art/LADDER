#pragma once
#include <vector>
#include <cstddef>

namespace ladder {

class tTile {
public:
    tTile() = default;
    tTile(const std::vector<size_t>& shape);

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

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;
};

} // namespace ladder
