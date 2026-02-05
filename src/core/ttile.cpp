#include "ttile.h"
#include <numeric>
#include <algorithm>
#include <functional>

namespace ladder {

static size_t product(const std::vector<size_t>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), (size_t)1, std::multiplies<size_t>());
}

tTile::tTile(const std::vector<size_t>& shape) : shape_(shape) {
    size_t s = product(shape_);
    data_.assign(s, 0.0f);
}

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

template<typename Func>
void tTile::map(Func f) {
    for (auto &v : data_) v = f(v);
}

// explicit instantiation for common lambdas
template void tTile::map(std::function<float(float)> f);

} // namespace ladder
