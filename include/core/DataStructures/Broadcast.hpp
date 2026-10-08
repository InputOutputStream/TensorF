#pragma once
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/Types/shape.hpp"

template <typename U>
class Matrix;


// Namespace for matrice broadcast helpers 
namespace mxd {

inline size_t numel(const shape_t& s) { 
    size_t n = 1;
    for (size_t d : s) n *= d;
    return n;
}

// Row-major element strides ("numElementsSeen" in the old code).
inline shape_t strides_of(const shape_t& s) {
    shape_t st(s.size());
    size_t p = 1;
    for (size_t i = s.size(); i-- > 0;) { st[i] = p; p *= s[i]; }
    return st;
}

inline std::string shape_str(const shape_t& s) {
    std::string r = "(";
    for (size_t i = 0; i < s.size(); i++) { if (i) r += ","; r += std::to_string(s[i]); }
    return r + ")";
}

// Right-aligned broadcast of two shapes. Returns false if incompatible.
// (0 broadcast with 1 gives 0.)
inline bool broadcast_shapes(const shape_t& a, const shape_t& b, shape_t& out) {
    const size_t n = std::max(a.size(), b.size());
    out.assign(n, 1);
    for (size_t i = 0; i < n; i++) {
        const size_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
        const size_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
        size_t r;
        if (da == db)      r = da;
        else if (da == 1)  r = db;
        else if (db == 1)  r = da;
        else               return false;
        out[n - 1 - i] = r;
    }
    return true;
}

// Materialise `src` into `out_shape` walking with the given per-dimension element strides
// (stride 0 repeats an element). Odometer increment: no per-element allocation, no div/mod.
template <typename T>
std::vector<T> gather(const std::vector<T>& src, const shape_t& out_shape,
                      const std::vector<size_t>& es) {
    const size_t total = numel(out_shape);
    std::vector<T> out;
    out.reserve(total);
    if (total == 0) return out;

    const size_t r = out_shape.size();
    std::vector<size_t> idx(r, 0);
    size_t pos = 0;
    for (size_t n = 0; n < total; n++) {
        out.push_back(src[pos]);
        for (size_t d = r; d-- > 0;) {
            idx[d]++;
            pos += es[d];
            if (idx[d] < out_shape[d]) break;
            pos -= es[d] * out_shape[d];
            idx[d] = 0;
        }
    }
    return out;
}

} // namespace mxd

template <typename T>
class Broadcast {
public:
    // All members are static (C7): callable as Broadcast<T>::f(...) or on an instance.

    static shape_t computeShapes(const shape_t& shape) { return mxd::strides_of(shape); }

    static bool assertBroadcast(const Matrix<T>& t1, const Matrix<T>& t2) {
        shape_t tmp;
        return mxd::broadcast_shapes(t1.shape, t2.shape, tmp);
    }

    static shape_t computeBroadcastResultShape(const Matrix<T>& t1, const Matrix<T>& t2) {
        shape_t res;
        if (!mxd::broadcast_shapes(t1.shape, t2.shape, res))
            throw std::runtime_error("Invalid broadcast operation: shapes " + mxd::shape_str(t1.shape) +
                                     " and " + mxd::shape_str(t2.shape) + " are not compatible");
        return res;
    }

    static std::pair<Matrix<T>, Matrix<T>> broadcast(const Matrix<T>& t1, const Matrix<T>& t2) {
        shape_t resShape = computeBroadcastResultShape(t1, t2);
        return std::make_pair(broadcastTo(t1, resShape), broadcastTo(t2, resShape));
    }

    // Broadcast `source` to `new_shape` (numpy rules: right-aligned, size-1 dims repeat).
    static Matrix<T> broadcastTo(const Matrix<T>& source, const shape_t& new_shape) {
        const size_t sr = source.shape.size();
        const size_t r  = new_shape.size();
        if (sr > r)
            throw std::runtime_error("Broadcast::broadcastTo: cannot broadcast shape " + mxd::shape_str(source.shape) +
                                     " to lower-rank shape " + mxd::shape_str(new_shape));
        const size_t offset = r - sr;
        for (size_t d = 0; d < sr; d++)
            if (source.shape[d] != new_shape[offset + d] && source.shape[d] != 1)
                throw std::runtime_error("Broadcast::broadcastTo: cannot broadcast shape " + mxd::shape_str(source.shape) +
                                         " to " + mxd::shape_str(new_shape));

        if (source.shape == new_shape) return source;

        if (source.data.size() < mxd::numel(source.shape))
            throw std::runtime_error("Broadcast::broadcastTo: source has shape " + mxd::shape_str(source.shape) +
                                     " (" + std::to_string(mxd::numel(source.shape)) + " elements) but its data vector holds only " +
                                     std::to_string(source.data.size()));

        const shape_t ss = mxd::strides_of(source.shape);
        std::vector<size_t> es(r, 0);
        for (size_t d = 0; d < sr; d++)
            es[offset + d] = (source.shape[d] == 1) ? 0 : ss[d];

        return Matrix<T>(mxd::gather(source.data, new_shape, es), new_shape);
    }
};
