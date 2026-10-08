#pragma once
// Matrix.hpp — dense row-major n-D tensor. Include THIS file only: it pulls in Broadcast.hpp
// first and MatrixOps.hpp (free operators) last, so existing includes keep working.
//
// Build switches:
//   MATRIX_NO_BLAS        do not use cblas (portable loops are used instead; default is BLAS ON)
//   MATRIX_IEEE_DIVISION  vector division stops throwing on a zero divisor for floating types (see VectorMath.hpp)
//   MATRIX_NO_VECTOR_OPERATORS  do not include Overloads/Overload.hpp (the global operators on std::vector); Matrix does not need it
//
// Invariants: data.size() == prod(shape) always (no tail padding is ever stored).
// 0-D results (sum of a 1-D matrix, ...) have shape {} and exactly one element.
// Not thread-safe: the random generator is one shared global (see mxd::rng()).

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/Types/shape.hpp"
#include "core/Types/scalar.hpp"
#include "core/Overloads/VectorMath.hpp"      
#ifndef MATRIX_NO_VECTOR_OPERATORS
    #include "core/Overloads/Overload.hpp"
#endif
#include "Broadcast.hpp"

#if !defined(MATRIX_USE_BLAS) && !defined(MATRIX_NO_BLAS)
    #define MATRIX_USE_BLAS 0
#endif
#ifdef MATRIX_USE_BLAS
    #include <cblas.h>
#endif

namespace mxd {

template <class> inline constexpr bool dependent_false_v = false;

// float-like = float/double/long double or FP8/FP4/float16.
template <class T> inline constexpr bool is_float_like_v = std::is_floating_point_v<T> || scalar::is_low_precision_v<T>;

// Type used inside accumulation buffers. bool uses int (vector<bool> has no usable proxy for +=).
template <class T> using store_t = std::conditional_t<std::is_same_v<T, bool>, int, scalar::acc_t<T>>;

// Type random distributions are evaluated in (float/double stay as they are; FP8/FP4/float16 use float; ints use double).
template <class T> using calc_t = std::conditional_t<std::is_floating_point_v<T>, T,
                                  std::conditional_t<scalar::is_low_precision_v<T>, float, double>>;

inline std::mt19937& rng(std::optional<unsigned int> seed = std::nullopt) {
    static std::mt19937 gen(std::random_device{}());
    if (seed.has_value()) gen.seed(*seed);
    return gen;
}
class CallRng {
    std::mt19937  local_;
    std::mt19937* g_;
public:
    explicit CallRng(std::optional<unsigned int> seed)
        : local_(seed ? *seed : 0u), g_(seed ? &local_ : &rng()) {}
    CallRng(const CallRng&) = delete;                 // g_ may point at local_
    std::mt19937& operator()() { return *g_; }
};

template <class T> inline T draw_uniform(double lo, double hi, std::mt19937& g) {
    using C = calc_t<T>;
    std::uniform_real_distribution<C> d(static_cast<C>(lo), static_cast<C>(hi));
    return static_cast<T>(d(g));
}

template <class T> inline T draw_normal(double mean, double sd, std::mt19937& g) {
    using C = calc_t<T>;
    std::normal_distribution<C> d(static_cast<C>(mean), static_cast<C>(sd));
    return static_cast<T>(d(g));
}

// Printable value of an element (uint8/int8 as numbers, float16 as float).
template <class T> inline auto printable(const T& x) {
    if constexpr (scalar::is_low_precision_v<T>) return static_cast<float>(x);
    else if constexpr (sizeof(T) == 1 && std::is_integral_v<T> && !std::is_same_v<T, bool>) return static_cast<int>(x);
    else return x;
}

#ifdef MATRIX_USE_BLAS
// D6: checked size_t -> int narrowing for cblas.
inline int blas_int(size_t v) {
    if (v > static_cast<size_t>(INT_MAX))
        throw std::overflow_error("Matrix: dimension too large for cblas (int)");
    return static_cast<int>(v);
}
#endif

// C[M,N] = A[M,K] * B[K,N], row-major, contiguous. Accumulates in store_t<T>.
template <class T>
void gemm_loops(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {
    using S = store_t<T>;
    for (size_t i = 0; i < M; i++)
        for (size_t n = 0; n < N; n++) {
            S s{};
            for (size_t k = 0; k < K; k++)
                s += static_cast<S>(scalar::to_acc<T>(A[i * K + k])) * static_cast<S>(scalar::to_acc<T>(B[k * N + n]));
            C[i * N + n] = scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(s));
        }
}

template <class T>
void gemm(const T* A, const T* B, T* C, size_t M, size_t N, size_t K) {
    if (M == 0 || N == 0) return;
    if (K == 0) { std::fill(C, C + M * N, T(0)); return; }

    if constexpr (scalar::is_low_precision_v<T>) {
        // F3: dequantise -> float gemm -> quantise.
        std::vector<float> fa(M * K), fb(K * N), fc(M * N);
        for (size_t i = 0; i < fa.size(); i++) fa[i] = static_cast<float>(A[i]);
        for (size_t i = 0; i < fb.size(); i++) fb[i] = static_cast<float>(B[i]);
        gemm<float>(fa.data(), fb.data(), fc.data(), M, N, K);
        for (size_t i = 0; i < fc.size(); i++) C[i] = T(fc[i]);
    }
#ifdef MATRIX_USE_BLAS
    else if constexpr (std::is_same_v<T, float>)
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blas_int(M), blas_int(N), blas_int(K),
                    1.0f, A, blas_int(K), B, blas_int(N), 0.0f, C, blas_int(N));
    else if constexpr (std::is_same_v<T, double>)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blas_int(M), blas_int(N), blas_int(K),
                    1.0, A, blas_int(K), B, blas_int(N), 0.0, C, blas_int(N));
#endif
    else
        gemm_loops<T>(A, B, C, M, N, K);
}

// sum_i a[i]*b[i]
template <class T>
T dot_flat(const T* a, const T* b, size_t n) {
#ifdef MATRIX_USE_BLAS
    if constexpr (std::is_same_v<T, float>)  return cblas_sdot(blas_int(n), a, 1, b, 1);
    else if constexpr (std::is_same_v<T, double>) return cblas_ddot(blas_int(n), a, 1, b, 1);
    else
#endif
    {
        using S = store_t<T>;
        S s{};
        for (size_t i = 0; i < n; i++)
            s += static_cast<S>(scalar::to_acc<T>(a[i])) * static_cast<S>(scalar::to_acc<T>(b[i]));
        return scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(s));
    }
}

} // namespace mxd


template <typename T>
class Matrix
{
    using S = mxd::store_t<T>;

    // ───────────────────────────── private helpers ─────────────────────────────
    bool gpu_nv = false;   // (unused, kept)
    bool gpu_it = false;   // (unused, kept)

    static size_t numel(const shape_t& s) { return mxd::numel(s); }
    static shape_t computeShapes(const shape_t& s) { return mxd::strides_of(s); }
    static bool verifyShape(const std::vector<T>& d, const shape_t& s) { return d.size() == mxd::numel(s); }

    static bool isRegular2D(const std::vector<std::vector<T>>& d) {
        for (size_t i = 1; i < d.size(); i++)
            if (d[i].size() != d[0].size()) return false;
        return true;
    }
    static bool isRegular2D(const std::initializer_list<std::initializer_list<T>>& d) {
        if (d.size() == 0) return true;
        const size_t cols = d.begin()->size();
        for (const auto& row : d) if (row.size() != cols) return false;
        return true;
    }
    static bool isRegular3D(const std::initializer_list<std::initializer_list<std::initializer_list<T>>>& d) {
        if (d.size() == 0) return true;
        const size_t d1 = d.begin()->size();
        const size_t d2 = d1 ? d.begin()->begin()->size() : 0;
        for (const auto& row : d) {
            if (row.size() != d1) return false;
            for (const auto& sub : row) if (sub.size() != d2) return false;
        }
        return true;
    }

    static bool areShapes1D(const shape_t& l, const shape_t& r) { return l.size() == 1 && r.size() == 1; }
    static bool areShapes2D(const shape_t& l, const shape_t& r) { return l.size() == 2 && r.size() == 2; }

    bool dotShapesAssert(const shape_t& rs) const {
        if (shape.size() < 2 || rs.size() < 2) return false;
        return rs[rs.size() - 2] == shape.back();
    }

    void axisDims(size_t axis, size_t& outer, size_t& n, size_t& inner, const char* who) const {
        if (axis >= shape.size())
            throw std::out_of_range(std::string(who) + ": axis " + std::to_string(axis) + " out of range for rank " + std::to_string(shape.size()));
        outer = 1; inner = 1;
        for (size_t d = 0; d < axis; d++) outer *= shape[d];
        n = shape[axis];
        for (size_t d = axis + 1; d < shape.size(); d++) inner *= shape[d];
    }

    // out[o*inner+i] = sum_k data[(o*n+k)*inner+i], accumulated in S.
    std::vector<S> reduceSum(size_t axis, shape_t& rs, size_t& n, const char* who) const {
        size_t outer, inner;
        axisDims(axis, outer, n, inner, who);
        rs.clear();
        for (size_t i = 0; i < shape.size(); i++) if (i != axis) rs.push_back(shape[i]);
        std::vector<S> acc(outer * inner, S{});
        if (inner == 1) {                       // reducing the last axis: contiguous rows, scalar accumulator
            for (size_t o = 0; o < outer; o++) {
                S s{};
                const size_t base = o * n;
                for (size_t k = 0; k < n; k++) s += static_cast<S>(scalar::to_acc<T>(data[base + k]));
                acc[o] = s;
            }
            return acc;
        }
        for (size_t o = 0; o < outer; o++)
            for (size_t k = 0; k < n; k++) {
                const size_t base = (o * n + k) * inner;
                S* dst = acc.data() + o * inner;
                for (size_t i = 0; i < inner; i++) dst[i] += static_cast<S>(scalar::to_acc<T>(data[base + i]));
            }
        return acc;
    }

    // Element-wise binary op with numpy broadcasting. `op` works on two equally sized vectors.
    template <class VecOp>
    Matrix<T> binop(const Matrix<T>& rhs, VecOp op) const {
        if (shape == rhs.shape) return Matrix<T>(op(data, rhs.data), shape);
        const shape_t rs = Broadcast<T>::computeBroadcastResultShape(*this, rhs);
        const Matrix<T>* a = this;
        const Matrix<T>* b = &rhs;
        Matrix<T> ta, tb;
        if (shape != rs)     { ta = Broadcast<T>::broadcastTo(*this, rs); a = &ta; }
        if (rhs.shape != rs) { tb = Broadcast<T>::broadcastTo(rhs, rs);   b = &tb; }
        return Matrix<T>(op(a->data, b->data), rs);
    }

    template <class F>
    Matrix<T> mapElems(F f) const {
        std::vector<T> res;
        res.reserve(data.size());
        for (size_t i = 0; i < data.size(); i++) res.push_back(f(static_cast<T>(data[i])));
        return Matrix<T>(std::move(res), shape);
    }

    std::vector<T> transpose_2D() const {
        const size_t rows = shape[shape.size() - 2];
        const size_t cols = shape[shape.size() - 1];
        std::vector<T> res(data.size());
        constexpr size_t B = 32;       // D4: blocked transpose
        for (size_t ib = 0; ib < rows; ib += B)
            for (size_t jb = 0; jb < cols; jb += B) {
                const size_t ie = std::min(rows, ib + B), je = std::min(cols, jb + B);
                for (size_t i = ib; i < ie; i++)
                    for (size_t j = jb; j < je; j++)
                        res[j * rows + i] = data[i * cols + j];
            }
        return res;
    }

    Matrix<T> transpose_1D() const {
        if (shape.size() == 2 && shape[0] == 1) return Matrix<T>(data, shape_t{shape[1], 1});
        if (shape.size() == 2 && shape[1] == 1) return Matrix<T>(data, shape_t{1, shape[0]});
        if (shape.size() == 1)                  return Matrix<T>(data, shape_t{shape[0], 1});
        throw std::runtime_error("transpose_1D: invalid shape for 1D transpose\n");
    }

    Matrix<T> permute(const shape_t& perm) const {
        const size_t nd = shape.size();
        shape_t resShape(nd), es(nd);
        const shape_t ns = computeShapes(shape);
        for (size_t i = 0; i < nd; i++) { resShape[i] = shape[perm[i]]; es[i] = ns[perm[i]]; }
        return Matrix<T>(mxd::gather(data, resShape, es), resShape);
    }

    static T dotProduct1D(const std::vector<T>& l, const std::vector<T>& r) {
        if (l.size() != r.size())
            throw std::invalid_argument("dot: vectors must have the same size (" + std::to_string(l.size()) + " vs " + std::to_string(r.size()) + ")\n");
        if constexpr (std::is_same_v<T, bool>) {
            S s{};
            for (size_t i = 0; i < l.size(); i++) s += static_cast<S>(l[i]) * static_cast<S>(r[i]);
            return s != 0;
        } else {
            return mxd::dot_flat<T>(l.data(), r.data(), l.size());
        }
    }

    // B14 [DECISION]: for two 2-D inputs dot() is the FLATTENED inner product (np.sum(a*b)), not a matrix product.
    Matrix<T> dotProduct2D(const Matrix<T>& rhs) const {
        if (shape != rhs.shape)
            throw std::invalid_argument("dot: two 2-D operands must have identical shapes (flattened inner product), got " +
                                        mxd::shape_str(shape) + " and " + mxd::shape_str(rhs.shape) + "\n");
        return Matrix<T>(std::vector<T>{dotProduct1D(data, rhs.data)}, shape_t{1});
    }

    static std::mt19937& get_gen(std::optional<unsigned int> seed = std::nullopt) { return mxd::rng(seed); }

public:
    std::vector<T> data;
    shape_t shape;
    bool gpu = false;

    friend class Broadcast<T>;

    // ───────────────────────────── constructors ─────────────────────────────

    static shape_t getShape(const std::initializer_list<size_t> shape)
    {
        if (shape.size() == 0) return shape_t{0};
        return shape_t(shape.begin(), shape.end());
    }

    Matrix() = default;

    Matrix(const T& indata) : data(1, indata), shape{1} {}

    // Pointer constructor kept for API compatibility. It is a template so that the literal `0` can never
    // pick it (that made Matrix<float>(0) ambiguous).
    template <class P, std::enable_if_t<std::is_same_v<std::remove_const_t<P>, Matrix<T>>, int> = 0>
    explicit Matrix(const P* two) {
        if (two == nullptr) throw std::runtime_error("Matrix(ptr): null pointer input\n");
        data = two->data;
        shape = two->shape;
    }

    Matrix(const Matrix<T>&) = default;
    Matrix(Matrix<T>&&) noexcept = default;
    Matrix<T>& operator=(const Matrix<T>&) = default;
    Matrix<T>& operator=(Matrix<T>&&) noexcept = default;

    Matrix(std::vector<T> indata) : data(std::move(indata)) { shape.push_back(data.size()); }

    Matrix(std::vector<T> indata, shape_t inshape)
    {
        if (!verifyShape(indata, inshape))
            throw std::runtime_error("Matrix: shape and number of elements do not match");
        data = std::move(indata);
        shape = std::move(inshape);
    }

    Matrix(std::vector<std::vector<T>> indata)
    {
        if (!isRegular2D(indata)) throw std::runtime_error("Matrix: shape must be uniform\n");
        const size_t rows = indata.size();
        const size_t cols = rows ? indata[0].size() : 0;
        shape = {rows, cols};
        data.reserve(rows * cols);
        for (const auto& r : indata) data.insert(data.end(), r.begin(), r.end());
    }

    Matrix(std::vector<std::vector<T>> indata, std::initializer_list<size_t> inshape)
    {
        shape = Matrix<T>::getShape(inshape);
        if (!isRegular2D(indata)) throw std::runtime_error("Matrix: shape must be uniform\n");
        for (const auto& r : indata) data.insert(data.end(), r.begin(), r.end());
        if (!verifyShape(data, shape)) throw std::runtime_error("Matrix: shape and number of elements do not match\n");
    }

    Matrix(std::vector<T> indata, std::initializer_list<size_t> inshape)
    {
        shape = Matrix<T>::getShape(inshape);
        if (!verifyShape(indata, shape)) throw std::runtime_error("Matrix: shape and number of elements do not match\n");
        data = std::move(indata);
    }

    Matrix(std::initializer_list<std::initializer_list<T>> indata)
    {
        if (!isRegular2D(indata)) throw std::runtime_error("Matrix: shape must be uniform\n");
        shape.push_back(indata.size());
        shape.push_back(indata.size() ? indata.begin()->size() : 0);
        flattenRecursive(indata, data);
    }

    Matrix(std::initializer_list<T> indata, std::initializer_list<size_t> inshape)
    {
        shape = Matrix<T>::getShape(inshape);
        flattenRecursive(indata, data);
        if (!verifyShape(data, shape)) throw std::runtime_error("Shape and number of elements of matrix do not match!!!\n");
    }

    Matrix(std::initializer_list<std::initializer_list<std::initializer_list<T>>> indata)
    {
        if (!isRegular3D(indata)) throw std::runtime_error("Matrix shape must be uniform!!!\n");
        shape.push_back(indata.size());
        shape.push_back(indata.size() ? indata.begin()->size() : 0);
        shape.push_back((indata.size() && indata.begin()->size()) ? indata.begin()->begin()->size() : 0);
        flattenRecursive(indata, data);
    }

    template <typename U>
    void flattenRecursive(const U& d, std::vector<T>& out) const
    {
        if constexpr (std::is_same_v<U, T>) out.push_back(d);
        else for (const auto& elem : d) flattenRecursive(elem, out);
    }
    // Build from "anything array-like" (J1: Tensor::from calls this).
    template <class K>
    static Matrix<T> from(const K& in)
    {
        if constexpr (std::is_same_v<K, Matrix<T>>)                        return in;
        else if constexpr (std::is_same_v<K, std::vector<T>>)              return Matrix<T>(in);
        else if constexpr (std::is_same_v<K, std::vector<std::vector<T>>>) return Matrix<T>(in);
        else if constexpr (std::is_arithmetic_v<K> || std::is_same_v<K, T>) return Matrix<T>(static_cast<T>(in));
        else static_assert(mxd::dependent_false_v<K>, "Matrix::from: unsupported source type");
    }
    static Matrix<T> from(std::initializer_list<T> l) { return Matrix<T>(std::vector<T>(l)); }

    // ───────────────────────────── arithmetic ─────────────────────────────

    Matrix<T> operator+(const Matrix<T>& rhs) const { return binop(rhs, [](const std::vector<T>& a, const std::vector<T>& b) { return vecmath::add(a, b); }); }
    Matrix<T> operator-(const Matrix<T>& rhs) const { return binop(rhs, [](const std::vector<T>& a, const std::vector<T>& b) { return vecmath::sub(a, b); }); }
    Matrix<T> operator*(const Matrix<T>& rhs) const { return binop(rhs, [](const std::vector<T>& a, const std::vector<T>& b) { return vecmath::mul(a, b); }); }
    Matrix<T> operator/(const Matrix<T>& rhs) const { return binop(rhs, [](const std::vector<T>& a, const std::vector<T>& b) { return vecmath::div(a, b); }); }

    Matrix<T> operator-() const { return Matrix<T>(vecmath::neg(data), shape); }

    bool operator==(const Matrix<T>& rhs) const { return (shape == rhs.shape) && vecmath::equal(data, rhs.data); }

    template <typename U>
    requires std::is_arithmetic_v<U>
    Matrix<bool> operator==(const U val) const {
        std::vector<bool> res;
        res.reserve(data.size());
        for (const auto& x : data) res.push_back(x == static_cast<T>(val));
        return Matrix<bool>(std::move(res), shape);
    }

    template <typename U>
    requires std::is_arithmetic_v<U>
    Matrix<bool> operator!=(const U val) const {
        std::vector<bool> res;
        res.reserve(data.size());
        for (const auto& x : data) res.push_back(!(x == static_cast<T>(val)));
        return Matrix<bool>(std::move(res), shape);
    }

    // A2: member pow names would hide a global vector pow, so the named vecmath:: function is called.
    Matrix<T> pow(const T rhs) const { return Matrix<T>(vecmath::pow_s(data, rhs), shape); }

    Matrix<T> pow(const Matrix<T>& rhs) const
    {
        return binop(rhs, [](const std::vector<T>& a, const std::vector<T>& b) {
            std::vector<T> r;
            r.reserve(a.size());
            for (size_t i = 0; i < a.size(); i++) r.push_back(scalar::pow<T>(a[i], b[i]));
            return r;
        });
    }

    static Matrix<T> pow(const Matrix<T>& input, T power) { return input.pow(power); }

    Matrix<T> exponent() const { return mapElems([](T x) { return scalar::exp<T>(x); }); }
    Matrix<T> sqrt()     const { return mapElems([](T x) { return scalar::sqrt<T>(x); }); }
    Matrix<T> cbrt()     const { return mapElems([](T x) { return scalar::cbrt<T>(x); }); }

    // A3: clamp at 1e-9 (in the accumulator type) only for float-like T.
    Matrix<T> ln() const
    {
        using A = scalar::acc_t<T>;
        return mapElems([](T x) {
            A v = scalar::to_acc<T>(x);
            if constexpr (mxd::is_float_like_v<T>) v = std::max(v, static_cast<A>(1e-9));
            return scalar::from_acc<T>(static_cast<A>(std::log(v)));
        });
    }

    // ───────────────────────────── reductions ─────────────────────────────

    T sum() const
    {
        S s{};
        for (size_t i = 0; i < data.size(); i++) s += static_cast<S>(scalar::to_acc<T>(data[i]));
        return scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(s));
    }

    // Sum over `axis`; the axis dimension is removed from the shape (a 1-D input gives shape {}).
    Matrix<T> sum(size_t axis) const
    {
        shape_t rs; size_t n;
        std::vector<S> acc = reduceSum(axis, rs, n, "sum");
        std::vector<T> res;
        res.reserve(acc.size());
        for (const S& v : acc) res.push_back(scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(v)));
        return Matrix<T>(std::move(res), std::move(rs));
    }

    Matrix<T> mean() const
    {
        if (data.empty()) throw std::runtime_error("mean: empty matrix\n");
        S s{};
        for (size_t i = 0; i < data.size(); i++) s += static_cast<S>(scalar::to_acc<T>(data[i]));
        s = s / static_cast<S>(data.size());
        return Matrix<T>(scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(s)));
    }

    Matrix<T> mean(size_t axis) const
    {
        shape_t rs; size_t n;
        std::vector<S> acc = reduceSum(axis, rs, n, "mean");
        std::vector<T> res;
        res.reserve(acc.size());
        for (const S& v : acc) res.push_back(scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(v / static_cast<S>(n))));
        return Matrix<T>(std::move(res), std::move(rs));
    }

    // Population variance (ddof = 0) of all elements, shape {1}.
    Matrix<T> variance() const
    {
        if (data.empty()) throw std::runtime_error("variance: empty matrix\n");
        double m = 0;
        for (size_t i = 0; i < data.size(); i++) m += static_cast<double>(scalar::to_acc<T>(data[i]));
        m /= static_cast<double>(data.size());
        double v = 0;
        for (size_t i = 0; i < data.size(); i++) {
            const double d = static_cast<double>(scalar::to_acc<T>(data[i])) - m;
            v += d * d;
        }
        v /= static_cast<double>(data.size());
        return Matrix<T>(scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(v)));
    }

    Matrix<T> std() const { return variance().sqrt(); }

    // ddof0 == true: divide by n (population); false: divide by n-1.
    Matrix<T> var(size_t axis, bool ddof0 = true) const
    {
        size_t outer, n, inner;
        axisDims(axis, outer, n, inner, "var");
        shape_t rs;
        for (size_t i = 0; i < shape.size(); i++) if (i != axis) rs.push_back(shape[i]);

        std::vector<double> mean(outer * inner, 0.0), m2(outer * inner, 0.0);
        for (size_t o = 0; o < outer; o++)
            for (size_t k = 0; k < n; k++)
                for (size_t i = 0; i < inner; i++)
                    mean[o * inner + i] += static_cast<double>(scalar::to_acc<T>(data[(o * n + k) * inner + i]));
        for (double& m : mean) m /= static_cast<double>(n);
        for (size_t o = 0; o < outer; o++)
            for (size_t k = 0; k < n; k++)
                for (size_t i = 0; i < inner; i++) {
                    const double d = static_cast<double>(scalar::to_acc<T>(data[(o * n + k) * inner + i])) - mean[o * inner + i];
                    m2[o * inner + i] += d * d;
                }
        const double denom = static_cast<double>(ddof0 ? n : n - 1);
        std::vector<T> res;
        res.reserve(m2.size());
        for (double v : m2) res.push_back(scalar::from_acc<T>(static_cast<scalar::acc_t<T>>(v / denom)));
        return Matrix<T>(std::move(res), std::move(rs));
    }

    Matrix<T> std(size_t axis, bool ddof0 = true) const { return var(axis, ddof0).sqrt(); }

    // ───────────────────────────── accessors / indexing ─────────────────────────────

    size_t get_size() const  { return data.size(); }
    size_t get_ndims() const { return shape.size(); }
    shape_t strides() const  { return computeShapes(shape); }
    shape_t numElementsSeen() const { return computeShapes(shape); }   // old name

    // 1-D: returns *this (kept). 2-D only otherwise.
    Matrix<T> col(size_t idx) const
    {
        if (shape.size() == 1) return *this;
        if (shape.size() != 2) throw std::runtime_error("col: only 1-D and 2-D matrices are supported");
        if (idx >= shape[1]) throw std::out_of_range("Invalid Col Index");
        const size_t rows = shape[0], cols = shape[1];
        std::vector<T> res;
        res.reserve(rows);
        for (size_t i = 0; i < rows; i++) res.push_back(data[cols * i + idx]);
        return Matrix<T>(std::move(res), shape_t{rows, 1});
    }

    Matrix<T> row(size_t idx) const
    {
        if (shape.size() == 1) return *this;
        if (shape.size() != 2) throw std::runtime_error("row: only 1-D and 2-D matrices are supported");
        if (idx >= shape[0]) throw std::out_of_range("Invalid Row Index");
        const size_t cols = shape[1];
        return Matrix<T>(std::vector<T>(data.begin() + idx * cols, data.begin() + (idx + 1) * cols), shape_t{1, cols});
    }

    // Index with fewer/equal indices than rank. Full index -> shape {1}; partial -> the remaining sub-tensor
    // (when the remaining rank is 1 the shape is {1, n}, matching row()).
    Matrix<T> at(shape_t index) const
    {
        if (index.empty() || index.size() > shape.size()) throw std::runtime_error("Invalid Index");
        const shape_t st = computeShapes(shape);
        size_t off = 0;
        for (size_t d = 0; d < index.size(); d++) {
            if (index[d] >= shape[d]) throw std::out_of_range("at: index out of range");
            off += index[d] * st[d];
        }
        if (index.size() == shape.size())
            return Matrix<T>(std::vector<T>{data[off]}, shape_t{1});
        shape_t rem(shape.begin() + index.size(), shape.end());
        const size_t cnt = numel(rem);
        if (rem.size() == 1) rem = shape_t{1, rem[0]};
        return Matrix<T>(std::vector<T>(data.begin() + off, data.begin() + off + cnt), rem);
    }

    Matrix<T> at(std::initializer_list<size_t> inshape) const { return at(shape_t(inshape.begin(), inshape.end())); }

    // Keeps the entries where `index` is true, writes 0 elsewhere (same shape).
    Matrix<T> at(const Matrix<bool>& index) const
    {
        if (index.shape != shape) throw std::runtime_error("Index Matrix not of the same shape");
        std::vector<T> res;
        res.reserve(data.size());
        for (size_t i = 0; i < data.size(); i++) res.push_back(index.data[i] ? data[i] : T(0));
        return Matrix<T>(std::move(res), shape);
    }

    Matrix<T> slice_row(size_t start, size_t end) const
    {
        if (shape.size() == 1) throw std::runtime_error("not impl for 1D");
        if (shape.size() != 2) throw std::runtime_error("slice_row: only 2-D matrices are supported");
        if (start > end || end > shape[0]) throw std::out_of_range("Invalid Row Slice");
        const size_t cols = shape[1];
        return Matrix<T>(std::vector<T>(data.begin() + start * cols, data.begin() + end * cols), shape_t{end - start, cols});
    }

    Matrix<T> slice_cols(size_t start, size_t end) const
    {
        if (shape.size() == 1) throw std::runtime_error("Not a 2D matrix");
        if (shape.size() != 2) throw std::runtime_error("slice_cols: only 2-D matrices are supported");
        if (start > end || end > shape[1]) throw std::out_of_range("Invalid Col Slice");
        const size_t rows = shape[0], cols = shape[1];
        std::vector<T> res;
        res.reserve(rows * (end - start));
        for (size_t j = 0; j < rows; j++)
            res.insert(res.end(), data.begin() + j * cols + start, data.begin() + j * cols + end);
        return Matrix<T>(std::move(res), shape_t{rows, end - start});
    }

    Matrix<T> slice_axis(size_t start, size_t end, size_t axis) const
    {
        if (axis >= shape.size()) throw std::out_of_range("slice_axis: axis out of range");
        if (start > end || end > shape[axis]) throw std::out_of_range("slice_axis: invalid [start, end)");
        size_t outer, n, inner;
        axisDims(axis, outer, n, inner, "slice_axis");
        shape_t out_shape = shape;
        out_shape[axis] = end - start;
        std::vector<T> out;
        out.reserve(outer * (end - start) * inner);
        for (size_t o = 0; o < outer; o++)
            out.insert(out.end(), data.begin() + (o * n + start) * inner, data.begin() + (o * n + end) * inner);
        return Matrix<T>(std::move(out), std::move(out_shape));
    }

    Matrix<T> flatten() const { return Matrix<T>(data); }

    Matrix<T> reshape(std::initializer_list<size_t> new_shape) const { return reshape(Matrix<T>::getShape(new_shape)); }

    Matrix<T> reshape(shape_t new_shape) const
    {
        if (numel(new_shape) != data.size()) throw std::runtime_error("reshape: size mismatch");
        return Matrix<T>(data, std::move(new_shape));
    }

    std::vector<T> get_data() const { return data; }

    // Embedding lookup: `this` is [vocab, dim]; result has indices.shape + {dim}.
    Matrix<T> elemsAt(const Matrix<T>& indices) const
    {
        if (shape.empty()) throw std::runtime_error("elemsAt: empty matrix");
        const size_t dim = shape.back();
        const size_t vocab_size = shape[0];
        const size_t n_tokens = indices.data.size();
        std::vector<T> out;
        out.reserve(n_tokens * dim);
        for (size_t i = 0; i < n_tokens; i++) {
            const double v = std::round(static_cast<double>(scalar::to_acc<T>(indices.data[i])));
            if (!(v >= 0)) throw std::runtime_error("Negative index in embedding lookup");
            const size_t idx = static_cast<size_t>(v);
            if (idx >= vocab_size)
                throw std::runtime_error("Index out of bounds in embedding lookup: " + std::to_string(idx));
            out.insert(out.end(), data.begin() + idx * dim, data.begin() + idx * dim + dim);
        }
        shape_t out_shape = indices.shape;
        out_shape.push_back(dim);
        return Matrix<T>(std::move(out), std::move(out_shape));
    }

    // ───────────────────────────── static factories ─────────────────────────────

    static T inf() { return std::numeric_limits<T>::infinity(); }
    static T nan() { return std::numeric_limits<T>::quiet_NaN(); }

    static Matrix<T> ravel(const Matrix<T>& mat) { return Matrix<T>(mat.data); }

    static Matrix<T> expand_dims(const Matrix<T>& m, size_t axis)
    {
        if (axis > m.shape.size()) throw std::out_of_range("expand_dims: axis out of range");
        shape_t new_shape = m.shape;
        new_shape.insert(new_shape.begin() + axis, 1);
        return Matrix<T>(m.data, std::move(new_shape));
    }

    static bool any(const Matrix<T>& m)
    {
        for (const auto& v : m.data) if (v != T(0)) return true;
        return false;
    }

    static bool hasNaN(const Matrix<T>& m)
    {
        if constexpr (mxd::is_float_like_v<T>) {
            for (size_t i = 0; i < m.data.size(); i++) if (scalar::isnan<T>(m.data[i])) return true;
        }
        return false;
    }

    template <typename Pred>
    static bool any(const Matrix<T>& m, Pred pred)
    {
        for (const auto& v : m.data) if (pred(v)) return true;
        return false;
    }

    // Concatenation along an existing axis.
    static Matrix<T> concat(const std::vector<Matrix<T>>& mats, size_t axis)
    {
        if (mats.empty()) throw std::invalid_argument("concat: empty input");
        const size_t rank = mats[0].shape.size();
        if (axis >= rank) throw std::out_of_range("concat: axis out of range");
        for (size_t i = 1; i < mats.size(); i++) {
            if (mats[i].shape.size() != rank) throw std::runtime_error("concat: rank mismatch");
            for (size_t d = 0; d < rank; d++)
                if (d != axis && mats[i].shape[d] != mats[0].shape[d])
                    throw std::runtime_error("concat: shape mismatch on non-concat axis");
        }
        shape_t out_shape = mats[0].shape;
        for (size_t i = 1; i < mats.size(); i++) out_shape[axis] += mats[i].shape[axis];

        size_t outer = 1, inner = 1;
        for (size_t d = 0; d < axis; d++) outer *= out_shape[d];
        for (size_t d = axis + 1; d < rank; d++) inner *= out_shape[d];

        std::vector<T> out;
        out.reserve(numel(out_shape));
        for (size_t o = 0; o < outer; o++)
            for (const auto& m : mats) {
                const size_t blk = m.shape[axis] * inner;
                out.insert(out.end(), m.data.begin() + o * blk, m.data.begin() + (o + 1) * blk);
            }
        return Matrix<T>(std::move(out), std::move(out_shape));
    }

    static Matrix<T> concat(std::initializer_list<Matrix<T>> list, size_t axis)
    {
        if (list.size() == 0) return Matrix<T>();
        return Matrix<T>::concat(std::vector<Matrix<T>>(list.begin(), list.end()), axis);
    }

    static Matrix<T> where(const Matrix<bool>& cond, T if_true, T if_false)
    {
        std::vector<T> res;
        res.reserve(cond.data.size());
        for (size_t i = 0; i < cond.data.size(); i++) res.push_back(cond.data[i] ? if_true : if_false);
        return Matrix<T>(std::move(res), cond.shape);
    }

    // Semantics (kept): axis 0 / 1 CONCATENATE equally shaped 2-D matrices vertically / horizontally;
    // only axis 2 adds a new trailing dimension (np.stack(..., axis=-1)). All inputs must be 2-D with identical shapes.
    static Matrix<T> stack(const std::vector<Matrix<T>>& s, size_t axis)
    {
        if (s.empty()) throw std::invalid_argument("stack: empty input");
        for (const auto& m : s)
            if (m.shape.size() != 2) throw std::runtime_error("stack: all inputs must be 2-D");
        const size_t rows = s[0].shape[0], cols = s[0].shape[1];
        for (size_t i = 1; i < s.size(); i++)
            if (s[i].shape[0] != rows || s[i].shape[1] != cols)
                throw std::runtime_error("stack: all inputs must have the same shape");
        std::vector<T> res;
        res.reserve(rows * cols * s.size());

        if (axis == 0) {
            for (const auto& m : s) res.insert(res.end(), m.data.begin(), m.data.end());
            return Matrix<T>(std::move(res), shape_t{rows * s.size(), cols});
        } else if (axis == 1) {
            for (size_t r = 0; r < rows; r++)
                for (const auto& m : s) res.insert(res.end(), m.data.begin() + r * cols, m.data.begin() + (r + 1) * cols);
            return Matrix<T>(std::move(res), shape_t{rows, cols * s.size()});
        } else if (axis == 2) {
            for (size_t i = 0; i < rows * cols; i++)
                for (size_t d = 0; d < s.size(); d++) res.push_back(s[d].data[i]);
            return Matrix<T>(std::move(res), shape_t{rows, cols, s.size()});
        }
        throw std::runtime_error("stack: axis must be 0, 1, or 2");
    }

    static Matrix<T> stack(std::initializer_list<Matrix<T>> list, size_t axis)
    {
        if (list.size() == 0) return Matrix<T>();
        return Matrix<T>::stack(std::vector<Matrix<T>>(list.begin(), list.end()), axis);
    }

    static Matrix<T> arrange(T stop) { return Matrix<T>::arrange(T(0), stop, T(1)); }

    // B8: count computed in double (integer division made arrange<int>(0,5,2) = [0,2]).
    static Matrix<T> arrange(T start, T stop, T step = T(1))
    {
        const double st = static_cast<double>(scalar::to_acc<T>(step));
        if (st == 0.0) throw std::invalid_argument("arrange: step must not be zero");
        const double cnt = std::ceil((static_cast<double>(scalar::to_acc<T>(stop)) - static_cast<double>(scalar::to_acc<T>(start))) / st);
        if (!(cnt > 0)) return Matrix<T>(std::vector<T>{}, shape_t{0});
        const size_t n = static_cast<size_t>(cnt);
        std::vector<T> res;
        res.reserve(n);
        using A = scalar::acc_t<T>;
        for (size_t i = 0; i < n; i++)
            res.push_back(scalar::from_acc<T>(static_cast<A>(scalar::to_acc<T>(start) + static_cast<A>(i) * scalar::to_acc<T>(step))));
        return Matrix<T>(std::move(res), shape_t{n});
    }
    static Matrix<T> arange(T stop) { return arrange(stop); }
    static Matrix<T> arange(T start, T stop, T step = T(1)) { return arrange(start, stop, step); }

    static Matrix<T> zeros(shape_t shape) { return Matrix<T>(std::vector<T>(numel(shape), T(0)), shape); }
    static Matrix<T> ones(shape_t shape)  { return Matrix<T>(std::vector<T>(numel(shape), T(1)), shape); }
    static Matrix<T> zeros(std::initializer_list<size_t> inshape) { return Matrix<T>::zeros(Matrix<T>::getShape(inshape)); }
    static Matrix<T> ones(std::initializer_list<size_t> inshape)  { return Matrix<T>::ones(Matrix<T>::getShape(inshape)); }

    // Samples one index from the distribution `probs` (1-D, sums to 1).
    static Matrix<T> choice(size_t n, const Matrix<T>& probs, std::optional<unsigned int> seed = std::nullopt)
    {
        mxd::CallRng g(seed);
        if (n == 0 || n > probs.data.size()) throw std::invalid_argument("choice: n out of range");
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        const double r = dist(g());
        double cumsum = 0.0;
        for (size_t i = 0; i < n; i++) {
            cumsum += static_cast<double>(scalar::to_acc<T>(probs.data[i]));
            if (r <= cumsum) return Matrix<T>(std::vector<T>{static_cast<T>(i)}, shape_t{1});
        }
        return Matrix<T>(std::vector<T>{static_cast<T>(n - 1)}, shape_t{1});   // rounding fallback
    }

    // B10 [BEHAVIOR CHANGE]: random() is now uniform [0,1) from the shared generator (was std::rand()).
    static Matrix<T> random(shape_t shape, std::optional<unsigned int> seed = std::nullopt) { return Matrix<T>::randu(std::move(shape), seed); }
    static Matrix<T> random(std::initializer_list<size_t> inshape, std::optional<unsigned int> seed = std::nullopt) { return Matrix<T>::random(Matrix<T>::getShape(inshape), seed); }

    static Matrix<T> sin(const Matrix<T>& input) { return input.mapElems([](T x) { return scalar::sin<T>(x); }); }
    static Matrix<T> cos(const Matrix<T>& input) { return input.mapElems([](T x) { return scalar::cos<T>(x); }); }
    static Matrix<T> tan(const Matrix<T>& input) { return input.mapElems([](T x) { return scalar::tan<T>(x); }); }

    // B-note [BEHAVIOR CHANGE]: log() now clamps exactly like ln() (log(0) = log(1e-9) instead of -inf).
    static Matrix<T> log(const Matrix<T>& mat) { return mat.ln(); }

    static Matrix<T> randu(shape_t shape, std::optional<unsigned int> seed = std::nullopt)
    {
        mxd::CallRng g(seed);
        const size_t n = numel(shape);
        std::vector<T> res;
        res.reserve(n);
        for (size_t k = 0; k < n; k++) res.push_back(mxd::draw_uniform<T>(0.0, 1.0, g()));
        return Matrix<T>(std::move(res), std::move(shape));
    }


    static Matrix<T> randu(T start, T stop, shape_t shape, std::optional<unsigned int> seed = std::nullopt)
    {
        mxd::CallRng g(seed);
        const size_t n = numel(shape);
        std::vector<T> res;
        res.reserve(n);
        if constexpr (std::is_integral_v<T>) {
            if (stop <= start) throw std::invalid_argument("randu: stop must be > start");
            std::uniform_int_distribution<long long> dist(static_cast<long long>(start), static_cast<long long>(stop) - 1);
            for (size_t k = 0; k < n; k++) res.push_back(static_cast<T>(dist(g())));
        } else {
            const double lo = static_cast<double>(scalar::to_acc<T>(start)), hi = static_cast<double>(scalar::to_acc<T>(stop));
            for (size_t k = 0; k < n; k++) res.push_back(mxd::draw_uniform<T>(lo, hi, g()));
        }
        return Matrix<T>(std::move(res), std::move(shape));
    }

    static Matrix<T> randu(std::initializer_list<size_t> inshape, std::optional<unsigned int> seed = std::nullopt) { return Matrix<T>::randu(Matrix<T>::getShape(inshape), seed); }
    static Matrix<T> randu(T start, T stop, std::initializer_list<size_t> inshape, std::optional<unsigned int> seed = std::nullopt) { return Matrix<T>::randu(start, stop, Matrix<T>::getShape(inshape), seed); }

    static void manual_seed(unsigned int seed) { get_gen(seed); }

    static Matrix<T> randn(shape_t shape, std::optional<unsigned int> seed = std::nullopt)
    {
        mxd::CallRng g(seed);
        const size_t n = numel(shape);
        std::vector<T> res;
        res.reserve(n);
        for (size_t k = 0; k < n; k++) res.push_back(mxd::draw_normal<T>(0.0, 1.0, g()));
        return Matrix<T>(std::move(res), std::move(shape));
    }

    static Matrix<T> randomn(std::initializer_list<size_t> s, std::optional<unsigned int> seed = std::nullopt) { return randn(getShape(s), seed); }
    static Matrix<T> randomn(shape_t s, std::optional<unsigned int> seed = std::nullopt)                       { return randn(std::move(s), seed); }

    // He / Kaiming normal: std = sqrt(2 / fan_in), fan_in = shape[0].
    static Matrix<T> he(shape_t shape, std::optional<unsigned int> seed = std::nullopt)
    {
        mxd::CallRng g(seed);
        if (shape.empty() || shape[0] == 0) throw std::invalid_argument("he: shape[0] (fan_in) must be > 0");
        const double sd = std::sqrt(2.0 / static_cast<double>(shape[0]));
        const size_t n = numel(shape);
        std::vector<T> res;
        res.reserve(n);
        for (size_t k = 0; k < n; k++) res.push_back(mxd::draw_normal<T>(0.0, sd, g()));
        return Matrix<T>(std::move(res), std::move(shape));
    }
    static Matrix<T> he(std::initializer_list<size_t> inshape, std::optional<unsigned int> seed = std::nullopt) { return Matrix<T>::he(Matrix<T>::getShape(inshape), seed); }

    static Matrix<T> eye(size_t n)
    {
        std::vector<T> res(n * n, T(0));
        for (size_t i = 0; i < n; i++) res[i * n + i] = T(1);
        return Matrix<T>(std::move(res), shape_t{n, n});
    }
    static Matrix<T> eye(std::initializer_list<size_t> s) { return eye(getShape(s)[0]); }

    // Lower triangle (incl. diagonal) of ones.
    static Matrix<T> tril(size_t n)
    {
        std::vector<T> res(n * n, T(0));
        for (size_t i = 0; i < n; i++) for (size_t j = 0; j <= i; j++) res[i * n + j] = T(1);
        return Matrix<T>(std::move(res), shape_t{n, n});
    }

    // B12: uses the last two dims and loops over leading batch dims.
    static Matrix<T> tril(const Matrix<T>& input)
    {
        if (input.shape.size() < 2) throw std::runtime_error("tril: needs at least 2 dimensions");
        const size_t rows = input.shape[input.shape.size() - 2], cols = input.shape.back();
        std::vector<T> res = input.data;
        if (rows * cols == 0) return Matrix<T>(std::move(res), input.shape);
        const size_t batch = res.size() / (rows * cols);
        for (size_t b = 0; b < batch; b++)
            for (size_t i = 0; i < rows; i++)
                for (size_t j = i + 1; j < cols; j++) res[b * rows * cols + i * cols + j] = T(0);
        return Matrix<T>(std::move(res), input.shape);
    }

    static Matrix<T> triup(size_t n)
    {
        std::vector<T> res(n * n, T(0));
        for (size_t i = 0; i < n; i++) for (size_t j = i; j < n; j++) res[i * n + j] = T(1);
        return Matrix<T>(std::move(res), shape_t{n, n});
    }

    static Matrix<T> triup(const Matrix<T>& input)
    {
        if (input.shape.size() < 2) throw std::runtime_error("triup: needs at least 2 dimensions");
        const size_t rows = input.shape[input.shape.size() - 2], cols = input.shape.back();
        std::vector<T> res = input.data;
        if (rows * cols == 0) return Matrix<T>(std::move(res), input.shape);
        const size_t batch = res.size() / (rows * cols);
        for (size_t b = 0; b < batch; b++)
            for (size_t i = 0; i < rows; i++)
                for (size_t j = 0; j < i && j < cols; j++) res[b * rows * cols + i * cols + j] = T(0);
        return Matrix<T>(std::move(res), input.shape);
    }
    static Matrix<T> triu(size_t n) { return triup(n); }
    static Matrix<T> triu(const Matrix<T>& input) { return triup(input); }

    static Matrix<T> one_hot(const Matrix<T>& labels, size_t num_classes)
    {
        const size_t n = labels.get_size();
        std::vector<T> res(n * num_classes, T(0));
        for (size_t i = 0; i < n; i++) {
            const double v = static_cast<double>(scalar::to_acc<T>(labels.data[i]));
            if (!(v >= 0) || v >= static_cast<double>(num_classes))
                throw std::out_of_range("one_hot: label " + std::to_string(v) + " outside [0, " + std::to_string(num_classes) + ")");
            res[i * num_classes + static_cast<size_t>(v)] = T(1);
        }
        return Matrix<T>(std::move(res), shape_t{n, num_classes});
    }

    // ───────────────────────────── in-place helpers ─────────────────────────────

    void ones()  { data.assign(numel(shape), T(1)); }
    void zeros() { data.assign(numel(shape), T(0)); }

    void copy_from(const Matrix<T>& two) { *this = two; }
    void copy_from(Matrix<T>& two)       { *this = two; }
    void copy_from(Matrix<T>* two)
    {
        if (two == nullptr) throw std::runtime_error("copy_from: null pointer input\n");
        *this = *two;
    }

    // B2 [BEHAVIOR CHANGE]: np.maximum(x, a): values below `a` become `a` (they used to become 0).
    // Identical for a == 0 (the ReLU case).
    Matrix<T> maximum(const T a) const
    {
        return mapElems([a](T x) { return (x < a) ? a : x; });
    }

    void clear()
    {
        data.clear();
        shape.clear();
    }

    // ───────────────────────────── transpose ─────────────────────────────

    // Reverses all axes. 1-D -> {n,1} (kept, differs from numpy).
    Matrix<T> transpose() const
    {
        if (shape.size() == 1) return transpose_1D();
        if (shape.size() == 2) return Matrix<T>(transpose_2D(), shape_t{shape[1], shape[0]});
        shape_t perm(shape.size());
        for (size_t i = 0; i < perm.size(); i++) perm[i] = perm.size() - 1 - i;
        return permute(perm);
    }

    Matrix<T> transpose(shape_t perm) const
    {
        if (shape.size() == 1) return transpose_1D();
        if (perm.size() != shape.size())
            throw std::runtime_error("transpose: perm size must match number of dimensions\n");
        std::vector<bool> seen(perm.size(), false);
        for (size_t p : perm) {
            if (p >= perm.size() || seen[p]) throw std::runtime_error("transpose: perm is not a permutation\n");
            seen[p] = true;
        }
        if (shape.size() == 2) {
            if (perm[0] == 0 && perm[1] == 1) return *this;     // B7: identity perm is honored
            return Matrix<T>(transpose_2D(), shape_t{shape[1], shape[0]});
        }
        return permute(perm);
    }

    Matrix<T> transpose(std::initializer_list<size_t> inperm) const { return transpose(Matrix<T>::getShape(inperm)); }

    // ───────────────────────────── products ─────────────────────────────

    // numpy matmul: 1-D operands are promoted, batch dims broadcast right-aligned.
    Matrix<T> matmul(const Matrix<T>& rhs) const
    {
        if constexpr (std::is_same_v<T, bool>) {
            throw std::runtime_error("matmul: not supported for bool");
        } else {
            const size_t lr = shape.size(), rr = rhs.shape.size();
            if (lr == 0 || rr == 0) throw std::runtime_error("matmul: empty shape");
            if (areShapes1D(shape, rhs.shape))
                throw std::runtime_error("matmul: cannot multiply two 1D tensors, use dot() instead\n");

            shape_t ls = shape, rs = rhs.shape;
            if (lr == 1) ls = shape_t{1, shape[0]};
            if (rr == 1) rs = shape_t{rhs.shape[0], 1};

            const size_t M = ls[ls.size() - 2], K = ls.back(), K2 = rs[rs.size() - 2], N = rs.back();
            if (K != K2)
                throw std::invalid_argument("matmul: inner dimensions do not match: " + mxd::shape_str(shape) + " @ " + mxd::shape_str(rhs.shape) + "\n");

            const shape_t bl(ls.begin(), ls.end() - 2), br(rs.begin(), rs.end() - 2);
            shape_t bs;
            if (!mxd::broadcast_shapes(bl, br, bs))
                throw std::invalid_argument("matmul: batch dimensions are not broadcastable: " + mxd::shape_str(shape) + " @ " + mxd::shape_str(rhs.shape) + "\n");

            // effective per-batch-dim element strides (0 for broadcast dims)
            const shape_t sl = computeShapes(ls), sr = computeShapes(rs);
            std::vector<size_t> esl(bs.size(), 0), esr(bs.size(), 0);
            for (size_t d = 0; d < bs.size(); d++) {
                if (d + bl.size() >= bs.size()) { size_t j = d + bl.size() - bs.size(); esl[d] = (bl[j] == 1) ? 0 : sl[j]; }
                if (d + br.size() >= bs.size()) { size_t j = d + br.size() - bs.size(); esr[d] = (br[j] == 1) ? 0 : sr[j]; }
            }

            const size_t nb = numel(bs);
            std::vector<T> out(nb * M * N);
            std::vector<size_t> idx(bs.size(), 0);
            size_t lo = 0, ro = 0;
            for (size_t b = 0; b < nb; b++) {
                mxd::gemm<T>(data.data() + lo, rhs.data.data() + ro, out.data() + b * M * N, M, N, K);
                for (size_t d = bs.size(); d-- > 0;) {
                    idx[d]++; lo += esl[d]; ro += esr[d];
                    if (idx[d] < bs[d]) break;
                    lo -= esl[d] * bs[d]; ro -= esr[d] * bs[d]; idx[d] = 0;
                }
            }

            shape_t os = bs;
            os.push_back(M);
            os.push_back(N);
            if (lr == 1) os.erase(os.end() - 2);
            if (rr == 1) os.pop_back();
            return Matrix<T>(std::move(out), std::move(os));
        }
    }

    Matrix<T> dot(const Matrix<T>& rhs) const
    {
        if (areShapes1D(shape, rhs.shape))
            return Matrix<T>(std::vector<T>{dotProduct1D(data, rhs.data)}, shape_t{1});
        if (areShapes2D(shape, rhs.shape))
            return dotProduct2D(rhs);        // [DECISION] flattened inner product, see dotProduct2D
        if (!dotShapesAssert(rhs.shape))
            throw std::runtime_error("dot: invalid shapes for dot product\n");
        return matmul(rhs);
    }

    // ───────────────────────────── printing ─────────────────────────────

    // indexStack entries are element offsets already (old contract).
    std::ostream& print(std::ostream& out, const shape_t& indexStack, size_t dim) const
    {
        size_t offset = 0;
        for (size_t o : indexStack) offset += o;
        return printAt(out, offset, dim);
    }

    std::ostream& print(std::ostream& out) const { return printAt(out, 0, 0); }

private:
    std::ostream& printAt(std::ostream& out, size_t offset, size_t dim) const
    {
        if (shape.empty()) {                           // 0-D
            out << " [";
            if (!data.empty()) out << mxd::printable(data[0]) << ",";
            out << "]\n";
            return out;
        }
        if (dim == shape.size() - 1) {
            out << " [";
            for (size_t i = 0; i < shape[dim]; i++) out << mxd::printable(data[offset + i]) << ",";
            out << "]\n";
            return out;
        }
        const shape_t st = computeShapes(shape);
        out << "[\n";
        for (size_t i = 0; i < shape[dim]; i++) printAt(out, offset + st[dim] * i, dim + 1);
        out << "]";
        return out;
    }
};

#include "MatrixOps.hpp"
