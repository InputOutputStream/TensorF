#pragma once
// VectorMath.hpp — element-wise math on std::vector<T> as NAMED functions in namespace `vecmath`.
//
// This is what Matrix<T> uses. There are NO operators and NOTHING in the global namespace here, so including
// this header can never change the meaning of an expression in unrelated code.
// The operator syntax (a + b on vectors) lives in the opt-in header Overloads/Overload.hpp and only forwards here.
//
// Every function is constrained to numeric element types (arithmetic, FP8, FP4, float16); std::vector<std::string>,
// std::vector<MyStruct>, ... never match.
//
// Semantics (unchanged from Overload.hpp):
//   * vector (op) vector throws std::invalid_argument on a size mismatch (no silent truncation)
//   * division by zero throws std::runtime_error ("Division by zero"); define MATRIX_IEEE_DIVISION to let
//     floating-point types follow IEEE (inf/NaN). Integral division by zero ALWAYS throws (it is UB otherwise).

#include <algorithm>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "core/Types/scalar.hpp"

namespace vecmath {

template <typename T>
concept Element = std::is_arithmetic_v<T> || scalar::is_low_precision_v<T>;

namespace impl {

template <typename T>
inline void div_check(const T& d) {
#ifdef MATRIX_IEEE_DIVISION
    if constexpr (std::is_integral_v<T>)
#endif
    {
        if (d == T(0)) throw std::runtime_error("Division by zero");
    }
}

// ── Kernel shapes ────────────────────────────────────────────────────────────────────────────────────────────
// The loops below used to be `r.reserve(n); for (...) r.push_back(f(a[i]))`. push_back re-checks capacity and
// stores the end pointer on every element, so the compiler cannot vectorise it: about 1.4 ns/element even for data
// that sits in L1. These helpers write through raw pointers instead, in blocks of a CONSTANT width. A constant trip
// count is what lets GCC (whose -O2 "very-cheap" cost model refuses a loop with an unknown trip count) and clang
// vectorise at plain -O2, with no flags; the scalar tail handles the remaining n % kBlock elements.
// Results are bit-identical to the old loops: each element is computed by the same expression, in the same type.
// Output and inputs must not alias (callers pass a fresh result vector).
inline constexpr size_t kBlock = 16;

// Ordered comparisons (< <= > >=) on floating point are SIGNALLING: they may raise FE_INVALID on a NaN. Under the
// default -ftrapping-math GCC therefore refuses to if-convert or vectorise a loop around them and emits a
// data-dependent branch per element, which mispredicts about half the time on data with random signs (6.5 ns per
// element instead of 0.4). The builtins below are the quiet versions: same results for every input including NaN
// (false), no exception flag, vectorisable. Other types keep the plain operators.
template <class T> inline bool gt_q(const T& a, const T& b) {
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (std::is_floating_point_v<T>) return __builtin_isgreater(a, b);
#endif
    return a > b;
}
template <class T> inline bool ge_q(const T& a, const T& b) {
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (std::is_floating_point_v<T>) return __builtin_isgreaterequal(a, b);
#endif
    return a >= b;
}
template <class T> inline bool lt_q(const T& a, const T& b) {
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (std::is_floating_point_v<T>) return __builtin_isless(a, b);
#endif
    return a < b;
}
template <class T> inline bool le_q(const T& a, const T& b) {
#if defined(__GNUC__) || defined(__clang__)
    if constexpr (std::is_floating_point_v<T>) return __builtin_islessequal(a, b);
#endif
    return a <= b;
}

template <class O, class I, class F>
inline void map1(O* __restrict o, const I* __restrict a, size_t n, F f) {
    size_t i = 0;
    for (; i + kBlock <= n; i += kBlock)
        for (size_t k = 0; k < kBlock; ++k) o[i + k] = f(a[i + k]);
    for (; i < n; ++i) o[i] = f(a[i]);
}

template <class O, class I, class F>
inline void map2(O* __restrict o, const I* __restrict a, const I* __restrict b, size_t n, F f) {
    size_t i = 0;
    for (; i + kBlock <= n; i += kBlock)
        for (size_t k = 0; k < kBlock; ++k) o[i + k] = f(a[i + k], b[i + k]);
    for (; i < n; ++i) o[i] = f(a[i], b[i]);
}

// a[i] = f(a[i], b[i]); a and b are different vectors (an exact self-alias is routed to map1 by the callers).
template <class T, class F>
inline void update2(T* __restrict a, const T* __restrict b, size_t n, F f) {
    size_t i = 0;
    for (; i + kBlock <= n; i += kBlock)
        for (size_t k = 0; k < kBlock; ++k) a[i + k] = f(a[i + k], b[i + k]);
    for (; i < n; ++i) a[i] = f(a[i], b[i]);
}

template <class T, class F>
inline void update1(T* __restrict a, size_t n, F f) {
    size_t i = 0;
    for (; i + kBlock <= n; i += kBlock)
        for (size_t k = 0; k < kBlock; ++k) a[i + k] = f(a[i + k]);
    for (; i < n; ++i) a[i] = f(a[i]);
}

// Container-level entry points. std::vector<bool> is bit-packed and has no data(), so for bool the plain indexed loop is
// used (as the push_back loops always did); every other element type goes through the pointer kernels above.
template <class O, class I, class F>
inline void map1v(std::vector<O>& o, const std::vector<I>& a, F f) {
    if constexpr (std::is_same_v<O, bool> || std::is_same_v<I, bool>) { for (size_t i = 0; i < a.size(); ++i) o[i] = f(a[i]); }
    else map1(o.data(), a.data(), a.size(), f);
}
template <class O, class I, class F>
inline void map2v(std::vector<O>& o, const std::vector<I>& a, const std::vector<I>& b, F f) {
    if constexpr (std::is_same_v<O, bool> || std::is_same_v<I, bool>) { for (size_t i = 0; i < a.size(); ++i) o[i] = f(a[i], b[i]); }
    else map2(o.data(), a.data(), b.data(), a.size(), f);
}
template <class T, class F>
inline void update1v(std::vector<T>& a, F f) {
    if constexpr (std::is_same_v<T, bool>) { for (size_t i = 0; i < a.size(); ++i) a[i] = f(a[i]); }
    else update1(a.data(), a.size(), f);
}
template <class T, class F>
inline void update2v(std::vector<T>& a, const std::vector<T>& b, F f) {
    if constexpr (std::is_same_v<T, bool>) { for (size_t i = 0; i < a.size(); ++i) a[i] = f(a[i], b[i]); }
    else update2(a.data(), b.data(), a.size(), f);
}
// ── Optional fast exp (-DTENSORF_FAST_EXP, float only, off by default) ─────────────────────────────────────
// std::exp cannot be vectorised, so a sigmoid / exponent() over N floats runs N scalar calls (about 3.4 ns each) and that
// call is then most of the cost of the op. exp_k is the single place the element-wise kernels take their exp from:
//   default:            scalar::exp<T>, i.e. std::exp, bit-identical to every earlier version (what saved models and
//                       reproducibility checks expect);
//   TENSORF_FAST_EXP:   for float, a branch-free Cephes polynomial the compiler vectorises (about 5-8x faster per element).
//                       Max error about 1 ulp against std::exp over the whole range; the results are NOT bit-identical.
//                       Arguments above 88.7228 give +inf and below -87.3365 give 0 (std::exp returns denormals there).
// double and the low-precision types always use scalar::exp.
#ifdef TENSORF_FAST_EXP
// Branch-free select (c ? a : b) on the bit patterns. A float ternary here would be if-converted only if the compiler may
// speculate both arms; under -ftrapping-math GCC refuses and the loop stays scalar (measured: 7 ns/elem, slower than std::exp).
inline float sel_bits(bool c, float a, float b) {
    uint32_t ua, ub; std::memcpy(&ua, &a, sizeof ua); std::memcpy(&ub, &b, sizeof ub);
    const uint32_t m = 0u - (uint32_t)c;
    const uint32_t r = (ua & m) | (ub & ~m);
    float o; std::memcpy(&o, &r, sizeof o);
    return o;
}
inline float fast_expf(float x) {
    const float hi = 88.72283905206835f, lo = -87.33654475055310f;
    const float inf = std::numeric_limits<float>::infinity();
    float xc = sel_bits(gt_q(x, hi), hi, x);                     // clamp (NaN passes through both compares unchanged)
    xc = sel_bits(lt_q(x, lo), lo, xc);
    // exp(x) = 2^n * exp(r),  n = round(x / ln 2),  r = x - n ln 2 (ln 2 split in two parts for accuracy)
    float fx = xc * 1.44269504088896341f + 0.5f;
    int n = (int)fx;
    n -= (int)lt_q(fx, (float)n);                                // floor without a libm call
    const float fn = (float)n;
    float r = xc - fn * 0.693359375f;
    r = r - fn * (-2.12194440e-4f);
    float y = 1.9875691500E-4f;
    y = y * r + 1.3981999507E-3f;
    y = y * r + 8.3334519073E-3f;
    y = y * r + 4.1665795894E-2f;
    y = y * r + 1.6666665459E-1f;
    y = y * r + 5.0000001201E-1f;
    y = y * (r * r) + r + 1.0f;
    // 2^n by building exponent fields. n spans [-126, 128] and 2^128 / 2^-126.. do not fit one float exponent field, so
    // the scale is split into two factors 2^n1 * 2^n2, each comfortably normal (n1, n2 in [-63, 64]).
    const int n1 = n >> 1, n2 = n - n1;
    const uint32_t b1 = (uint32_t)(n1 + 127) << 23, b2 = (uint32_t)(n2 + 127) << 23;
    float s1, s2; std::memcpy(&s1, &b1, sizeof s1); std::memcpy(&s2, &b2, sizeof s2);
    float out = (y * s1) * s2;
    out = sel_bits(gt_q(x, hi), inf, out);                       // overflow -> +inf (also +inf in -> +inf out)
    out = sel_bits(lt_q(x, lo), 0.0f, out);                      // underflow -> 0  (also -inf in -> 0)
    return out;
}
#endif

template <class T> inline T exp_k(T x) {
#ifdef TENSORF_FAST_EXP
    if constexpr (std::is_same_v<T, float>) return fast_expf(x);
#endif
    return scalar::exp<T>(x);
}

// Throws unless every divisor is non-zero: one pass that vectorises, instead of a branch per element.
template <class T>
inline void div_check_all(const T* b, size_t n) {
#ifdef MATRIX_IEEE_DIVISION
    if constexpr (!std::is_integral_v<T>) return;
#endif
    // An integer count, not a bool OR: GCC vectorises the former (compare + subtract mask) and not the latter.
    unsigned zeros = 0;
    size_t i = 0;
    for (; i + kBlock <= n; i += kBlock) {
        unsigned z = 0;
        for (size_t k = 0; k < kBlock; ++k) z += (unsigned)(b[i + k] == T(0));
        zeros += z;
    }
    for (; i < n; ++i) zeros += (unsigned)(b[i] == T(0));
    const bool zero = zeros != 0;
    if (zero) throw std::runtime_error("Division by zero");
}

template <class T>
inline void div_check_allv(const std::vector<T>& b) {
    if constexpr (std::is_same_v<T, bool>) { for (size_t i = 0; i < b.size(); ++i) div_check(b[i]); }
    else div_check_all(b.data(), b.size());
}

template <typename T>
inline void same_size(const std::vector<T>& a, const std::vector<T>& b, const char* op) {
    if (a.size() != b.size())
        throw std::invalid_argument(std::string("vecmath::") + op + ": size mismatch (" +
                                    std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
}

} // namespace impl

// ───────────────────────────── vector (op) vector ─────────────────────────────
template <Element T> std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "add"); std::vector<T> r(a.size());
    impl::map2v(r, a, b, [](T x, T y) { return (T)(x + y); });
    return r;
}
template <Element T> std::vector<T> sub(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "sub"); std::vector<T> r(a.size());
    impl::map2v(r, a, b, [](T x, T y) { return (T)(x - y); });
    return r;
}
template <Element T> std::vector<T> mul(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "mul"); std::vector<T> r(a.size());
    impl::map2v(r, a, b, [](T x, T y) { return (T)(x * y); });
    return r;
}
template <Element T> std::vector<T> div(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "div"); impl::div_check_allv(b); std::vector<T> r(a.size());
    impl::map2v(r, a, b, [](T x, T y) { return (T)(x / y); });
    return r;
}

// ───────────────────────────── vector (op) scalar, scalar (op) vector ─────────────────────────────
// Scalar parameters are non-deduced (std::type_identity_t<T>) so add_s(vec_of_float, 2) works.
template <Element T> std::vector<T> add_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b + a  ==  a + b
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(x + a); });
    return r;
}
template <Element T> std::vector<T> mul_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b * a  ==  a * b
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(x * a); });
    return r;
}
template <Element T> std::vector<T> sub_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b - a
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(x - a); });
    return r;
}
template <Element T> std::vector<T> s_sub(std::type_identity_t<T> a, const std::vector<T>& b) {   // a - b
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(a - x); });
    return r;
}
template <Element T> std::vector<T> div_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b / a
    impl::div_check(a);
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(x / a); });
    return r;
}
template <Element T> std::vector<T> s_div(std::type_identity_t<T> a, const std::vector<T>& b) {   // a / b
    impl::div_check_allv(b);
    std::vector<T> r(b.size());
    impl::map1v(r, b, [a](T x) { return (T)(a / x); });
    return r;
}
template <Element T> std::vector<T> neg(const std::vector<T>& a) {
    std::vector<T> r(a.size());
    impl::map1v(r, a, [](T x) { return (T)(-x); });
    return r;
}

// modulo (integral %0 throws)
template <Element T> std::vector<T> mod_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b % a
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) {
        if constexpr (std::is_integral_v<T>) {
            if (a == T(0)) throw std::runtime_error("Modulo by zero");
            r.push_back((T)(b[i] % a));
        } else r.push_back((T)std::fmod(scalar::to_acc<T>(b[i]), scalar::to_acc<T>(a)));
    }
    return r;
}
template <Element T> std::vector<T> s_mod(std::type_identity_t<T> a, const std::vector<T>& b) {   // a % b
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) {
        if constexpr (std::is_integral_v<T>) {
            if (b[i] == T(0)) throw std::runtime_error("Modulo by zero");
            r.push_back((T)(a % b[i]));
        } else r.push_back((T)std::fmod(scalar::to_acc<T>(a), scalar::to_acc<T>(b[i])));
    }
    return r;
}

// ───────────────────────────── in-place ─────────────────────────────
// `a op= a` (the same vector twice) is legal and must keep working; it is routed to the single-pointer kernel.
template <Element T> void add_inplace(std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "add_inplace");
    if (&a == &b) impl::update1v(a, [](T x) { return (T)(x + x); });
    else impl::update2v(a, b, [](T x, T y) { return (T)(x + y); });
}
template <Element T> void sub_inplace(std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "sub_inplace");
    if (&a == &b) impl::update1v(a, [](T x) { return (T)(x - x); });
    else impl::update2v(a, b, [](T x, T y) { return (T)(x - y); });
}
template <Element T> void mul_inplace(std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "mul_inplace");
    if (&a == &b) impl::update1v(a, [](T x) { return (T)(x * x); });
    else impl::update2v(a, b, [](T x, T y) { return (T)(x * y); });
}
template <Element T> void div_inplace(std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "div_inplace");
    impl::div_check_allv(b);
    if (&a == &b) impl::update1v(a, [](T x) { return (T)(x / x); });
    else impl::update2v(a, b, [](T x, T y) { return (T)(x / y); });
}
template <Element T> void add_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { impl::update1v(a, [b](T x) { x += b; return x; }); }
template <Element T> void sub_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { impl::update1v(a, [b](T x) { x -= b; return x; }); }
template <Element T> void mul_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { impl::update1v(a, [b](T x) { x *= b; return x; }); }
template <Element T> void div_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { impl::div_check(b); impl::update1v(a, [b](T x) { x /= b; return x; }); }

// ───────────────────────────── comparisons ─────────────────────────────
// cmp_xx(vec, s): vec[i] xx s      cmp_xx(s, vec): s xx vec[i]       (result: 0/1 bytes)
#define VECMATH_CMP(NAME, CMP)                                                                             \
    template <Element T> std::vector<uint8_t> NAME(const std::vector<T>& a, std::type_identity_t<T> b) {   \
        std::vector<uint8_t> r(a.size());                                                                  \
        impl::map1v(r, a, [b](T x) { return (uint8_t)(CMP(x, b)); });               \
        return r; }                                                                                        \
    template <Element T> std::vector<uint8_t> NAME(std::type_identity_t<T> b, const std::vector<T>& a) {   \
        std::vector<uint8_t> r(a.size());                                                                  \
        impl::map1v(r, a, [b](T x) { return (uint8_t)(CMP(b, x)); });               \
        return r; }
VECMATH_CMP(cmp_lt, impl::lt_q)
VECMATH_CMP(cmp_le, impl::le_q)
VECMATH_CMP(cmp_gt, impl::gt_q)
VECMATH_CMP(cmp_ge, impl::ge_q)
#undef VECMATH_CMP

template <Element T> bool equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (!(a[i] == b[i])) return false;
    return true;
}
template <Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> eq_s(const std::vector<T>& a, const U& b) {
    const T bt = static_cast<T>(b);
    std::vector<uint8_t> r(a.size());
    impl::map1v(r, a, [bt](T x) { return (uint8_t)(x == bt); });
    return r;
}
template <Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> ne_s(const std::vector<T>& a, const U& b) {
    const T bt = static_cast<T>(b);
    std::vector<uint8_t> r(a.size());
    impl::map1v(r, a, [bt](T x) { return (uint8_t)(x != bt); });
    return r;
}

// ───────────────────────────── math ─────────────────────────────
template <Element T> std::vector<T> exponent(const std::vector<T>& a) {
    std::vector<T> r(a.size());
    impl::map1v(r, a, [](T x) { return impl::exp_k<T>(x); });
    return r;
}

template <Element T> std::vector<T> pow_s(const std::vector<T>& a, std::type_identity_t<T> n) {
    std::vector<T> r(a.size());
    if (n == T(2)) { impl::map1v(r, a, [](T x) { return (T)(x * x); }); return r; }
    if (n == T(3)) { impl::map1v(r, a, [](T x) { return (T)(x * x * x); }); return r; }
    if (n == T(4)) { impl::map1v(r, a, [](T x) { return (T)(x * x * x * x); }); return r; }
    impl::map1v(r, a, [n](T x) { return scalar::pow(x, n); });
    return r;
}

// b.size()==1 is a documented scalar broadcast.
template <Element T> std::vector<T> pow_v(const std::vector<T>& a, const std::vector<T>& b) {
    if (b.size() == 1) return pow_s<T>(a, b[0]);
    impl::same_size(a, b, "pow_v");
    std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back(scalar::pow(a[i], b[i]));
    return r;
}

template <Element T> bool has_nan_or_inf(const std::vector<T>& v) {
    for (size_t i = 0; i < v.size(); i++)
        if (scalar::isnan(v[i]) || scalar::isinf(v[i])) return true;
    return false;
}

} // namespace vecmath
