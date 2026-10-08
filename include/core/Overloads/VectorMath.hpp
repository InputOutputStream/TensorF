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

template <typename T>
inline void same_size(const std::vector<T>& a, const std::vector<T>& b, const char* op) {
    if (a.size() != b.size())
        throw std::invalid_argument(std::string("vecmath::") + op + ": size mismatch (" +
                                    std::to_string(a.size()) + " vs " + std::to_string(b.size()) + ")");
}

} // namespace impl

// ───────────────────────────── vector (op) vector ─────────────────────────────
template <Element T> std::vector<T> add(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "add"); std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] + b[i]));
    return r;
}
template <Element T> std::vector<T> sub(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "sub"); std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] - b[i]));
    return r;
}
template <Element T> std::vector<T> mul(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "mul"); std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] * b[i]));
    return r;
}
template <Element T> std::vector<T> div(const std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "div"); std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) { impl::div_check(b[i]); r.push_back((T)(a[i] / b[i])); }
    return r;
}

// ───────────────────────────── vector (op) scalar, scalar (op) vector ─────────────────────────────
// Scalar parameters are non-deduced (std::type_identity_t<T>) so add_s(vec_of_float, 2) works.
template <Element T> std::vector<T> add_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b + a  ==  a + b
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) r.push_back((T)(b[i] + a));
    return r;
}
template <Element T> std::vector<T> mul_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b * a  ==  a * b
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) r.push_back((T)(b[i] * a));
    return r;
}
template <Element T> std::vector<T> sub_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b - a
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) r.push_back((T)(b[i] - a));
    return r;
}
template <Element T> std::vector<T> s_sub(std::type_identity_t<T> a, const std::vector<T>& b) {   // a - b
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) r.push_back((T)(a - b[i]));
    return r;
}
template <Element T> std::vector<T> div_s(const std::vector<T>& b, std::type_identity_t<T> a) {   // b / a
    impl::div_check(a);
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) r.push_back((T)(b[i] / a));
    return r;
}
template <Element T> std::vector<T> s_div(std::type_identity_t<T> a, const std::vector<T>& b) {   // a / b
    std::vector<T> r; r.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++) { impl::div_check(b[i]); r.push_back((T)(a / b[i])); }
    return r;
}
template <Element T> std::vector<T> neg(const std::vector<T>& a) {
    std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back((T)(-a[i]));
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
template <Element T> void add_inplace(std::vector<T>& a, const std::vector<T>& b) { impl::same_size(a, b, "add_inplace"); for (size_t i = 0; i < a.size(); i++) a[i] += b[i]; }
template <Element T> void sub_inplace(std::vector<T>& a, const std::vector<T>& b) { impl::same_size(a, b, "sub_inplace"); for (size_t i = 0; i < a.size(); i++) a[i] -= b[i]; }
template <Element T> void mul_inplace(std::vector<T>& a, const std::vector<T>& b) { impl::same_size(a, b, "mul_inplace"); for (size_t i = 0; i < a.size(); i++) a[i] *= b[i]; }
template <Element T> void div_inplace(std::vector<T>& a, const std::vector<T>& b) {
    impl::same_size(a, b, "div_inplace");
    for (size_t i = 0; i < a.size(); i++) { impl::div_check(b[i]); a[i] /= b[i]; }
}
template <Element T> void add_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { for (size_t i = 0; i < a.size(); i++) a[i] += b; }
template <Element T> void sub_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { for (size_t i = 0; i < a.size(); i++) a[i] -= b; }
template <Element T> void mul_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { for (size_t i = 0; i < a.size(); i++) a[i] *= b; }
template <Element T> void div_inplace_s(std::vector<T>& a, std::type_identity_t<T> b) { impl::div_check(b); for (size_t i = 0; i < a.size(); i++) a[i] /= b; }

// ───────────────────────────── comparisons ─────────────────────────────
// cmp_xx(vec, s): vec[i] xx s      cmp_xx(s, vec): s xx vec[i]       (result: 0/1 bytes)
#define VECMATH_CMP(NAME, OP)                                                                              \
    template <Element T> std::vector<uint8_t> NAME(const std::vector<T>& a, std::type_identity_t<T> b) {   \
        std::vector<uint8_t> r; r.reserve(a.size());                                                       \
        for (size_t i = 0; i < a.size(); i++) r.push_back(a[i] OP b);                                      \
        return r; }                                                                                        \
    template <Element T> std::vector<uint8_t> NAME(std::type_identity_t<T> b, const std::vector<T>& a) {   \
        std::vector<uint8_t> r; r.reserve(a.size());                                                       \
        for (size_t i = 0; i < a.size(); i++) r.push_back(b OP a[i]);                                      \
        return r; }
VECMATH_CMP(cmp_lt, <)
VECMATH_CMP(cmp_le, <=)
VECMATH_CMP(cmp_gt, >)
VECMATH_CMP(cmp_ge, >=)
#undef VECMATH_CMP

template <Element T> bool equal(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (!(a[i] == b[i])) return false;
    return true;
}
template <Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> eq_s(const std::vector<T>& a, const U& b) {
    std::vector<uint8_t> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back(a[i] == static_cast<T>(b));
    return r;
}
template <Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> ne_s(const std::vector<T>& a, const U& b) {
    std::vector<uint8_t> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back(a[i] != static_cast<T>(b));
    return r;
}

// ───────────────────────────── math ─────────────────────────────
template <Element T> std::vector<T> exponent(const std::vector<T>& a) {
    std::vector<T> r; r.reserve(a.size());
    for (size_t i = 0; i < a.size(); i++) r.push_back(scalar::exp(a[i]));
    return r;
}

template <Element T> std::vector<T> pow_s(const std::vector<T>& a, std::type_identity_t<T> n) {
    std::vector<T> r; r.reserve(a.size());
    if (n == T(2)) { for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] * a[i])); return r; }
    if (n == T(3)) { for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] * a[i] * a[i])); return r; }
    if (n == T(4)) { for (size_t i = 0; i < a.size(); i++) r.push_back((T)(a[i] * a[i] * a[i] * a[i])); return r; }
    for (size_t i = 0; i < a.size(); i++) r.push_back(scalar::pow(a[i], n));
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
