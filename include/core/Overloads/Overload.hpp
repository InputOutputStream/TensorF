#pragma once

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

#include "core/Types/scalar.hpp"
#include "VectorMath.hpp"

// Backwards-compatible names (the comparison helpers used to live in namespace `detail`).
namespace detail {
using vecmath::cmp_lt; using vecmath::cmp_le; using vecmath::cmp_gt; using vecmath::cmp_ge;
template <typename T> std::vector<T> as_t(const std::vector<uint8_t>& m) { std::vector<T> r; r.reserve(m.size()); for (uint8_t v : m) r.push_back((T)v); return r; }
}

// ─── vector (op) vector ──────────────────────────────────────────────────────
template <vecmath::Element T> std::vector<T> operator+(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::add(a, b); }
template <vecmath::Element T> std::vector<T> operator-(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::sub(a, b); }
template <vecmath::Element T> std::vector<T> operator*(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::mul(a, b); }
template <vecmath::Element T> std::vector<T> operator/(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::div(a, b); }

// ─── vector (op) scalar and scalar (op) vector (scalar parameter non-deduced) ───
template <vecmath::Element T> std::vector<T> operator+(const std::vector<T>& b, std::type_identity_t<T> a) { return vecmath::add_s(b, a); }
template <vecmath::Element T> std::vector<T> operator+(std::type_identity_t<T> a, const std::vector<T>& b) { return vecmath::add_s(b, a); }
template <vecmath::Element T> std::vector<T> operator*(const std::vector<T>& b, std::type_identity_t<T> a) { return vecmath::mul_s(b, a); }
template <vecmath::Element T> std::vector<T> operator*(std::type_identity_t<T> a, const std::vector<T>& b) { return vecmath::mul_s(b, a); }
template <vecmath::Element T> std::vector<T> operator-(const std::vector<T>& b, std::type_identity_t<T> a) { return vecmath::sub_s(b, a); }
template <vecmath::Element T> std::vector<T> operator-(std::type_identity_t<T> a, const std::vector<T>& b) { return vecmath::s_sub(a, b); }
template <vecmath::Element T> std::vector<T> operator/(const std::vector<T>& b, std::type_identity_t<T> a) { return vecmath::div_s(b, a); }
template <vecmath::Element T> std::vector<T> operator/(std::type_identity_t<T> a, const std::vector<T>& b) { return vecmath::s_div(a, b); }
template <vecmath::Element T> std::vector<T> operator-(const std::vector<T>& a) { return vecmath::neg(a); }
template <vecmath::Element T> std::vector<T> operator%(const std::vector<T>& b, std::type_identity_t<T> a) { return vecmath::mod_s(b, a); }
template <vecmath::Element T> std::vector<T> operator%(std::type_identity_t<T> a, const std::vector<T>& b) { return vecmath::s_mod(a, b); }

// ─── comparisons returning std::vector<T> (kept for API compat; Matrix uses the uint8_t versions) ───
template <vecmath::Element T> std::vector<T> operator<(const std::vector<T>& b, std::type_identity_t<T> a)  { return detail::as_t<T>(vecmath::cmp_lt(b, a)); }
template <vecmath::Element T> std::vector<T> operator<(std::type_identity_t<T> a, const std::vector<T>& b)  { return detail::as_t<T>(vecmath::cmp_lt(a, b)); }
template <vecmath::Element T> std::vector<T> operator>(const std::vector<T>& b, std::type_identity_t<T> a)  { return detail::as_t<T>(vecmath::cmp_gt(b, a)); }
template <vecmath::Element T> std::vector<T> operator>(std::type_identity_t<T> a, const std::vector<T>& b)  { return detail::as_t<T>(vecmath::cmp_gt(a, b)); }
template <vecmath::Element T> std::vector<T> operator<=(const std::vector<T>& b, std::type_identity_t<T> a) { return detail::as_t<T>(vecmath::cmp_le(b, a)); }
template <vecmath::Element T> std::vector<T> operator<=(std::type_identity_t<T> a, const std::vector<T>& b) { return detail::as_t<T>(vecmath::cmp_le(a, b)); }
template <vecmath::Element T> std::vector<T> operator>=(const std::vector<T>& b, std::type_identity_t<T> a) { return detail::as_t<T>(vecmath::cmp_ge(b, a)); }
template <vecmath::Element T> std::vector<T> operator>=(std::type_identity_t<T> a, const std::vector<T>& b) { return detail::as_t<T>(vecmath::cmp_ge(a, b)); }

// ─── equality ────────────────────────────────────────────────────────────────
template <vecmath::Element T> bool operator==(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::equal(a, b); }

template <vecmath::Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> operator==(const std::vector<T>& a, const U& b) { return vecmath::eq_s(a, b); }
template <vecmath::Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> operator==(const U& b, const std::vector<T>& a) { return vecmath::eq_s(a, b); }
template <vecmath::Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> operator!=(const std::vector<T>& a, const U& b) { return vecmath::ne_s(a, b); }
template <vecmath::Element T, typename U> requires std::is_arithmetic_v<U>
std::vector<uint8_t> operator!=(const U& b, const std::vector<T>& a) { return vecmath::ne_s(a, b); }

// ─── compound assignment ─────────────────────────────────────────────────────
template <vecmath::Element T> std::vector<T>& operator+=(std::vector<T>& a, std::type_identity_t<T> b) { vecmath::add_inplace_s(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator-=(std::vector<T>& a, std::type_identity_t<T> b) { vecmath::sub_inplace_s(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator*=(std::vector<T>& a, std::type_identity_t<T> b) { vecmath::mul_inplace_s(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator/=(std::vector<T>& a, std::type_identity_t<T> b) { vecmath::div_inplace_s(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b) { vecmath::add_inplace(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator-=(std::vector<T>& a, const std::vector<T>& b) { vecmath::sub_inplace(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator*=(std::vector<T>& a, const std::vector<T>& b) { vecmath::mul_inplace(a, b); return a; }
template <vecmath::Element T> std::vector<T>& operator/=(std::vector<T>& a, const std::vector<T>& b) { vecmath::div_inplace(a, b); return a; }

// ─── global helper functions (same names as before; constrained) ──────────────
template <vecmath::Element T> std::vector<T> exponent(const std::vector<T>& a) { return vecmath::exponent(a); }
template <vecmath::Element T> std::vector<T> pow(const std::vector<T>& a, std::type_identity_t<T> n) { return vecmath::pow_s(a, n); }
template <vecmath::Element T> std::vector<T> pow(const std::vector<T>& a, const std::vector<T>& b) { return vecmath::pow_v(a, b); }
template <vecmath::Element T> bool has_nan_or_inf(const std::vector<T>& v) { return vecmath::has_nan_or_inf(v); }

template <vecmath::Element T>
void check_nan(const std::vector<T>& v) {
    if (vecmath::has_nan_or_inf(v)) {
        std::cerr << "Invalid value (nan or inf) in vector\n";
        std::abort();
    }
}

// ─── stream operator ─────────────────────────────────────────────────────────
// Numeric element types only. uint8_t/int8_t print as numbers; size_t prints with "(...)"; every element is followed by a comma.
template <vecmath::Element T>
std::ostream& operator<<(std::ostream& out, const std::vector<T>& a) {
    char open = '[', close = ']';
    if constexpr (std::is_same_v<T, std::size_t>) { open = '('; close = ')'; }

    out << open;
    for (const auto& v : a) {
        if constexpr (std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
                      std::is_same_v<T, signed char> || std::is_same_v<T, unsigned char>) {
            out << static_cast<int>(v) << ",";
        } else if constexpr (std::is_same_v<T, float16>) {
            out << static_cast<float>(v) << ",";      // ostream << _Float16 is ambiguous on GCC 13
        } else {
            out << v << ",";
        }
    }
    out << close;
    return out;
}
