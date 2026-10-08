#pragma once
// Types/scalar.hpp — element traits + accumulator type + math wrappers.
//
// Every math call that must work for float, double, integers, FP8, FP4 and
// float16 goes through `scalar::` here so that low-precision types are
// promoted to `float` for the operation and demoted back once at the end.
//
// Include this header instead of the whole Types/types.hpp when you only
// need shape_t + scalar traits (Matrix / Broadcast do exactly that).

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "fp8.hpp"
#include "fp4.hpp"
#include "shape.hpp"

// ─── float16 ─────────────────────────────────────────────────────────────────
#if defined(__FLT16_MAX__)
    using float16 = _Float16;
    inline constexpr bool float16_is_native = true;
#else
    // No native _Float16 on this compiler. float16 becomes a DISTINCT storage-only type (it used to be a
    // uint16_t alias, which silently did integer arithmetic and was indistinguishable from uint16_t).
    // The header still parses and every other type keeps working; only an attempt to USE float16 as a
    // scalar (e.g. Matrix<float16>) fails, at the point of use, with the message in acc<> below.
    struct float16 { std::uint16_t bits; };
    inline constexpr bool float16_is_native = false;
#endif

namespace scalar {

// ─── traits ──────────────────────────────────────────────────────────────────
template <typename T> inline constexpr bool is_fp8_v = false;
template <int E, int M> inline constexpr bool is_fp8_v<FP8<E, M>> = true;

template <typename T> inline constexpr bool is_fp4_v = false;
template <unsigned short E, unsigned short M> inline constexpr bool is_fp4_v<FP4<E, M>> = true;

template <typename T> inline constexpr bool is_low_precision_v =
    is_fp8_v<T> || is_fp4_v<T> || std::is_same_v<T, float16>;

// ─── accumulator type ────────────────────────────────────────────────────────
// acc<T>::type is the type used for INTERMEDIATE values (sums, dot products, math functions) of element type T.
// Low-precision types accumulate in float; everything else keeps its own type (float stays float, double
// stays double, ints stay ints), so existing float/double/int results are unchanged.
template <typename T> struct acc {
    // Checked when acc<T> is instantiated (i.e. on first real use), never when the header is parsed.
    static_assert(float16_is_native || !std::is_same_v<T, float16>,
        "float16 is not native on this compiler (no _Float16): it is storage-only here and cannot be used "
        "as a scalar type. Use float, or a compiler/flags that provide _Float16.");
    using type = T;
};

template <int E, int M> struct acc<FP8<E, M>> { using type = float; };
template <unsigned short E, unsigned short M> struct acc<FP4<E, M>> { using type = float; };

#if defined(__FLT16_MAX__)
template <> struct acc<float16> { using type = float; };
#else
// non-native: leave acc<float16> to the primary template so its static_assert fires on use
#endif

template <typename T> using acc_t = typename acc<T>::type;

// ─── conversions ─────────────────────────────────────────────────────────────
template <typename T>
inline acc_t<T> to_acc(T v) {
    return static_cast<acc_t<T>>(v);
}

template <typename T>
inline T from_acc(acc_t<T> v) {
    return static_cast<T>(v);
}

// ─── math wrappers ───────────────────────────────────────────────────────────
template <typename T> inline T exp (T v) { return from_acc<T>(std::exp (to_acc<T>(v))); }
template <typename T> inline T log (T v) { return from_acc<T>(std::log (to_acc<T>(v))); }
template <typename T> inline T sqrt(T v) { return from_acc<T>(std::sqrt(to_acc<T>(v))); }
template <typename T> inline T cbrt(T v) { return from_acc<T>(std::cbrt(to_acc<T>(v))); }
template <typename T> inline T sin (T v) { return from_acc<T>(std::sin (to_acc<T>(v))); }
template <typename T> inline T cos (T v) { return from_acc<T>(std::cos (to_acc<T>(v))); }
template <typename T> inline T tan (T v) { return from_acc<T>(std::tan (to_acc<T>(v))); }

template <typename T> inline T pow(T a, T b) {
    return from_acc<T>(std::pow(to_acc<T>(a), to_acc<T>(b)));
}

template <typename T> inline T abs(T v) {
    using A = acc_t<T>;
    A a = to_acc<T>(v);
    if constexpr (std::is_unsigned_v<A>) return from_acc<T>(a);
    else                                return from_acc<T>(std::abs(a));
}

template <typename T> inline bool isnan(T v) { return std::isnan(to_acc<T>(v)); }
template <typename T> inline bool isinf(T v) { return std::isinf(to_acc<T>(v)); }

} // namespace scalar