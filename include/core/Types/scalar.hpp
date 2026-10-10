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
#include <limits>

#include "fp8.hpp"
#include "fp4.hpp"
#include "shape.hpp"

// ─── float16 ─────────────────────────────────────────────────────────────────
// `float16` is the compiler's native _Float16 when available, otherwise our own software binary16 (soft_float16,
// Types/float16.hpp) with identical numerics. Either way Matrix<float16>, Tensor<float16> and the accumulators work.
//   TENSORF_FORCE_SOFT_FLOAT16   use the software type even if _Float16 exists (testing / bit-exact across targets)
#include "float16.hpp"

// libstdc++ without <stdfloat> ships NO numeric_limits for the builtin _Float16: every member silently
// returned 0 (max()==0, infinity()==0, quiet_NaN()==0). Provide the IEEE binary16 values ourselves.
#if defined(__FLT16_MAX__) && !defined(__STDCPP_FLOAT16_T__)
namespace std {
template <> class numeric_limits<_Float16> {
public:
    static constexpr bool is_specialized = true, is_signed = true, is_integer = false, is_exact = false,
                          has_infinity = true, has_quiet_NaN = true, has_signaling_NaN = true,
                          is_iec559 = true, is_bounded = true, is_modulo = false;
    static constexpr int  digits = 11, digits10 = 3, max_digits10 = 5, radix = 2,
                          min_exponent = -13, max_exponent = 16, min_exponent10 = -4, max_exponent10 = 4;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static constexpr _Float16 min() noexcept           { return __FLT16_MIN__; }
    static constexpr _Float16 max() noexcept           { return __FLT16_MAX__; }
    static constexpr _Float16 lowest() noexcept        { return -__FLT16_MAX__; }
    static constexpr _Float16 epsilon() noexcept       { return __FLT16_EPSILON__; }
    static constexpr _Float16 round_error() noexcept   { return (_Float16)0.5f; }
    static constexpr _Float16 infinity() noexcept      { return __builtin_inff16(); }
    static constexpr _Float16 quiet_NaN() noexcept     { return __builtin_nanf16(""); }
    static constexpr _Float16 signaling_NaN() noexcept { return __builtin_nansf16(""); }
    static constexpr _Float16 denorm_min() noexcept    { return __FLT16_DENORM_MIN__; }
};
}
#endif

#if defined(__FLT16_MAX__) && !defined(TENSORF_FORCE_SOFT_FLOAT16)
    using float16 = _Float16;
    inline constexpr bool float16_is_native = true;

#else
    using float16 = soft_float16;
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
    using type = T;
};

template <int E, int M> struct acc<FP8<E, M>> { using type = float; };
template <unsigned short E, unsigned short M> struct acc<FP4<E, M>> { using type = float; };

template <> struct acc<float16> { using type = float; };   // native or software

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