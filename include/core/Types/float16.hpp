#pragma once
// Types/float16.hpp — software IEEE-754 binary16 ("half"): [1 sign | 5 exponent | 10 mantissa], bias 15.
//
// Always available as `soft_float16`. scalar.hpp aliases `float16` to the compiler's native _Float16 when there is
// one (and TENSORF_FORCE_SOFT_FLOAT16 is not defined), otherwise to this type. Arithmetic is done in float and
// rounded back once (same model as FP8 / FP4), conversions are round-to-nearest-even and handle subnormals,
// +-0, +-inf and NaN exactly like hardware _Float16 (verified bit-for-bit against _Float16 in lowprec_tests).

#include <compare>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iostream>
#include <limits>

struct soft_float16 {
    uint16_t bits = 0;

    soft_float16() = default;
    soft_float16(float f) : bits(encode(f)) {}
    explicit operator float() const { return decode(bits); }

    static soft_float16 from_bits(uint16_t b) { soft_float16 r; r.bits = b; return r; }
    uint16_t            to_bits() const       { return bits; }

    soft_float16 operator-() const { return from_bits((uint16_t)(bits ^ 0x8000u)); }   // exact, also for NaN/0

    soft_float16& operator+=(const soft_float16& o) { *this = *this + o; return *this; }
    soft_float16& operator-=(const soft_float16& o) { *this = *this - o; return *this; }
    soft_float16& operator*=(const soft_float16& o) { *this = *this * o; return *this; }
    soft_float16& operator/=(const soft_float16& o) { *this = *this / o; return *this; }

    soft_float16 operator+(const soft_float16& o) const { return soft_float16(float(*this) + float(o)); }
    soft_float16 operator-(const soft_float16& o) const { return soft_float16(float(*this) - float(o)); }
    soft_float16 operator*(const soft_float16& o) const { return soft_float16(float(*this) * float(o)); }
    soft_float16 operator/(const soft_float16& o) const { return soft_float16(float(*this) / float(o)); }

    bool operator==(const soft_float16& o) const { return float(*this) == float(o); }
    bool operator!=(const soft_float16& o) const { return !(*this == o); }
    bool operator< (const soft_float16& o) const { return float(*this) <  float(o); }

    std::partial_ordering operator<=>(const soft_float16& o) const {
        float a = float(*this), b = float(o);
        if (a < b) return std::partial_ordering::less;
        if (a > b) return std::partial_ordering::greater;
        if (a == b) return std::partial_ordering::equivalent;
        return std::partial_ordering::unordered;
    }

    // float -> half, round to nearest even
    static uint16_t encode(float f) {
        uint32_t x; std::memcpy(&x, &f, 4);
        const uint32_t sign = (x >> 16) & 0x8000u;
        const uint32_t e    = (x >> 23) & 0xFFu;
        uint32_t       m    = x & 0x7FFFFFu;
        if (e == 0xFFu)                                              // inf / NaN (NaN stays NaN: quiet bit forced)
            return (uint16_t)(sign | 0x7C00u | (m ? (0x200u | (m >> 13)) : 0u));
        const int exp = (int)e - 127 + 15;
        if (exp >= 31) return (uint16_t)(sign | 0x7C00u);            // overflow -> inf
        if (exp <= 0) {                                              // half subnormal or zero
            if (exp < -10) return (uint16_t)sign;
            m |= 0x800000u;                                          // make the implicit 1 explicit
            const int shift = 14 - exp;                              // 14..24
            uint32_t h = m >> shift;
            const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
            if (rem > half || (rem == half && (h & 1u))) h++;        // may carry into the smallest normal: still correct
            return (uint16_t)(sign | h);
        }
        uint32_t h = ((uint32_t)exp << 10) | (m >> 13);
        const uint32_t rem = m & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;      // carry into the exponent (or to inf) is correct
        return (uint16_t)(sign | h);
    }

    static float decode(uint16_t h) {
        const uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
        const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
        uint32_t out;
        if (e == 0) {
            if (m == 0) out = sign;                                  // +-0
            else {                                                   // subnormal: m * 2^-24 (exact in float)
                float v = std::ldexp((float)m, -24);
                std::memcpy(&out, &v, 4); out |= sign;
            }
        } else if (e == 31) out = sign | 0x7F800000u | (m << 13);   // inf / NaN
        else out = sign | ((e + 112u) << 23) | (m << 13);
        float r; std::memcpy(&r, &out, 4); return r;
    }
};

inline std::ostream& operator<<(std::ostream& os, const soft_float16& v) { return os << float(v); }

namespace std {
template <> class numeric_limits<soft_float16> {
public:
    static constexpr bool is_specialized = true, is_signed = true, is_integer = false, is_exact = false,
                          has_infinity = true, has_quiet_NaN = true, has_signaling_NaN = false,
                          is_iec559 = true, is_bounded = true, is_modulo = false;
    static constexpr int  digits = 11, digits10 = 3, max_digits10 = 5, radix = 2,
                          min_exponent = -13, max_exponent = 16, min_exponent10 = -4, max_exponent10 = 4;
    static constexpr float_denorm_style has_denorm = denorm_present;
    static soft_float16 min() noexcept          { return soft_float16::from_bits(0x0400); }   // 2^-14
    static soft_float16 max() noexcept          { return soft_float16::from_bits(0x7BFF); }   // 65504
    static soft_float16 lowest() noexcept       { return soft_float16::from_bits(0xFBFF); }
    static soft_float16 epsilon() noexcept      { return soft_float16::from_bits(0x1400); }   // 2^-10
    static soft_float16 round_error() noexcept  { return soft_float16::from_bits(0x3800); }   // 0.5
    static soft_float16 infinity() noexcept     { return soft_float16::from_bits(0x7C00); }
    static soft_float16 quiet_NaN() noexcept    { return soft_float16::from_bits(0x7E00); }
    static soft_float16 signaling_NaN() noexcept{ return soft_float16::from_bits(0x7D00); }
    static soft_float16 denorm_min() noexcept   { return soft_float16::from_bits(0x0001); }   // 2^-24
};
}
