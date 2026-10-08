#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// FP8<E,M> — 8-bit float, layout [sign | E exponent | M mantissa] MSB-first.
//
// [DECISION — do not change, only documented]
//   * E4M3 here is IEEE-style: exponent field of all-ones encodes inf / NaN
//     and the max finite value is 240. This is NOT the OCP MX E4M3 format
//     (no inf/NaN, max finite 448).
//   * There are no subnormals: anything below 2^(1-bias) flushes to signed 0.
//   * Rounding is half-away-from-zero (std::round semantics).
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdint>
#include <cmath>
#include <iostream>
#include <ostream>
#include <limits>
#include <compare>

template <int E, int M>
struct FP8 {
    static_assert(E + M + 1 == 8, "E + M + 1 must equal 8 bits for FP8");
    static_assert(E >= 1 && M >= 0, "FP8 requires E >= 1, M >= 0");

    static constexpr int  exp_bits   = E;
    static constexpr int  mant_bits  = M;
    static constexpr int  bias       = (1 << (E - 1)) - 1;
    static constexpr int  max_exp    = (1 << E) - 1;
    static constexpr int  mant_scale = (1 << M);

    uint8_t bits = 0;   // I5: zero-initialised despite the comment on the old header

    FP8() = default;
    FP8(float f);
    explicit operator float() const;

    // I6: bit accessors for loaders.
    static FP8 from_bits(uint8_t b) { FP8 r; r.bits = b; return r; }
    uint8_t   to_bits() const       { return bits; }

    FP8  operator-() const { return FP8(-float(*this)); }

    FP8& operator+=(const FP8& o) { *this = *this + o; return *this; }
    FP8& operator-=(const FP8& o) { *this = *this - o; return *this; }
    FP8& operator*=(const FP8& o) { *this = *this * o; return *this; }
    FP8& operator/=(const FP8& o) { *this = *this / o; return *this; }

    FP8 operator+(const FP8& o) const { return FP8(float(*this) + float(o)); }
    FP8 operator-(const FP8& o) const { return FP8(float(*this) - float(o)); }
    FP8 operator*(const FP8& o) const { return FP8(float(*this) * float(o)); }
    FP8 operator/(const FP8& o) const { return FP8(float(*this) / float(o)); }

    bool operator==(const FP8& o) const { return float(*this) == float(o); }
    bool operator!=(const FP8& o) const { return !(*this == o); }
    bool operator< (const FP8& o) const { return float(*this) <  float(o); }

    std::partial_ordering operator<=>(const FP8& o) const {
        float a = float(*this), b = float(o);
        if (a < b) return std::partial_ordering::less;
        if (a > b) return std::partial_ordering::greater;
        if (a == b) return std::partial_ordering::equivalent;
        return std::partial_ordering::unordered;
    }
};

// ── encode / decode ──────────────────────────────────────────────────────────
// I10: frexp/ldexp instead of log2/pow. Bit-identical to the previous version
// on every finite input (see support_tests.cpp sweep).

template <int E, int M>
FP8<E, M>::FP8(float f) {
    if (f == 0.0f)     { bits = std::signbit(f) ? 0x80 : 0x00; return; ;    return; }
    if (std::isnan(f)) { bits = 0xFF; return; }

    uint8_t sign = (f < 0.0f) ? 1 : 0;
    f = std::fabs(f);

    if (std::isinf(f)) {
        bits = (uint8_t)((sign << 7) | (max_exp << M));   // exp all 1s, mant 0
        return;
    }

    int   e;
    float m   = std::frexp(f, &e);   // f = m * 2^e, 0.5 <= m < 1
    int   exp = e - 1;               // f = sig * 2^exp, 1 <= sig < 2
    float sig = 2.0f * m;

    int biased_exp = exp + bias;

    if (biased_exp <= 0) { bits = (uint8_t)(sign << 7); return; }   // underflow → signed 0
    if (biased_exp >= max_exp) {                                     // overflow → max finite
        bits = (uint8_t)((sign << 7) | ((max_exp - 1) << M) | ((1 << M) - 1));
        return;
    }

    float mantissa = sig - 1.0f;
    int   mant_val = (int)std::round(mantissa * mant_scale);

    if (mant_val >= (1 << M)) {          // rounded up into the next exponent
        mant_val = 0;
        biased_exp += 1;
        if (biased_exp >= max_exp) {
            bits = (uint8_t)((sign << 7) | ((max_exp - 1) << M) | ((1 << M) - 1));
            return;
        }
    }

    bits = (uint8_t)((sign << 7) | (biased_exp << M) | mant_val);
}

template <int E, int M>
FP8<E, M>::operator float() const {
    uint8_t sign       = (bits >> 7) & 0x1;
    uint8_t exp_bits_  = (bits >> M) & ((1 << E) - 1);
    uint8_t mant_bits_ =  bits       & ((1 << M) - 1);

    if (exp_bits_ == 0) return sign ? -0.0f : 0.0f;
    if (exp_bits_ == max_exp)
        return mant_bits_ ? NAN : (sign ? -INFINITY : INFINITY);

    float value = (1.0f + (float)mant_bits_ / mant_scale)
                * std::ldexp(1.0f, (int)exp_bits_ - bias);
    return sign ? -value : value;
}

template <int E, int M>
std::ostream& operator<<(std::ostream& os, const FP8<E, M>& v) {
    os << float(v);
    return os;
}

// ── I7: proper numeric_limits specialisation (class, not struct) ────────────
namespace std {

template <int E, int M>
class numeric_limits<FP8<E, M>> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed      = true;
    static constexpr bool has_infinity   = true;
    static constexpr bool has_quiet_NaN  = true;
    static constexpr int  digits         = M + 1;   // significand bits incl. implicit 1

    static FP8<E, M> infinity() noexcept {
        FP8<E, M> f; f.bits = (uint8_t)(((1 << E) - 1) << M);
        return f;
    }
    static FP8<E, M> quiet_NaN() noexcept {
        FP8<E, M> f; f.bits = 0xFF;
        return f;
    }
    static FP8<E, M> max() noexcept {
        FP8<E, M> f;
        f.bits = (uint8_t)(((FP8<E, M>::max_exp - 1) << M) | ((1 << M) - 1));
        return f;
    }
    static FP8<E, M> lowest() noexcept {
        FP8<E, M> f = max(); f.bits |= 0x80; return f;
    }
    static FP8<E, M> min() noexcept {
        FP8<E, M> f; f.bits = (uint8_t)(1 << M);   // biased exp = 1, mant = 0
        return f;
    }
    static FP8<E, M> epsilon() noexcept {
        int e = FP8<E, M>::bias - M;   // biased exponent of 2^-M
        FP8<E, M> f;
        f.bits = (e <= 0) ? 0 : (uint8_t)(e << M);
        return f;
    }
};

} // namespace std