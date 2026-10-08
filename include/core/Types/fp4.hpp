#pragma once

// ─────────────────────────────────────────────────────────────────────────────
// FP4<E,M> — 4-bit float, layout [sign | E exponent | M mantissa] MSB-first.
//
// [DECISION — do not change, only documented]
//   * E2M1 (fp4_e2m1) max finite is 3.0, NOT 6.0 (no subnormals, no bias trick).
//   * No subnormals; anything below 2^(1-bias) flushes to signed 0.
//   * Rounding is half-away-from-zero.
//
// `uint4_t` is retained for ABI compatibility only; no code path uses it.
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <iostream>
#include <limits>
#include <compare>

struct uint4_t { uint8_t val : 4; };  // unused; kept for external users

template <unsigned short E, unsigned short M>
struct FP4 {
    static_assert(E + M + 1 == 4, "E + M + 1 must equal 4 bits");
    // I1: FP4<1,2> is degenerate (bias 0, max_exp 1 → every bit pattern decodes to 0).
    static_assert(E >= 2,
        "FP4<E,M> requires E >= 2: FP4<1,2> is degenerate (bias 0, max_exp 1). "
        "fp4_e1m2 stays declared but fails when instantiated.");

    static constexpr int bias       = (1 << (E - 1)) - 1;
    static constexpr int max_exp    = (1 << E) - 1;
    static constexpr int mant_scale = (1 << M);

    uint8_t bits = 0;

    FP4() = default;
    FP4(float f);
    explicit operator float() const;

    static FP4 from_bits(uint8_t b) { FP4 r; r.bits = (uint8_t)(b & 0x0F); return r; }
    uint8_t   to_bits() const       { return (uint8_t)(bits & 0x0F); }

    FP4  operator-() const { return FP4(-float(*this)); }

    FP4& operator+=(const FP4& o) { *this = *this + o; return *this; }
    FP4& operator-=(const FP4& o) { *this = *this - o; return *this; }
    FP4& operator*=(const FP4& o) { *this = *this * o; return *this; }
    FP4& operator/=(const FP4& o) { *this = *this / o; return *this; }

    FP4 operator+(const FP4& o) const { return FP4(float(*this) + float(o)); }
    FP4 operator-(const FP4& o) const { return FP4(float(*this) - float(o)); }
    FP4 operator*(const FP4& o) const { return FP4(float(*this) * float(o)); }
    FP4 operator/(const FP4& o) const { return FP4(float(*this) / float(o)); }

    bool operator==(const FP4& o) const { return float(*this) == float(o); }
    bool operator!=(const FP4& o) const { return !(*this == o); }
    bool operator< (const FP4& o) const { return float(*this) <  float(o); }

    std::partial_ordering operator<=>(const FP4& o) const {
        float a = float(*this), b = float(o);
        if (a < b) return std::partial_ordering::less;
        if (a > b) return std::partial_ordering::greater;
        if (a == b) return std::partial_ordering::equivalent;
        return std::partial_ordering::unordered;
    }
};

// ── encode ───────────────────────────────────────────────────────────────────
template <unsigned short E, unsigned short M>
FP4<E, M>::FP4(float f) {
    // I3: handle inf / NaN before any cast (log2 -> int was UB).
    if (std::isnan(f)) {
        bits = (uint8_t)(((max_exp - 1) << M) | ((1 << M) - 1));  // NaN → max finite
        return;
    }
    if (std::isinf(f)) {
        uint8_t s = (f < 0.0f) ? 1 : 0;
        bits = (uint8_t)((s << 3) | (max_exp << M));             // inf encoding
        return;
    }

    // I4: preserve the sign of zero.
    if (f == 0.0f) {
        bits = std::signbit(f) ? 0x08 : 0x00;
        return;
    }

    uint8_t sign = (f < 0.0f) ? 1 : 0;
    f = std::fabs(f);

    // I10: frexp / ldexp.
    int   e;
    float m   = std::frexp(f, &e);
    int   exp = e - 1;
    float sig = 2.0f * m;
    int   biased_exp = exp + bias;

    if (biased_exp <= 0) { bits = (uint8_t)(sign << 3); return; }   // underflow
    if (biased_exp >= max_exp) {                                     // overflow → max finite
        bits = (uint8_t)((sign << 3) | ((max_exp - 1) << M) | ((1 << M) - 1));
        return;
    }

    float mantissa = sig - 1.0f;
    int   mant_bits = (int)std::round(mantissa * mant_scale);

    // I2: carry into the exponent on mantissa overflow (1.9f in e2m1 → 2.0f).
    if (mant_bits >= (1 << M)) {
        mant_bits = 0;
        biased_exp += 1;
        if (biased_exp >= max_exp) {
            bits = (uint8_t)((sign << 3) | ((max_exp - 1) << M) | ((1 << M) - 1));
            return;
        }
    }

    bits = (uint8_t)((sign << 3) | (biased_exp << M) | mant_bits);
}

// ── decode ───────────────────────────────────────────────────────────────────
template <unsigned short E, unsigned short M>
FP4<E, M>::operator float() const {
    uint8_t sign       = (bits >> 3) & 0x1;
    uint8_t exp_bits_  = (bits >> M) & ((1 << E) - 1);
    uint8_t mant_bits_ =  bits       & ((1 << M) - 1);

    if (exp_bits_ == 0) return sign ? -0.0f : 0.0f;   // I4: keep sign of zero
    if (exp_bits_ == max_exp)
        return mant_bits_ ? NAN : (sign ? -INFINITY : INFINITY);

    float value = (1.0f + (float)mant_bits_ / mant_scale)
                * std::ldexp(1.0f, (int)exp_bits_ - bias);
    return sign ? -value : value;
}

template <unsigned short E, unsigned short M>
std::ostream& operator<<(std::ostream& os, const FP4<E, M>& v) {
    os << float(v);
    return os;
}

// ── I7: numeric_limits ───────────────────────────────────────────────────────
namespace std {

template <unsigned short E, unsigned short M>
class numeric_limits<FP4<E, M>> {
public:
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed      = true;
    static constexpr bool has_infinity   = true;
    static constexpr bool has_quiet_NaN  = true;
    static constexpr int  digits         = M + 1;

    static FP4<E, M> infinity() noexcept {
        FP4<E, M> f; f.bits = (uint8_t)(((1 << E) - 1) << M);
        return f;
    }
    static FP4<E, M> quiet_NaN() noexcept {
        FP4<E, M> f; f.bits = (uint8_t)((((1 << E) - 1) << M) | 1);
        return f;
    }
    static FP4<E, M> max() noexcept {
        FP4<E, M> f;
        f.bits = (uint8_t)(((FP4<E, M>::max_exp - 1) << M) | ((1 << M) - 1));
        return f;
    }
    static FP4<E, M> lowest() noexcept {
        FP4<E, M> f = max(); f.bits |= 0x08; return f;
    }
    static FP4<E, M> min() noexcept {
        FP4<E, M> f; f.bits = (uint8_t)(1 << M);
        return f;
    }
    static FP4<E, M> epsilon() noexcept {
        int e = FP4<E, M>::bias - M;
        FP4<E, M> f;
        f.bits = (e <= 0) ? 0 : (uint8_t)(e << M);
        return f;
    }
};

} // namespace std