#pragma once

#include <ctype.h>
#include <stdint.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <cmath>

template<int E, int M>
struct FP8 {

    static_assert(E + M + 1 == 8, "E + M + 1 must equal 8 bits for FP8");

    uint8_t bits;  // raw storage, nothing more

    static constexpr int  exp_bits  = E;
    static constexpr int  mant_bits = M;
    static constexpr int  bias      = (1 << (E-1)) - 1;
    static constexpr int  max_exp   = (1 << E) - 1;
    static constexpr int  mant_scale = (1 << M);

    FP8() = default;                  // zero-init
    FP8(float f);                     // float → FP8  (encoding)
    explicit operator float() const;  // FP8 → float  (decoding)

    FP8 operator+(const FP8& o) const { return FP8(float(*this) + float(o)); }
    FP8 operator-(const FP8& o) const { return FP8(float(*this) - float(o)); }
    FP8 operator*(const FP8& o) const { return FP8(float(*this) * float(o)); }
    FP8 operator/(const FP8& o) const { return FP8(float(*this) / float(o)); }

    bool operator==(const FP8& o) const { return float(*this) == float(o); }
    bool operator< (const FP8& o) const { return float(*this) <  float(o); }

};

// ── FP8<E,M> encode / decode ────────────────────────────────────────────────
// Layout matches FP4's scheme: sign bit,
// then E exponent bits, then M mantissa bits, MSB-first.

template<int E, int M>
FP8<E,M>::FP8(float f) {
    if (f == 0.0f) { bits = 0; return; }
    if (std::isnan(f)) { bits = 0xFF; return; }

    uint8_t sign = (f < 0.0f) ? 1 : 0;
    f = std::fabs(f);

    if (std::isinf(f)) {
        bits = (uint8_t)((sign << 7) | (max_exp << M)); // exp=all 1s, mant=0 → inf
        return;
    }

    int exp        = (int)std::floor(std::log2(f));
    int biased_exp = exp + bias;

    if (biased_exp <= 0) { bits = (uint8_t)(sign << 7); return; }   // underflow → (signed) zero
    if (biased_exp >= max_exp) {                                     // overflow → max finite
        bits = (uint8_t)((sign << 7) | ((max_exp - 1) << M) | ((1 << M) - 1));
        return;
    }

    float mantissa = f / std::pow(2.0f, exp) - 1.0f;   // fractional part
    int   mant_val = (int)std::round(mantissa * mant_scale);

    if (mant_val >= (1 << M)) {        // rounded up into the next exponent
        mant_val = 0;
        biased_exp += 1;
        if (biased_exp >= max_exp) {
            bits = (uint8_t)((sign << 7) | ((max_exp - 1) << M) | ((1 << M) - 1));
            return;
        }
    }

    bits = (uint8_t)((sign << 7) | (biased_exp << M) | mant_val);
}

template<int E, int M>
FP8<E,M>::operator float() const {
    uint8_t sign      = (bits >> 7) & 0x1;
    uint8_t exp_bits_ = (bits >> M) & ((1 << E) - 1);
    uint8_t mant_bits_=  bits       & ((1 << M) - 1);

    if (exp_bits_ == 0) return sign ? -0.0f : 0.0f;               // zero
    if (exp_bits_ == max_exp)                                      // inf / nan
        return mant_bits_ ? NAN : (sign ? -INFINITY : INFINITY);

    float value = (1.0f + (float)mant_bits_ / mant_scale)
                * std::pow(2.0f, (int)exp_bits_ - bias);
    return sign ? -value : value;
}

template<int E, int M>
std::ostream& operator<<(std::ostream& os, const FP8<E,M>& v) {
        os << float(v);
        return os;
    }

template<int E, int M>
struct std::numeric_limits<FP8<E,M>> {
    static FP8<E,M> infinity() { 
        FP8<E,M> f; 
        f.bits = (1 << M) * ((1 << E) - 1); // all exp bits set, mant=0, sign=0
        return f; 
    }
    static FP8<E,M> quiet_NaN() {
        FP8<E,M> f;
        f.bits = 0xFF; // all bits set
        return f;
    }
};