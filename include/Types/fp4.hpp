#include <cstdint>
#include <cmath>
#include <iostream>

#ifndef __FP4_HPP
#define __FP4_HPP

// Type 4 bits via champ de bits dans un uint8_t
struct uint4_t {
    uint8_t val : 4;  // seuls 4 bits utilisés
};

template<unsigned short E, unsigned short M>
struct FP4 {
    static_assert(E + M + 1 == 4, "E + M + 1 doit valoir 4 bits");
    // ex: E=2, M=1 → 1 signe + 2 exp + 1 mantisse = 4 bits

    static constexpr int bias       = (1 << (E - 1)) - 1;
    static constexpr int max_exp    = (1 << E) - 1;
    static constexpr int mant_scale = (1 << M);

    uint8_t bits : 4;  // ← stockage réel sur 4 bits

    FP4() : bits(0) {}

    FP4(float f) {
        if (f == 0.0f) { bits = 0; return; }

        uint8_t sign = (f < 0) ? 1 : 0;
        f = std::fabs(f);

        int exp = (int)std::floor(std::log2(f));
        int biased_exp = exp + bias;

        // Clamp exponent
        if (biased_exp <= 0) { bits = 0; return; }       // underflow → 0
        if (biased_exp >= max_exp) {                       // overflow → max
            bits = (sign << 3) | ((max_exp - 1) << M) | ((1 << M) - 1);
            return;
        }

        float mantissa = f / std::pow(2.0f, exp) - 1.0f; // partie fractionnaire
        int mant_bits  = (int)std::round(mantissa * mant_scale);
        mant_bits = std::min(mant_bits, (1 << M) - 1);

        bits = (sign << 3) | (biased_exp << M) | mant_bits;
    }

    explicit operator float() const {
        uint8_t sign      = (bits >> 3) & 0x1;
        uint8_t exp_bits  = (bits >> M) & ((1 << E) - 1);
        uint8_t mant_bits =  bits       & ((1 << M) - 1);

        if (exp_bits == 0)       return 0.0f;   // zéro / dénormalisé
        if (exp_bits == max_exp) return sign ? -INFINITY : INFINITY; // spécial

        float value = (1.0f + (float)mant_bits / mant_scale)
                    * std::pow(2.0f, (int)exp_bits - bias);
        return sign ? -value : value;
    }

    FP4 operator+(const FP4& o) const { return FP4(float(*this) + float(o)); }
    FP4 operator-(const FP4& o) const { return FP4(float(*this) - float(o)); }
    FP4 operator*(const FP4& o) const { return FP4(float(*this) * float(o)); }
    FP4 operator/(const FP4& o) const { return FP4(float(*this) / float(o)); }

    bool operator==(const FP4& o) const { return float(*this) == float(o); }
    bool operator< (const FP4& o) const { return float(*this) <  float(o); }
};

template<unsigned short E, unsigned short M>
std::ostream& operator<<(std::ostream& os, const FP4<E,M>& v) {
    os << float(v);
    return os;
}



#endif