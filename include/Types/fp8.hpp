#pragma once

#include <ctype.h>
#include <stdint.h>
#include <iostream>
#include <ostream>
#include <fstream>
#include <cmath>

template<int E, int M>
struct FP8 {

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

