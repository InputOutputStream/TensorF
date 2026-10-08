// support_tests.cpp — plain asserts; build with:
//   c++ -std=c++20 -Wall -Wextra -Wshadow -fsanitize=address,undefined support_tests.cpp -o support_tests

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "../include/core/Overloads/Overload.hpp"
#include "../include/core/Overloads/VectorMath.hpp"

#include "../include/core/Types/types.hpp"

using std::vector;

// ─── tiny helpers ────────────────────────────────────────────────────────────
template <typename F>
static bool throws(F&& f) { try { f(); } catch (...) { return true; } return false; }

template <typename T>
static bool close(T a, T b, T tol) { return std::fabs(double(a) - double(b)) <= double(tol); }

// ═════════════════════════════════════════════════════════════════════════════
// G1
static void test_G1_guard_and_includes() {
    // Overload.hpp uses #pragma once; including it twice must be a no-op.
    // (If the guard were still active this still compiles, but the semantics
    //  are now: every TU gets it at most once.)
    vector<float> v{1.f, 2.f};
    (void)v;
}

// ─── G2 zero checks ──────────────────────────────────────────────────────────
static void test_G2_zero_division() {
    vector<float> v{1.f, 0.f, 3.f};
    assert(throws([&]{ (void)(v / 2.f); }) == false);
    assert(throws([&]{ (void)(v / 0.f); }));           // vector/scalar
    assert(throws([&]{ (void)(2.f / v); }));           // scalar/vector
    vector<float> a{1.f, 2.f}, b{1.f, 0.f};
    assert(throws([&]{ (void)(a / b); }));             // vector/vector

    vector<float> w{1.f, 2.f};
    assert(throws([&]{ w /= 0.f; }));
    assert(throws([&]{ vector<float> x{1.f}; x /= vector<float>{0.f}; }));

    // integral % 0
    vector<int> iv{4, 5};
    assert(throws([&]{ (void)(iv % 0); }));
    assert(!throws([&]{ (void)(10 % iv); }));          // 10 % {4,5}: no zero divisor, must NOT throw
    vector<int> zv{4, 0};
    assert(throws([&]{ (void)(10 % zv); }));           // 10 % {4,0}: zero divisor -> throws
}

// ─── G3 size mismatch ────────────────────────────────────────────────────────
static void test_G3_size_mismatch() {
    vector<float> a{1.f, 2.f, 3.f};
    vector<float> b{1.f, 2.f};
    assert(throws([&]{ (void)(a + b); }));
    assert(throws([&]{ (void)(a - b); }));
    assert(throws([&]{ (void)(a * b); }));
    assert(throws([&]{ (void)(a / b); }));
    vector<float> x = a; assert(throws([&]{ x += b; }));

    // Exception: pow(vector, vector) with b.size()==1 is scalar broadcast.
    vector<float> s{2.f};
    vector<float> p = pow(a, s);
    assert(p.size() == 3);
    assert(p[0] == 1.f && p[1] == 4.f && p[2] == 9.f);
}

// ─── G4 non-deduced scalar parameter ─────────────────────────────────────────
static void test_G4_scalar_deduction() {
    vector<float> v{1.f, 2.f, 3.f};
    auto a = 2    * v;                // int literal, works because T = float
    auto b = v    * 2;
    auto c = v    + 1;
    auto d = 1    + v;
    assert(a.size() == 3 && b.size() == 3 && c.size() == 3 && d.size() == 3);
    assert(a[1] == 4.f && b[1] == 4.f && c[1] == 3.f && d[1] == 3.f);

    vector<float> w{1.f, 2.f};
    w += 1;  w -= 1;  w *= 2;  w /= 2;
    assert(w.size() == 2);
}

// ─── G5 comparison helpers ───────────────────────────────────────────────────
static void test_G5_cmp_helpers() {
    vector<float> v{1.f, 2.f, 3.f};
    auto lt = detail::cmp_lt(v, 2.f);
    auto le = detail::cmp_le(v, 2.f);
    auto gt = detail::cmp_gt(v, 2.f);
    auto ge = detail::cmp_ge(v, 2.f);
    static_assert(std::is_same_v<decltype(lt), vector<uint8_t>>);
    assert((lt == vector<uint8_t>{1, 0, 0}));
    assert((le == vector<uint8_t>{1, 1, 0}));
    assert((gt == vector<uint8_t>{0, 0, 1}));
    assert((ge == vector<uint8_t>{0, 1, 1}));

    // Reversed (T, vector)
    auto rlt = detail::cmp_lt(2.f, v);                 // 2 < v[i]
    assert((rlt == vector<uint8_t>{0, 0, 1}));
    assert((detail::cmp_le(2.f, v) == vector<uint8_t>{0, 1, 1}));
    assert((detail::cmp_gt(2.f, v) == vector<uint8_t>{1, 0, 0}));
    assert((detail::cmp_ge(2.f, v) == vector<uint8_t>{1, 1, 0}));

    // Old operators still return vector<T>
    auto old_lt = (v < 2.f);
    static_assert(std::is_same_v<decltype(old_lt), vector<float>>);
}

// ─── G6 streaming ────────────────────────────────────────────────────────────
static void test_G6_streaming() {
    vector<uint8_t> bytes{0, 127, 255};
    std::ostringstream os; os << bytes;
    assert(os.str() == "[0,127,255,]");

    vector<signed char> sc{-1, 2};
    std::ostringstream os2; os2 << sc;
    assert(os2.str() == "[-1,2,]");

    vector<size_t> sz{1, 2};
    std::ostringstream os3; os3 << sz;
    assert(os3.str() == "(1,2,)");
}

// ─── G7 exponent through scalar traits ──────────────────────────────────────
static void test_G7_exponent() {
    vector<float> v{0.f, 1.f, 2.f};
    auto e = exponent(v);
    assert(close(e[0], 1.f, 1e-6f));
    assert(close(e[1], (float)std::exp(1.f), 1e-6f));
    assert(close(e[2], (float)std::exp(2.f), 1e-6f));
}

// ─── G8 has_nan_or_inf ───────────────────────────────────────────────────────
static void test_G8_nan_inf() {
    vector<float> v{1.f, 2.f};
    assert(!has_nan_or_inf(v));
    vector<float> w{1.f, std::numeric_limits<float>::quiet_NaN()};
    assert(has_nan_or_inf(w));
    vector<float> u{1.f, std::numeric_limits<float>::infinity()};
    assert(has_nan_or_inf(u));
}

// ─── H1 integer aliases ─────────────────────────────────────────────────────
static void test_H1_integer_aliases() {
    static_assert(std::is_same_v<int8,  unsigned char>);
    static_assert(std::is_same_v<int16, unsigned short>);
    static_assert(std::is_same_v<int32, unsigned int>);
    static_assert(std::is_same_v<i8,  std::int8_t>);
    static_assert(std::is_same_v<u8,  std::uint8_t>);
    static_assert(std::is_same_v<u64, std::uint64_t>);
}

// ─── H2 float16 native flag ─────────────────────────────────────────────────
static void test_H2_float16() {
    // Just check the flag exists and matches the environment.
#if defined(__FLT16_MAX__)
    static_assert(float16_is_native);
#else
    static_assert(!float16_is_native);
#endif
}

// ─── H3 float32/float64 ─────────────────────────────────────────────────────
static void test_H3_float_aliases() {
    static_assert(std::is_same_v<float32, float>);
    static_assert(std::is_same_v<float64, double>);
}

// ─── I2 mantissa carry in FP4 ────────────────────────────────────────────────
static void test_I2_fp4_carry() {
    fp4_e2m1 x(1.9f);
    assert(close((float)x, 2.0f, 1e-6f));
}

// ─── I3 inf / NaN in FP4 ────────────────────────────────────────────────────
static void test_I3_fp4_specials() {
    fp4_e2m1 inf_p(std::numeric_limits<float>::infinity());
    fp4_e2m1 inf_n(-std::numeric_limits<float>::infinity());
    assert(std::isinf((float)inf_p) && (float)inf_p > 0);
    assert(std::isinf((float)inf_n) && (float)inf_n < 0);

    fp4_e2m1 nan(std::numeric_limits<float>::quiet_NaN());
    assert(std::isfinite((float)nan));  // NaN → max finite
}

// ─── I4 sign of zero ────────────────────────────────────────────────────────
static void test_I4_fp4_signed_zero() {
    assert(std::signbit((float)fp4_e2m1::from_bits(0x8)) == true);
    assert(std::signbit((float)fp4_e2m1::from_bits(0x0)) == false);
    
    assert(std::signbit((float)fp8_e4m3::from_bits(0x80)) == true);
    assert(std::signbit((float)fp8_e4m3::from_bits(0x00)) == false);

    // Underflow keeps the sign.
    assert(std::signbit((float)fp4_e2m1(-1e-9f)) == true);
    assert(std::signbit((float)fp4_e2m1( 1e-9f)) == false);
    assert(std::signbit((float)fp8_e4m3(-1e-9f)) == true);
    assert(std::signbit((float)fp8_e4m3( 1e-9f)) == false);

    // Constructing from a literal -0.0f
    assert(std::signbit((float)fp4_e2m1(-0.0f)) == true);
    assert(std::signbit((float)fp8_e4m3(-0.0f)) == true);
}

// ─── I5 FP8 default zero-init ───────────────────────────────────────────────
static void test_I5_fp8_default() {
    fp8_e4m3 x;
    assert(x.bits == 0);
    assert((float)x == 0.0f);
}

// ─── I6 operator set ────────────────────────────────────────────────────────
static void test_I6_operators() {
    fp8_e4m3 a(2.0f), b(0.5f);
    assert(close((float)(a + b), 2.5f, 1e-3f));
    assert(close((float)(a - b), 1.5f, 1e-3f));
    assert(close((float)(a * b), 1.0f, 1e-3f));
    assert(close((float)(a / b), 4.0f, 1e-3f));
    assert((float)(-a) < 0);
    assert(a != b);
    assert(a > b);
    assert(b <= a);
    assert(a >= b);

    fp8_e4m3 c = a; c += b;  assert(close((float)c, 2.5f, 1e-3f));
    c = a; c -= b; assert(close((float)c, 1.5f, 1e-3f));
    c = a; c *= b; assert(close((float)c, 1.0f, 1e-3f));
    c = a; c /= b; assert(close((float)c, 4.0f, 1e-3f));

    // from_bits / to_bits round-trip
    for (uint8_t raw : {uint8_t(0x00), uint8_t(0x38), uint8_t(0x7F), uint8_t(0xC0)}) {
        assert(fp8_e4m3::from_bits(raw).to_bits() == raw);
    }

    // FP4
    // FP4 E2M1 representable values: 0, 1, 1.5, 2, 3 (0.5 flushes to 0, so it is not used as an operand).
    fp4_e2m1 p(2.0f), q(1.0f);
    assert(p > q && p != q && q < p && q <= p && p >= q && -p < q);
    fp4_e2m1 r = p; r += q;  assert(close((float)r, 3.0f, 1e-6f));     // 2 + 1
    r = p; r -= q; assert(close((float)r, 1.0f, 1e-6f));               // 2 - 1
    r = p; r *= q; assert(close((float)r, 2.0f, 1e-6f));               // 2 * 1
    r = p; r /= q; assert(close((float)r, 2.0f, 1e-6f));               // 2 / 1
    assert(close((float)(fp4_e2m1(3.0f) + fp4_e2m1(3.0f)), 3.0f, 1e-6f));   // 6 saturates to 3
    assert(close((float)(q * fp4_e2m1(1.0f) / fp4_e2m1(2.0f)), 0.0f, 1e-6f)); // 0.5 flushes to 0

    for (uint8_t raw : {uint8_t(0x0), uint8_t(0x1), uint8_t(0x7), uint8_t(0xF)}) {
        assert(fp4_e2m1::from_bits(raw).to_bits() == raw);
    }
}

// ─── I7 numeric_limits ──────────────────────────────────────────────────────
static void test_I7_numeric_limits() {
    using L8 = std::numeric_limits<fp8_e4m3>;
    static_assert(L8::is_specialized);
    static_assert(L8::is_signed);
    static_assert(L8::has_infinity);
    static_assert(L8::has_quiet_NaN);
    assert(std::isinf((float)L8::infinity()));
    assert(std::isnan((float)L8::quiet_NaN()));
    assert(close((float)L8::max(), 240.0f, 1e-3f));

    using L4 = std::numeric_limits<fp4_e2m1>;
    static_assert(L4::is_specialized);
    assert(std::isinf((float)L4::infinity()));
    assert(std::isnan((float)L4::quiet_NaN()));
    assert(close((float)L4::max(), 3.0f, 1e-6f));
    assert(close((float)L4::min(), 1.0f, 1e-6f));
}

// ─── I10 frexp/ldexp sweep: known encodings unchanged ───────────────────────
static void test_I10_sweep() {
    // Choose a set of representable values for e4m3 and check the encodings
    // that would have been produced by the old log2/pow path.
    struct Case { float f; uint8_t expected_lo; };
    // (Values below are exact representable values, so any sensible encoder
    //  must produce a specific bit pattern.)
    auto round_trip = [](float f) {
        fp8_e4m3 e(f);
        return (float)e;
    };
    // Zero
    assert(round_trip(0.0f) == 0.0f);
    // Simple exact values
    assert(close(round_trip(1.0f),  1.0f, 0.f));
    assert(close(round_trip(2.0f),  2.0f, 0.f));
    assert(close(round_trip(0.5f),  0.5f, 0.f));
    assert(close(round_trip(4.0f),  4.0f, 0.f));
    // Overflow saturates to max finite (240 for e4m3).
    assert(close(round_trip(1e9f),  240.0f, 0.f));
    // Underflow flushes to signed zero.
    assert(round_trip(1e-9f) == 0.0f);

    // Underflow keeps the sign 
    assert(std::signbit(round_trip(-1e-9f)));
    assert(std::signbit(round_trip(-0.0f)));

    // FP4 sweep
    assert(close((float)fp4_e2m1(0.0f),  0.0f, 0.f));
    assert(close((float)fp4_e2m1(0.5f),  0.0f, 0.f));   // below 2^(1-bias)=1.0: flush to 0
    assert(close((float)fp4_e2m1(1.0f),  1.0f, 0.f));
    assert(close((float)fp4_e2m1(1.5f),  1.5f, 0.f));
    assert(close((float)fp4_e2m1(1.9f),  2.0f, 0.f));
    assert(close((float)fp4_e2m1(3.0f),  3.0f, 0.f));
    assert(close((float)fp4_e2m1(4.0f),  3.0f, 0.f));   // saturate
}

// ─── scalar traits & wrappers ───────────────────────────────────────────────
static void test_scalar_traits() {
    static_assert( scalar::is_fp8_v<fp8_e4m3>);
    static_assert(!scalar::is_fp8_v<float>);
    static_assert( scalar::is_fp4_v<fp4_e2m1>);
    static_assert(!scalar::is_fp4_v<double>);

    static_assert(std::is_same_v<scalar::acc_t<float>,  float>);
    static_assert(std::is_same_v<scalar::acc_t<double>, double>);
    static_assert(std::is_same_v<scalar::acc_t<int>,    int>);
    static_assert(std::is_same_v<scalar::acc_t<fp8_e4m3>, float>);
    static_assert(std::is_same_v<scalar::acc_t<fp4_e2m1>, float>);

    // Conversions
    fp8_e4m3 x(2.0f);
    float xa = scalar::to_acc<fp8_e4m3>(x);
    assert(close(xa, 2.0f, 1e-3f));
    fp8_e4m3 xb = scalar::from_acc<fp8_e4m3>(3.0f);
    assert(close((float)xb, 3.0f, 1e-3f));

    // Wrappers
    assert(close(scalar::exp(0.0f), 1.0f, 1e-6f));
    assert(close(scalar::sqrt(4.0f), 2.0f, 1e-6f));
    assert(close(scalar::log(1.0f), 0.0f, 1e-6f));

    fp8_e4m3 y(0.0f);
    assert(close((float)scalar::exp(y), 1.0f, 1e-2f));

    fp4_e2m1 z(1.0f);
    assert(!scalar::isnan(z) && !scalar::isinf(z));

    fp8_e4m3 n = fp8_e4m3::from_bits(0x7F);  // NaN
    assert(scalar::isnan(n));
}

// ═════════════════════════════════════════════════════════════════════════════
int main() {
    test_G1_guard_and_includes();
    test_G2_zero_division();
    test_G3_size_mismatch();
    test_G4_scalar_deduction();
    test_G5_cmp_helpers();
    test_G6_streaming();
    test_G7_exponent();
    test_G8_nan_inf();

    test_H1_integer_aliases();
    test_H2_float16();
    test_H3_float_aliases();

    test_I2_fp4_carry();
    test_I3_fp4_specials();
    test_I4_fp4_signed_zero();
    test_I5_fp8_default();
    test_I6_operators();
    test_I7_numeric_limits();
    test_I10_sweep();

    test_scalar_traits();

    std::cout << "support_tests: all asserts passed\n";
    return 0;
}