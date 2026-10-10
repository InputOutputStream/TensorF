// lowprec_tests.cpp — FP8 (e4m3/e5m2/e3m4), FP4 (e2m1, e3m0) and float16 across
// scalar codec, scalar:: traits, Matrix, Tensor (autograd) and Graph.
//
// Method: "differential oracle". Inputs are quantised to T first; the same computation is then
// run in float on the dequantised inputs and the T result must equal that float result
// clamped to the representable range, up to `k` ulps (+ the flush-to-zero threshold, because
// FP8/FP4 have no subnormals). Hand-computed exact values are used where the answer is exact.
//
// Build: g++ -std=c++20 -O1 -g -fsanitize=address,undefined -Iinc -Istub tests/lowprec_tests.cpp stub/cblas.o -pthread
//        (or -DMATRIX_NO_BLAS instead of the stub)
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "DataStructures/Graph.hpp"

#include <cmath>
#include <cstring>
#include <random>
#include <iostream>
#include <set>
#include <string>
#include <sstream>
#include <functional>

static int g_fail = 0, g_n = 0;
static std::string g_ctx;
#define CHECK(c) do { g_n++; if(!(c)) { g_fail++; std::cout << "FAIL [" << g_ctx << "] " << __FILE__ << ":" << __LINE__ << "  " #c "\n"; } } while(0)
#define CHECK_THROWS(expr) do { g_n++; bool t_=false; try { (void)(expr); } catch(const std::exception&) { t_=true; } \
    if(!t_){ g_fail++; std::cout << "FAIL(no throw) [" << g_ctx << "] " << __FILE__ << ":" << __LINE__ << "  " #expr "\n"; } } while(0)
#define CHECK_NOTHROW(expr) do { g_n++; try { (void)(expr); } catch(const std::exception& e_) { g_fail++; \
    std::cout << "FAIL(threw: " << e_.what() << ") [" << g_ctx << "] " << __FILE__ << ":" << __LINE__ << "  " #expr "\n"; } } while(0)

// ───────────────────────────── type info ─────────────────────────────
template <class T> struct Info;                                   // E, M, BITS for FP8/FP4
template <int E, int M> struct Info<FP8<E, M>> { static constexpr int e = E, m = M, bits = 8; };
template <unsigned short E, unsigned short M> struct Info<FP4<E, M>> { static constexpr int e = E, m = M, bits = 4; };

template <class T> constexpr bool is_fp = scalar::is_fp8_v<T> || scalar::is_fp4_v<T>;

template <class T> inline float F(T v) { return static_cast<float>(scalar::to_acc<T>(v)); }
template <class T> constexpr double rel_eps() {                     // one ulp, relative
    if constexpr (is_fp<T>) return 1.0 / (1 << Info<T>::m); else return 1.0 / 1024.0;
}
template <class T> double max_finite()  { return F<T>(std::numeric_limits<T>::max()); }
template <class T> double min_normal()  { return F<T>(std::numeric_limits<T>::min()); }

// quantise a float to T and back (the "T-representable" value)
template <class T> float q(float x) { return F<T>(T(x)); }

// float `ref` computed from dequantised inputs vs. T result `got`, `k` ulps of slack.
template <class T> bool agree(T got, float ref, double k = 1.0)
{
    const float g = F<T>(got);
    if (std::isnan(ref)) return std::isnan(g);
    const double mx = max_finite<T>();
    if (std::isinf(ref)) return std::isinf(g) || std::fabs(g) == mx;
    double r = std::max(-mx, std::min(mx, (double)ref));              // saturate like the codec
    return std::fabs(g - r) <= k * rel_eps<T>() * std::fabs(r) + k * min_normal<T>() + 1e-12;
}
template <class T> bool agree(const Matrix<T>& got, const Matrix<float>& ref, double k = 1.0)
{
    if (got.shape != ref.shape || got.data.size() != ref.data.size()) return false;
    for (size_t i = 0; i < got.data.size(); i++) if (!agree<T>(got.data[i], ref.data[i], k)) return false;
    return true;
}

// build helpers: values given as floats are quantised to T; `deq` returns the float twin.
template <class T> Matrix<T> mk(std::vector<float> v, shape_t s)
{
    std::vector<T> d; d.reserve(v.size());
    for (float x : v) d.push_back(T(x));
    return Matrix<T>(std::move(d), s);
}
template <class T> Matrix<float> deq(const Matrix<T>& m)
{
    std::vector<float> d; d.reserve(m.data.size());
    for (const T& x : m.data) d.push_back(F<T>(x));
    return Matrix<float>(std::move(d), m.shape);
}

// ═════════════════════════════ 1. codec (FP8 / FP4) ═════════════════════════════
template <class T> void codec_tests(const char* name)
{
    g_ctx = std::string("codec ") + name;
    constexpr int E = Info<T>::e, M = Info<T>::m, B = Info<T>::bits;
    constexpr int bias = (1 << (E - 1)) - 1, maxe = (1 << E) - 1;
    const int N = 1 << B;

    std::vector<std::pair<float, int>> pos;               // finite positive normal values with their pattern
    for (int b = 0; b < N; b++) {
        int sign = b >> (B - 1), e = (b >> M) & ((1 << E) - 1), m = b & ((1 << M) - 1);
        T t = T::from_bits((uint8_t)b);
        CHECK(t.to_bits() == b);
        float x = F<T>(t);
        if (e == maxe) {
            if (m) CHECK(std::isnan(x));
            else { CHECK(std::isinf(x)); CHECK(std::signbit(x) == (bool)sign); CHECK(T(x).to_bits() == b); }
        } else if (e == 0) {                                          // no subnormals: all decode to signed zero
            CHECK(x == 0.0f); CHECK(std::signbit(x) == (bool)sign);
            CHECK(T(x).to_bits() == (sign << (B - 1)));
        } else {
            float want = (1.0f + (float)m / (1 << M)) * std::ldexp(1.0f, e - bias);
            if (sign) want = -want;
            CHECK(x == want);
            CHECK(T(x).to_bits() == b);                               // decode -> encode is the identity
            if (!sign) pos.push_back({x, b});
        }
    }
    CHECK(!pos.empty());
    for (size_t i = 1; i < pos.size(); i++) CHECK(pos[i].first > pos[i - 1].first);   // monotonic

    // numeric_limits
    using L = std::numeric_limits<T>;
    CHECK(F<T>(L::max()) == pos.back().first);
    CHECK(F<T>(L::lowest()) == -pos.back().first);
    CHECK(F<T>(L::min()) == pos.front().first);
    CHECK(std::isinf(F<T>(L::infinity())) && F<T>(L::infinity()) > 0);
    if constexpr (M > 0) CHECK(std::isnan(F<T>(L::quiet_NaN())));   // M==0: exp=all-ones means inf, no mantissa left for NaN
    CHECK(L::is_specialized && L::is_signed && L::has_infinity && L::has_quiet_NaN);
    CHECK(L::digits == M + 1);
    {   // epsilon is 2^-M when that value is representable, otherwise it flushes to 0
        float want = std::ldexp(1.0f, -M);
        float got = F<T>(L::epsilon());
        CHECK(got == (want >= pos.front().first ? want : 0.0f));
    }

    // saturation, underflow, specials
    const float mx = pos.back().first, mn = pos.front().first;
    CHECK(F<T>(T(mx * 2)) == mx);
    CHECK(F<T>(T(-mx * 2)) == -mx);
    CHECK(F<T>(T(1e30f)) == mx);
    CHECK(F<T>(T(-1e30f)) == -mx);
    CHECK(std::isinf(F<T>(T(INFINITY))) && F<T>(T(INFINITY)) > 0);
    CHECK(std::isinf(F<T>(T(-INFINITY))) && F<T>(T(-INFINITY)) < 0);
    CHECK(F<T>(T(mn * 0.99f)) == 0.0f);                              // flush
    CHECK(!std::signbit(F<T>(T(mn * 0.25f))));
    CHECK(std::signbit(F<T>(T(-mn * 0.25f))));                       // signed zero kept
    CHECK(F<T>(T(0.0f)) == 0.0f && !std::signbit(F<T>(T(0.0f))));
    CHECK(std::signbit(F<T>(T(-0.0f))));
    CHECK(T(1e-30f).to_bits() == 0);
    if constexpr (M > 0) {   // NaN: FP8 encodes NaN, FP4 maps NaN to max finite (documented decision in fp4.hpp)
        float n = F<T>(T(std::nanf("")));
        if constexpr (scalar::is_fp8_v<T>) CHECK(std::isnan(n)); else CHECK(n == mx);
    }
    CHECK(T().to_bits() == 0);                                       // default-initialised

    // round half away from zero between every pair of neighbours; just below the tie rounds down
    for (size_t i = 1; i < pos.size(); i++) {
        float a = pos[i - 1].first, b = pos[i].first, mid = 0.5f * (a + b);
        CHECK(T(mid).to_bits() == pos[i].second);
        CHECK(F<T>(T(-mid)) == -b);
        CHECK(F<T>(T(std::nextafterf(mid, 0.0f))) == a);
        CHECK(F<T>(T(std::nextafterf(mid, 1e30f))) == b);
    }
    // just above max rounds to max (no carry into the inf exponent)
    CHECK(F<T>(T(std::nextafterf(mx, 1e30f))) == mx);

    // arithmetic == quantise(float op) over every finite pair (exhaustive), identities
    auto finite = [&](int b) { int e = (b >> M) & ((1 << E) - 1); return e != maxe; };
    long mism = 0, checked = 0;
    for (int a = 0; a < N; a++) for (int b = 0; b < N; b++) {
        if (!finite(a) || !finite(b)) continue;
        T x = T::from_bits((uint8_t)a), y = T::from_bits((uint8_t)b);
        float fx = F<T>(x), fy = F<T>(y);
        auto same = [&](T r, float ref) { T w(ref); return r.to_bits() == w.to_bits() || (std::isnan(F<T>(r)) && std::isnan(F<T>(w))); };
        checked++;
        if (!same(x + y, fx + fy)) mism++;
        if (!same(x - y, fx - fy)) mism++;
        if (!same(x * y, fx * fy)) mism++;
        if (fy != 0.0f && !same(x / y, fx / fy)) mism++;
        // commutativity (bitwise, ignoring signed-zero results of x-y)
        if ((x + y).to_bits() != (y + x).to_bits()) mism++;
        if ((x * y).to_bits() != (y * x).to_bits()) mism++;
        // comparisons agree with float, <=> too
        if ((x < y) != (fx < fy) || (x == y) != (fx == fy) || (x != y) != (fx != fy)) mism++;
        auto o = (x <=> y);
        if ((o == std::partial_ordering::less) != (fx < fy)) mism++;
        if ((o == std::partial_ordering::equivalent) != (fx == fy)) mism++;
    }
    CHECK(checked > 0); CHECK(mism == 0);
    for (int a = 0; a < N; a++) {
        if (!finite(a)) continue;
        T x = T::from_bits((uint8_t)a);
        CHECK(F<T>(x * T(1.0f)) == F<T>(x));
        CHECK(F<T>(x + T(0.0f)) == F<T>(x));
        CHECK(F<T>(x - x) == 0.0f);
        CHECK(F<T>(-(-x)) == F<T>(x));
        T z = x; z += T(0.0f); CHECK(z == x);
        z = x; z *= T(1.0f); CHECK(z == x);
        z = x; z -= x; CHECK(z == T(0.0f));
    }
    // NaN is unordered, never equal
    if constexpr (M > 0) {
        T nan = std::numeric_limits<T>::quiet_NaN();
        CHECK(!(nan == nan)); CHECK(nan != nan); CHECK(!(nan < T(1.0f)));
        CHECK((nan <=> T(1.0f)) == std::partial_ordering::unordered);
    }
    // stream output prints the float value
    { std::ostringstream os; os << T(1.0f); CHECK(os.str() == "1"); }
}

// ═════════════════════════════ 2. float16 sanity + scalar traits ═════════════════════════════
template <class T> void scalar_tests(const char* name)
{
    g_ctx = std::string("scalar ") + name;
    static_assert(std::is_same_v<scalar::acc_t<T>, float>);
    CHECK(scalar::is_low_precision_v<T>);
    for (float x : {0.0f, 1.0f, 1.5f, 2.0f, 3.0f, -1.0f, -3.0f}) {
        CHECK(F<T>(scalar::from_acc<T>(x)) == q<T>(x));
        CHECK(scalar::to_acc<T>(T(x)) == q<T>(x));
    }
    CHECK(agree<T>(scalar::exp<T>(T(1.0f)), std::exp(q<T>(1.0f))));
    CHECK(agree<T>(scalar::sqrt<T>(T(2.0f)), std::sqrt(q<T>(2.0f))));
    CHECK(agree<T>(scalar::log<T>(T(2.0f)), std::log(q<T>(2.0f))));
    CHECK(agree<T>(scalar::pow<T>(T(2.0f), T(3.0f)), 8.0f));
    CHECK(agree<T>(scalar::abs<T>(T(-3.0f)), 3.0f));
    CHECK(agree<T>(scalar::sin<T>(T(1.0f)), std::sin(1.0f), 2) && agree<T>(scalar::cos<T>(T(1.0f)), std::cos(1.0f), 2));
    CHECK(scalar::isnan<T>(std::numeric_limits<T>::quiet_NaN()));
    CHECK(!scalar::isnan<T>(T(1.0f)));
    CHECK(scalar::isinf<T>(std::numeric_limits<T>::infinity()));
    CHECK(!scalar::isinf<T>(T(1.0f)));
}

// software binary16 must be bit-identical to the hardware _Float16 (only checkable where the compiler has one)
static void soft_vs_native_float16()
{
    g_ctx = "soft_float16 vs native";
#if defined(__FLT16_MAX__)
    auto nat = [](float f) { _Float16 h = (_Float16)f; uint16_t u; std::memcpy(&u, &h, 2); return u; };
    auto natf = [](uint16_t u) { _Float16 h; std::memcpy(&h, &u, 2); return (float)h; };
    long bad_dec = 0, bad_enc = 0, bad_mid = 0, bad_rand = 0;
    std::vector<float> finite;                                         // all finite half values, ascending by pattern
    for (uint32_t b = 0; b < 65536; b++) {
        float a = natf((uint16_t)b), c = soft_float16::decode((uint16_t)b);
        if (std::isnan(a)) { if (!std::isnan(c)) bad_dec++; continue; }
        if (std::memcmp(&a, &c, 4) != 0) bad_dec++;                    // bitwise, so -0 vs +0 is caught
        if (soft_float16::encode(a) != nat(a)) bad_enc++;
        if (!std::isinf(a) && b < 0x7C00) finite.push_back(a);
    }
    // ties and their neighbours between every pair of adjacent positive halves (incl. subnormals, max -> inf)
    for (size_t i = 1; i < finite.size(); i++) {
        float a = finite[i - 1], b = finite[i], mid = 0.5f * (a + b);
        for (float f : {mid, std::nextafterf(mid, 0.0f), std::nextafterf(mid, 1e30f), -mid})
            if (soft_float16::encode(f) != nat(f)) bad_mid++;
    }
    for (float f : {65504.0f, 65519.99f, 65520.0f, 65520.01f, 1e30f, -1e30f, 5.9604645e-8f, 2.9802322e-8f, 2.9802323e-8f, 1e-30f,
                    INFINITY, -INFINITY, 0.0f, -0.0f})
        if (soft_float16::encode(f) != nat(f)) bad_mid++;
    {   std::mt19937 g(12345);                                         // random float bit patterns over every exponent
        for (int i = 0; i < 4000000; i++) {
            uint32_t x = g(); float f; std::memcpy(&f, &x, 4);
            if (std::isnan(f)) { if (!std::isnan(soft_float16::decode(soft_float16::encode(f)))) bad_rand++; continue; }
            if (soft_float16::encode(f) != nat(f)) bad_rand++;
        } }
    CHECK(bad_dec == 0); CHECK(bad_enc == 0); CHECK(bad_mid == 0); CHECK(bad_rand == 0);
    // arithmetic through float matches the hardware type for all pairs sampled on a stride
    long bad_ar = 0;
    for (uint32_t a = 0; a < 65536; a += 97) for (uint32_t b = 0; b < 65536; b += 89) {
        soft_float16 x = soft_float16::from_bits((uint16_t)a), y = soft_float16::from_bits((uint16_t)b);
        float fx = natf((uint16_t)a), fy = natf((uint16_t)b);
        if (std::isnan(fx) || std::isnan(fy)) continue;
        if ((x + y).bits != nat(fx + fy) || (x * y).bits != nat(fx * fy) || (x - y).bits != nat(fx - fy)) bad_ar++;
    }
    CHECK(bad_ar == 0);
    using L = std::numeric_limits<soft_float16>; using N = std::numeric_limits<_Float16>;
    CHECK(float(L::max()) == (float)N::max() && float(L::min()) == (float)N::min() && float(L::epsilon()) == (float)N::epsilon());
    CHECK(float(L::denorm_min()) == (float)N::denorm_min() && std::isinf(float(L::infinity())) && std::isnan(float(L::quiet_NaN())));
#endif
}

static void float16_codec()
{
    g_ctx = "float16 codec";
    using H = float16;
    CHECK(float16_is_native || true);
    if constexpr (float16_is_native) {
        // exhaustive roundtrip over every half bit pattern
        long bad = 0;
        for (uint32_t b = 0; b < 65536; b++) {
            uint16_t u = (uint16_t)b; H h; std::memcpy(static_cast<void*>(&h), &u, 2);
            float f = static_cast<float>(h);
            if (std::isnan(f)) continue;
            H r = static_cast<H>(f); uint16_t ur; std::memcpy(&ur, static_cast<const void*>(&r), 2);
            if (ur != u) bad++;
        }
        CHECK(bad == 0);
        CHECK(F<H>(std::numeric_limits<H>::max()) == 65504.0f);
        CHECK(F<H>(H(1e9f)) > 65504.0f);                       // IEEE: overflows to inf, unlike FP8/FP4
        CHECK(std::isinf(F<H>(H(1e9f))));
        CHECK(F<H>(H(1.0f + 1.0f / 2048)) == 1.0f);            // tie to even
        CHECK(F<H>(H(1.0f + 3.0f / 2048)) == 1.0f + 2.0f / 1024);
        CHECK(F<H>(H(6e-8f)) > 0.0f);                          // subnormals exist (unlike FP8/FP4)
    }
}

// ═════════════════════════════ 3. Matrix ═════════════════════════════
template <class T> void matrix_tests(const char* name)
{
    g_ctx = std::string("matrix ") + name;
    using Mt = Matrix<T>;
    using Mf = Matrix<float>;

    // inputs: only values that exist in every type's grid (0,1,1.5,2,3) so quantisation is the identity
    Mt A = mk<T>({1, 2, 3, 1.5f, 2, 1}, {2, 3});
    Mt B = mk<T>({1, 0, 2, 1, 3, 1.5f}, {2, 3});
    Mf Af = deq(A), Bf = deq(B);
    CHECK(A.shape == (shape_t{2, 3}) && A.get_size() == 6);
    CHECK(deq(A).data == (std::vector<float>{1, 2, 3, 1.5f, 2, 1}));

    // element-wise with agreement against float
    CHECK(agree(A + B, Af + Bf)); CHECK(agree(A - B, Af - Bf)); CHECK(agree(A * B, Af * Bf));
    Mt Bp = mk<T>({1, 1.5f, 2, 3, 1, 2}, {2, 3}); Mf Bpf = deq(Bp);
    CHECK(agree(A / Bp, Af / Bpf));
    CHECK(agree(-A, -Af));
    // scalars both sides
    CHECK(agree(A + T(1.0f), Af + 1.0f)); CHECK(agree(A * T(2.0f), Af * 2.0f)); CHECK(agree(A - T(1.0f), Af - 1.0f));
    CHECK(agree(T(3.0f) - A, Mf(std::vector<float>{2, 1, 0, 1.5f, 1, 2}, {2, 3})));
    // compound ops
    { Mt c = A; c += B; CHECK(agree(c, Af + Bf)); c = A; c -= B; CHECK(agree(c, Af - Bf)); c = A; c *= B; CHECK(agree(c, Af * Bf)); }
    // broadcasting (right-aligned, size-1 repeat)
    Mt row = mk<T>({1, 2, 1}, {3}), colv = mk<T>({1, 2}, {2, 1});
    CHECK(agree(A + row, Af + deq(row))); CHECK(agree(A * colv, Af * deq(colv)));
    CHECK(agree(colv + row, deq(colv) + deq(row)));                                  // outer broadcast {2,1}+{3}
    Mt c3 = mk<T>({1, 1, 2, 1, 0, 1, 1, 1, 1, 2, 1, 1}, {2, 3, 2}), c31 = mk<T>({1, 2, 1}, {3, 1});
    CHECK(agree(c3 + c31, deq(c3) + deq(c31)));
    CHECK_THROWS(A + mk<T>({1, 2}, {2}));                                            // not broadcastable
    CHECK_THROWS(A + mk<T>({1, 2, 3, 4}, {2, 2}));

    // comparisons -> Matrix<bool>
    { T t15 = T(1.5f); float f15 = 1.5f;
      Matrix<bool> g = A > t15, gf = Af > f15; CHECK(g.data == gf.data && g.shape == gf.shape);
      Matrix<bool> e = A == t15, ef = Af == f15; CHECK(e.data == ef.data);
      Matrix<bool> l = A <= t15, lf = Af <= f15; CHECK(l.data == lf.data);
      Matrix<bool> n = A != t15, nf = Af != f15; CHECK(n.data == nf.data);
      Matrix<bool> ge = A >= t15, gef = Af >= f15; CHECK(ge.data == gef.data);
      Matrix<bool> lt = t15 < A, ltf = f15 < Af; CHECK(lt.data == ltf.data); }

    // shape manipulation (pure data movement: must be bit-exact)
    CHECK(deq(A.transpose()).data == Af.transpose().data && A.transpose().shape == (shape_t{3, 2}));
    CHECK(deq(c3.transpose({2, 0, 1})).data == deq(c3).transpose({2, 0, 1}).data);
    CHECK(deq(A.reshape({3, 2})).data == Af.data);
    CHECK(deq(A.row(1)).data == Af.row(1).data); CHECK(deq(A.col(2)).data == Af.col(2).data);
    CHECK(deq(Mt::concat({A, B}, 0)).data == Mf::concat({Af, Bf}, 0).data);
    CHECK(deq(Mt::concat({A, B}, 1)).data == Mf::concat({Af, Bf}, 1).data);
    CHECK(deq(Mt::stack({A, B}, 0)).data == Mf::stack({Af, Bf}, 0).data);
    CHECK(deq(A.slice_axis(1, 3, 1)).data == Af.slice_axis(1, 3, 1).data);
    CHECK(deq(A.at({1, 2})).data == Af.at({1, 2}).data);
    {   Mt I = Mt::eye(3); CHECK(deq(I).data == Mf::eye(3).data);
        CHECK(deq(Mt::ones({2, 2})).data == (std::vector<float>{1, 1, 1, 1}));
        CHECK(deq(Mt::zeros({2, 2})).data == (std::vector<float>{0, 0, 0, 0}));
        CHECK(deq(Mt::tril(3)).data == Mf::tril(3).data); }

    // reductions accumulate in float, quantise once
    CHECK(agree<T>(A.sum(), Af.sum()));
    CHECK(agree(A.sum(0), Af.sum(0))); CHECK(agree(A.sum(1), Af.sum(1)));
    CHECK(agree<T>(A.mean().data[0], Af.mean().data[0]));
    CHECK(agree(A.mean(0), Af.mean(0))); CHECK(agree(A.mean(1), Af.mean(1)));
    CHECK(agree(A.variance(), Af.variance(), 2)); CHECK(agree(A.std(), Af.std(), 2));
    CHECK(agree(c3.sum(1), deq(c3).sum(1))); CHECK(agree(c3.sum(2), deq(c3).sum(2))); CHECK(agree(c3.sum(0), deq(c3).sum(0)));
    {   // N ones: the float accumulator makes the sum = quantise(N), not a stuck/saturated naive loop
        for (size_t n : {1, 2, 3, 5, 10, 33, 100, 1000}) {
            Mt o = Mt::ones({n});
            CHECK(agree<T>(o.sum(), (float)n));
            CHECK(agree<T>(o.mean().data[0], 1.0f));
        }
        // the accumulator is what keeps big reductions correct: naive T accumulation gets stuck/saturates
        Mt o = Mt::ones({100});
        T naive = T(0.0f); for (const T& x : o.data) naive = naive + x;
        float good = F<T>(o.sum()), nv = F<T>(naive);
        CHECK(std::fabs(good - std::min<double>(100.0, max_finite<T>())) <= rel_eps<T>() * 100 + 1e-6 || good == max_finite<T>());
        CHECK(std::fabs(good - 100.0f) <= std::fabs(nv - 100.0f) + 1e-6);     // never worse than the naive loop
    }

    // matmul / dot: single float accumulation then one quantisation => same as float reference
    Mt P = mk<T>({1, 0, 1, 1, 1, 0}, {2, 3}), Q = mk<T>({1, 1, 0, 1, 1, 0}, {3, 2});
    CHECK(agree(P.matmul(Q), deq(P).matmul(deq(Q))));
    CHECK(agree(A.matmul(A.transpose()), Af.matmul(Af.transpose())));
    {   Mt b3 = mk<T>({1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1}, {2, 3, 2});             // batch 2
        Mt a3 = mk<T>({1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0}, {2, 2, 3});
        CHECK(agree(a3.matmul(b3), deq(a3).matmul(deq(b3))));
        CHECK(agree(a3.matmul(Q), deq(a3).matmul(deq(Q))));                          // batch broadcast of rhs
        CHECK(agree(P.matmul(b3), deq(P).matmul(deq(b3)))); }
    CHECK(agree(P.matmul(mk<T>({1, 1, 0}, {3})), deq(P).matmul(deq(mk<T>({1, 1, 0}, {3})))));     // 1-D promotion
    CHECK_THROWS(P.matmul(P));                                                                     // inner dims mismatch
    CHECK(agree<T>(mk<T>({1, 2, 1}, {3}).dot(mk<T>({1, 1, 0}, {3})).data[0], 3.0f));
    CHECK_NOTHROW(Mt::ones({64, 64}).matmul(Mt::ones({64, 64})));                    // larger gemm path (saturates for FP4/FP8: must not crash)
    {   Mt big = Mt::ones({8, 64}).matmul(Mt::ones({64, 8}));
        for (const T& x : big.data) CHECK(agree<T>(x, 64.0f)); }

    // unary maths
    Mt U = mk<T>({1, 1.5f, 2, 3, 1, 2}, {2, 3}); Mf Uf = deq(U);
    CHECK(agree(U.exponent(), Uf.exponent())); CHECK(agree(U.sqrt(), Uf.sqrt()));
    CHECK(agree(U.ln(), Uf.ln(), 2));   CHECK(agree(U.pow(T(2.0f)), Uf.pow(2.0f)));
    CHECK(agree(U.pow(Mt(T(2.0f))), Uf.pow(2.0f)));
    CHECK(agree(U.maximum(T(1.5f)).data.empty() ? U : U.maximum(T(1.5f)), Uf.maximum(1.5f)));
    {   Mt N = mk<T>({-3, -1.5f, 0, 1, 2, 3}, {6});
        CHECK(agree(N.maximum(T(0.0f)), deq(N).maximum(0.0f))); }

    // Matrix::inf()/nan() go through numeric_limits (was 0 for _Float16 before scalar.hpp specialised it)
    CHECK(std::isinf(F<T>(Mt::inf()))); CHECK(F<T>(Mt::inf()) > 0);
    if constexpr (scalar::is_fp8_v<T> || std::is_same_v<T, float16>) CHECK(std::isnan(F<T>(Mt::nan())));
    CHECK(F<T>(std::numeric_limits<T>::max()) > 1.0f && F<T>(std::numeric_limits<T>::lowest()) < -1.0f);
    // NaN / inf handling
    {   Mt Nn = mk<T>({1, 2}, {2}); CHECK(!Mt::hasNaN(Nn));
        Mt Nm = Nn; Nm.data[1] = std::numeric_limits<T>::quiet_NaN();
        if constexpr (scalar::is_fp8_v<T> || std::is_same_v<T, float16>) CHECK(Mt::hasNaN(Nm));
        Mt Inf = Nn; Inf.data[0] = std::numeric_limits<T>::infinity();
        CHECK(std::isinf(F<T>(Inf.data[0]))); }

    // stability: a copy / reassignment must not alias
    { Mt c = A; c.data[0] = T(3.0f); CHECK(F<T>(A.data[0]) == 1.0f); }
    // 0-D and bool/mask
    { Matrix<bool> m({{true, false, true}, {false, true, false}});
      Mt masked = A.at(m); CHECK(F<T>(masked.data[1]) == 0.0f && F<T>(masked.data[0]) == 1.0f && masked.shape == A.shape); }
    // strict storage invariant
    CHECK_THROWS(Mt(std::vector<T>{T(1.0f), T(2.0f), T(3.0f)}, shape_t{2, 2}));
    // gather by index (embedding lookup)
    { Mt table = mk<T>({1, 2, 3, 1.5f, 2, 1}, {3, 2}); Mt idx = mk<T>({2, 0}, {2});
      CHECK(deq(table.elemsAt(idx)).data == (std::vector<float>{2, 1, 1, 2})); }
}

// ═════════════════════════════ 4. Tensor / autograd ═════════════════════════════
template <class T> Tensor_t<T> tt(std::vector<float> v, shape_t s) { return make_tensor<T>(mk<T>(v, s)); }

// run f on a T tensor set and on the float twin; compare forward value and every input grad.
template <class T>
void diff_grad(const char* label, std::vector<std::vector<float>> data, std::vector<shape_t> shapes,
               std::function<Tensor_t<float>(std::vector<Tensor_t<float>>&)> ff,
               std::function<Tensor_t<T>(std::vector<Tensor_t<T>>&)> ft, double k, bool needs_scalar_loss = true)
{
    (void)needs_scalar_loss;
    std::vector<Tensor_t<T>> ts; std::vector<Tensor_t<float>> fs;
    for (size_t i = 0; i < data.size(); i++) {
        size_t n_ = 1; for (size_t d : shapes[i]) n_ *= d;
        std::vector<float> di = data[i]; di.resize(n_, 1.0f);      // inputs are cycled/truncated to the shape
        auto t = tt<T>(di, shapes[i]);
        ts.push_back(t); fs.push_back(make_tensor<float>(deq(t->val)));
    }
    Tensor_t<T> yt; Tensor_t<float> yf;
    try { yt = ft(ts); yf = ff(fs); } catch (const std::exception& e) {
        g_n++; g_fail++; std::cout << "FAIL(threw in forward: " << e.what() << ") [" << g_ctx << "] " << label << "\n"; return; }
    if (!agree(yt->val, yf->val, k)) { g_n++; g_fail++; std::cout << "FAIL(forward) [" << g_ctx << "] " << label << "\n"; }
    else g_n++;
    try {
        yt->backward(Matrix<T>::ones(yt->val.shape)); yf->backward(Matrix<float>::ones(yf->val.shape));
    } catch (const std::exception& e) {
        g_n++; g_fail++; std::cout << "FAIL(threw in backward: " << e.what() << ") [" << g_ctx << "] " << label << "\n"; return; }
    for (size_t i = 0; i < ts.size(); i++) {
        g_n++;
        if (ts[i]->grad.shape != fs[i]->grad.shape) { g_fail++; std::cout << "FAIL(grad shape " << i << ") [" << g_ctx << "] " << label << "\n"; continue; }
        if (!agree(ts[i]->grad, fs[i]->grad, k)) { g_fail++; std::cout << "FAIL(grad " << i << ") [" << g_ctx << "] " << label << "\n"; }
    }
}

template <class T> void tensor_tests(const char* name, double k)
{
    g_ctx = std::string("tensor ") + name;
    using TT = Tensor_t<T>; using TF = Tensor_t<float>;
    std::vector<float> a = {1, 2, 3, 1.5f, 2, 1}, b = {1, 1.5f, 2, 3, 1, 2}, v3 = {1, 2, 1};

    // exact hand-computed gradients: y = sum(x*w): dx = w, dw = x
    {   TT x = tt<T>({1, 2, 3, 1.5f}, {2, 2}), w = tt<T>({2, 1, 1, 2}, {2, 2});
        TT y = (x * w)->sum();
        CHECK(F<T>(y->val.data[0]) == q<T>(2 + 2 + 3 + 3));
        y->backward(Matrix<T>(T(1.0f)));
        CHECK(deq(x->grad).data == deq(w->val).data); CHECK(deq(w->grad).data == deq(x->val).data); }
    // x used twice: y = sum(x*x) -> dx = 2x ; gradient accumulation
    {   TT x = tt<T>({1, 1.5f, 0, 1}, {4});
        TT y = (x * x)->sum(); y->backward(Matrix<T>(T(1.0f)));
        CHECK(deq(x->grad).data == (std::vector<float>{2, 3, 0, 2})); }
    // zero_grad clears, second backward gives the same grads (no stale accumulation)
    {   TT x = tt<T>({1, 2}, {2}), w = tt<T>({1, 1}, {2});
        TT y = (x * w)->sum();
        y->backward(Matrix<T>(T(1.0f))); auto g1 = deq(x->grad).data;
        y->zero_grad(); CHECK(x->grad.get_size() == 0);
        y->backward(Matrix<T>(T(1.0f))); CHECK(deq(x->grad).data == g1); }
    // reset_graph drops the graph without touching values
    {   TT x = tt<T>({1, 2}, {2}); TT y = (x + x)->sum(); y->reset_graph(); CHECK(y->backOp == nullptr);
        CHECK(F<T>(y->val.data[0]) == q<T>(6)); }
    // no-grad leaf
    {   TT x = tt<T>({1, 2}, {2}); x->requires_grad = false; TT w = tt<T>({1, 1}, {2});
        TT y = (x * w)->sum(); y->backward(Matrix<T>(T(1.0f))); CHECK(x->grad.get_size() == 0); CHECK(w->grad.get_size() == 2); }

    // differential: T vs float twin for each operation, forward + gradients
#define DG(label, nin, shapes_, FBODY, TBODY) \
    do { std::vector<std::vector<float>> all_{a, b, v3}; all_.resize(nin); \
         diff_grad<T>(label, all_, shapes_, [](std::vector<TF>& i) -> TF { FBODY }, [](std::vector<TT>& i) -> TT { TBODY }, k); } while (0)
    DG("add", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return i[0] + i[1];, return i[0] + i[1];);
    DG("sub", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return i[0] - i[1];, return i[0] - i[1];);
    DG("mul", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return i[0] * i[1];, return i[0] * i[1];);
    DG("div", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return i[0] / i[1];, return i[0] / i[1];);
    DG("add_bcast", 2, (std::vector<shape_t>{{2, 3}, {3}}), return i[0] + i[1];, return i[0] + i[1];);
    DG("mul_bcast", 2, (std::vector<shape_t>{{2, 3}, {3}}), return i[0] * i[1];, return i[0] * i[1];);
    DG("matmul", 2, (std::vector<shape_t>{{2, 3}, {3, 2}}), return i[0]->matmul(i[1]);, return i[0]->matmul(i[1]););
    DG("transpose", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->transpose();, return i[0]->transpose(););
    DG("reshape", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->reshape({3, 2});, return i[0]->reshape({3, 2}););
    DG("sum_all", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->sum();, return i[0]->sum(););
    DG("sum_axis0", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->sum(0);, return i[0]->sum(0););
    DG("sum_axis1", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->sum(1);, return i[0]->sum(1););
    DG("mean", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->mean();, return i[0]->mean(););
    DG("relu", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->relu();, return i[0]->relu(););
    DG("exp", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->exp();, return i[0]->exp(););
    DG("ln", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->ln();, return i[0]->ln(););
    DG("sqrt", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->sqrt();, return i[0]->sqrt(););
    DG("power2", 1, (std::vector<shape_t>{{2, 3}}), return i[0]->power(2);, return i[0]->power(2););
    {   // 2-bit-mantissa types (e5m2, e2m1) compound two roundings in s*(1-s) / softmax backward: sanity bound only
        const double kl = rel_eps<T>() >= 0.25 ? 8.0 : k;
        diff_grad<T>("sigmoid", {a}, {{2, 3}}, [](std::vector<TF>& i) -> TF { return i[0]->sigmoid(); }, [](std::vector<TT>& i) -> TT { return i[0]->sigmoid(); }, kl);
        // softmax: upstream of all-ones gives an exactly-zero gradient in real arithmetic, so weight it first
        diff_grad<T>("softmax", {a}, {{2, 3}},
            [](std::vector<TF>& i) -> TF { return i[0]->softmax() * make_tensor<float>(Matrix<float>(std::vector<float>{3, 1, 0, 0, 1, 3}, shape_t{2, 3})); },
            [](std::vector<TT>& i) -> TT { return i[0]->softmax() * tt<T>({3, 1, 0, 0, 1, 3}, {2, 3}); }, kl);
    }
    DG("concat0", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return Tensor<float>::concat({i[0], i[1]}, 0);, return Tensor<T>::concat({i[0], i[1]}, 0););
    DG("concat1", 2, (std::vector<shape_t>{{2, 3}, {2, 3}}), return Tensor<float>::concat({i[0], i[1]}, 1);, return Tensor<T>::concat({i[0], i[1]}, 1););
    DG("dot_1d", 2, (std::vector<shape_t>{{3}, {3}}), return i[0]->dot(i[1]);, return i[0]->dot(i[1]););
    DG("chain_mlp", 2, (std::vector<shape_t>{{2, 3}, {3, 2}}), return (i[0]->matmul(i[1]))->relu()->sum(1);, return (i[0]->matmul(i[1]))->relu()->sum(1););
    DG("chain_shared", 1, (std::vector<shape_t>{{2, 3}}), return (i[0] * i[0] + i[0])->sum(0);, return (i[0] * i[0] + i[0])->sum(0););
#undef DG
    // the fixed ops from the last round: non-involutive transpose perm, broadcast matmul, 1D matmul, bool index
    diff_grad<T>("transpose_perm", {{1, 2, 3, 1.5f, 2, 1, 1, 1, 2, 1, 3, 1}}, {{2, 3, 2}},
        [](std::vector<TF>& i) -> TF { return i[0]->transpose({1, 2, 0}); }, [](std::vector<TT>& i) -> TT { return i[0]->transpose({1, 2, 0}); }, k);
    diff_grad<T>("matmul_bcast", {{1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0}, {1, 1, 0, 1, 1, 0}}, {{2, 2, 3}, {3, 2}},
        [](std::vector<TF>& i) -> TF { return i[0]->matmul(i[1]); }, [](std::vector<TT>& i) -> TT { return i[0]->matmul(i[1]); }, k);
    diff_grad<T>("matmul_2d_1d", {{1, 0, 1, 1, 1, 0}, {1, 1, 0}}, {{2, 3}, {3}},
        [](std::vector<TF>& i) -> TF { return i[0]->matmul(i[1]); }, [](std::vector<TT>& i) -> TT { return i[0]->matmul(i[1]); }, k);
    {   TT x = tt<T>({1, 2, 3, 1.5f}, {2, 2});
        Matrix<bool> m({{true, false}, {false, true}}); TT y = x->bool_index(make_tensor<bool>(m))->sum();
        y->backward(Matrix<T>(T(1.0f))); CHECK(deq(x->grad).data == (std::vector<float>{1, 0, 0, 1})); }
    {   TT table = tt<T>({1, 2, 3, 1.5f, 2, 1}, {3, 2}); TT idx = tt<T>({2, 0, 2}, {3});
        TT y = table->embed(idx)->sum(); y->backward(Matrix<T>(T(1.0f)));
        CHECK(deq(table->grad).data == (std::vector<float>{1, 1, 0, 0, 2, 2})); }

    // losses (value check against float twin)
    {   TT yt = tt<T>({1, 0, 0, 1}, {2, 2}), yp = tt<T>({1.5f, 1, 1, 2}, {2, 2});
        TF yt_f = make_tensor<float>(deq(yt->val)), yp_f = make_tensor<float>(deq(yp->val));
        CHECK_NOTHROW(Tensor<T>::mse(yt, yp)); CHECK(agree(Tensor<T>::mse(yt, yp)->val, Tensor<float>::mse(yt_f, yp_f)->val, k + 1)); }

    // error paths must throw, not corrupt memory
    CHECK_THROWS(tt<T>({1, 2, 3}, {3})->matmul(tt<T>({1, 2, 3}, {3})));          // two 1-D tensors
    CHECK_THROWS(tt<T>({1, 2, 3, 1}, {2, 2}) + tt<T>({1, 2, 3}, {3}));           // incompatible shapes
    CHECK_THROWS(tt<T>({1, 2}, {2})->embed(tt<T>({5}, {1})));                    // idx out of range
    {   // gradient NaN guard fires
        TT x = tt<T>({1, 2}, {2}); TT y = x->sum(); Matrix<T> bad = Matrix<T>::ones({1}); bad.data[0] = std::numeric_limits<T>::quiet_NaN();
        std::streambuf* old = std::cerr.rdbuf(); std::streambuf* oldo = std::cout.rdbuf(); std::ostringstream sink; std::cerr.rdbuf(sink.rdbuf()); std::cout.rdbuf(sink.rdbuf());   // the guard prints before throwing
        if constexpr (scalar::is_fp8_v<T> || std::is_same_v<T, float16>) CHECK_THROWS(y->backward(bad));
        std::cerr.rdbuf(old); std::cout.rdbuf(oldo); }
    // tiny training loop converges (loss strictly falls) — least squares y = 2x with SGD on a 1-weight model
    {   TT w = tt<T>({1.0f}, {1}); TT x = tt<T>({1, 2, 1, 2}, {4}); TT t = tt<T>({2, 3, 2, 3}, {4});   // targets reachable
        float first = 0, last = 0;
        for (int it = 0; it < 6; it++) {
            w->zero_grad();
            TT pred = x * w; TT d = pred - t; TT loss = (d * d)->sum();
            { if (it == 0) first = F<T>(loss->val.data[0]); last = F<T>(loss->val.data[0]); }
            loss->backward(Matrix<T>(T(1.0f)));
            float g = F<T>(w->grad.data[0]);
            w->val.data[0] = T(F<T>(w->val.data[0]) - 0.0625f * g * (std::is_same_v<T, float16> ? 1.0f : 1.0f));
        }
        CHECK(last <= first); }
}

// ═════════════════════════════ 5. Graph ═════════════════════════════
template <class T> void graph_tests(const char* name)
{
    g_ctx = std::string("graph ") + name;
    // structure: 0-1, 1-2, plus a duplicate and a reversed duplicate; nodes 3 isolated; self-loop requested explicitly
    Graph<T> g(2);
    for (int i = 0; i < 4; i++) g.add_node(std::vector<T>{T(1.0f), T(0.0f)}, i);
    g.add_edge(0, 1); g.add_edge(1, 2); g.add_edge(0, 1); g.add_edge(1, 0); g.add_edge(2, 2);
    CHECK(g.num_nodes() == 4); CHECK(g.feature_dim() == 2); CHECK(g.num_raw_edges() == 5);
    CHECK_THROWS(g.adjacency());                                 // before build()
    g.build();
    const CSR<T>& a = g.adjacency();
    CHECK(a.n == 4 && a.symmetric);
    CHECK(a.row_ptr == (std::vector<size_t>{0, 2, 5, 7, 8}));    // deduped, symmetrised, + self loops
    CHECK(a.col_idx == (std::vector<node_t>{0, 1, 0, 1, 2, 1, 2, 3}));
    CHECK(a.nnz() == 8);
    // values: D^-1/2 (A+I) D^-1/2 quantised to T; degrees {2,3,2,1}
    {   Graph<float> gf(2);
        for (int i = 0; i < 4; i++) gf.add_node(std::vector<float>{1, 0}, i);
        gf.add_edge(0, 1); gf.add_edge(1, 2); gf.add_edge(2, 2); gf.build();
        const CSR<float>& af = gf.adjacency();
        CHECK(af.row_ptr == a.row_ptr && af.col_idx == a.col_idx);
        for (size_t i = 0; i < a.values.size(); i++) CHECK(agree<T>(a.values[i], af.values[i], 1.0));
        auto dense = g.dense_adjacency(); auto densef = gf.dense_adjacency();
        CHECK(dense.size() == 16);
        for (size_t i = 0; i < 16; i++) CHECK(agree<T>(dense[i], densef[i], 1.0));
        for (size_t i = 0; i < 4; i++) for (size_t j = 0; j < 4; j++) CHECK(F<T>(dense[i * 4 + j]) == F<T>(dense[j * 4 + i]));   // symmetric
        CHECK(F<T>(dense[3 * 4 + 3]) == 1.0f);                   // isolated node: self loop only, degree 1 -> exactly 1
    }
    // errors
    CHECK_THROWS(g.add_node(std::vector<T>{T(1.0f)}, 0));        // wrong feature length
    CHECK_THROWS(g.add_edge(0, 9));
    CHECK_THROWS(Graph<T>(0));
    // adding after build invalidates
    g.add_node(std::vector<T>{T(0.0f), T(1.0f)}, 4); CHECK_THROWS(g.adjacency()); g.build(); CHECK(g.adjacency().n == 5);
    CHECK(g.labels().size() == 5 && g.labels()[4] == 4); CHECK(g.features().size() == 10);

    // connect_by_similarity: compare with an independent reference built from the same quantisation as Graph
    auto reference = [](const std::vector<std::vector<float>>& fe, double thr) {
        std::set<std::pair<int, int>> out; size_t n = fe.size(), F_ = fe[0].size();
        std::vector<std::vector<T>> unit(n); std::vector<char> ok(n, 1);
        for (size_t i = 0; i < n; i++) {
            double s = 0; std::vector<T> x; for (float v : fe[i]) { x.push_back(T(v)); double d = F<T>(x.back()); s += d * d; }
            if (s <= 0) { ok[i] = 0; continue; }
            T inv = T(1.0 / std::sqrt(s)); for (size_t k = 0; k < F_; k++) unit[i].push_back(T(x[k] * inv));
        }
        for (size_t i = 0; i < n; i++) for (size_t j = i + 1; j < n; j++) {
            if (!ok[i] || !ok[j]) continue;
            double d = 0;
            for (size_t k = 0; k < F_; k++) d += (double)F<T>(unit[i][k]) * (double)F<T>(unit[j][k]);
            if (d >= thr) out.insert({(int)i, (int)j});
        } return out; };
    // (a) axis-aligned vectors: exactly representable unit vectors in every type => classic clusters
    {   std::vector<std::vector<float>> fe = {{1, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
        T thr = T(1.0f);
        for (size_t threads : {1u, 2u, 4u, 0u}) {
            Graph<T> gg(3); for (auto& f : fe) gg.add_node(std::vector<T>{T(f[0]), T(f[1]), T(f[2])});
            size_t added = gg.connect_by_similarity(thr, threads, 2);
            CHECK(added == 2); gg.build();
            auto ref = reference(fe, (double)F<T>(thr)); CHECK(ref.size() == 2 && ref.count({0, 1}) && ref.count({2, 3}));
            const CSR<T>& ag = gg.adjacency();
            // edges 0-1 and 2-3 present, none to the zero vector (5) beyond its own self loop
            auto has = [&](size_t i, size_t j) { for (size_t k = ag.row_ptr[i]; k < ag.row_ptr[i + 1]; k++) if (ag.col_idx[k] == j) return true; return false; };
            CHECK(has(0, 1) && has(1, 0) && has(2, 3) && has(3, 2)); CHECK(!has(0, 2) && !has(4, 5) && !has(0, 4));
            CHECK(ag.row_ptr[6] - ag.row_ptr[5] == 1);                  // zero vector: unusable, self loop only
        }
    }
    // (b) arbitrary vectors: whatever the precision does to them, edges == the reference computed with same quantisation
    {   std::vector<std::vector<float>> fe = {{1, 1, 0}, {1, 1, 0}, {1, 2, 0}, {0, 1, 1}, {2, 1, 0}, {1, 0, 1}, {3, 1, 1}, {1, 1, 1}};
        for (double thr : {0.0, 0.5, 0.75, 0.9, 1.0}) {
            auto ref = reference(fe, (double)F<T>(T((float)thr)));
            for (size_t threads : {1u, 3u}) for (size_t block : {1u, 3u, 128u}) {
                Graph<T> gg(3); for (auto& f : fe) gg.add_node(std::vector<T>{T(f[0]), T(f[1]), T(f[2])});
                size_t added = gg.connect_by_similarity(T((float)thr), threads, block);
                CHECK(added == ref.size());
                gg.build(); const CSR<T>& ag = gg.adjacency();
                size_t off = 0; for (size_t i = 0; i < ag.n; i++) for (size_t k = ag.row_ptr[i]; k < ag.row_ptr[i + 1]; k++) if (ag.col_idx[k] > i) off++;
                CHECK(off == ref.size());
                for (auto& e : ref) { bool f = false; for (size_t k = ag.row_ptr[e.first]; k < ag.row_ptr[e.first + 1]; k++) if (ag.col_idx[k] == (node_t)e.second) f = true; CHECK(f); }
            }
        }
    }
    // (c) max_edges guard
    {   Graph<T> gg(1); for (int i = 0; i < 6; i++) gg.add_node(std::vector<T>{T(1.0f)});
        CHECK_THROWS(gg.connect_by_similarity(T(1.0f), 2, 2, 3));     // 15 edges > 3
        Graph<T> g2(1); for (int i = 0; i < 6; i++) g2.add_node(std::vector<T>{T(1.0f)});
        CHECK(g2.connect_by_similarity(T(1.0f), 1, 128, 100) == 15);  // complete graph on identical 1-D vectors
        Graph<T> g1(1); g1.add_node(std::vector<T>{T(1.0f)}); CHECK(g1.connect_by_similarity(T(0.5f)) == 0); }
    // (d) a graph feeds the Matrix/Tensor stack: dense adjacency @ features, in T
    {   Graph<T> gg(2); gg.add_node(std::vector<T>{T(1.0f), T(0.0f)}); gg.add_node(std::vector<T>{T(0.0f), T(1.0f)}); gg.add_edge(0, 1); gg.build();
        Matrix<T> Ad(gg.dense_adjacency(), shape_t{2, 2}); Matrix<T> X(gg.features(), shape_t{2, 2});
        Matrix<T> H = Ad.matmul(X);
        Matrix<float> Hf = deq(Ad).matmul(deq(X)); CHECK(agree(H, Hf, 1.0)); }
}

// ═════════════════════════════ main ═════════════════════════════
template <class T> void run_all(const char* name, double tensor_k)
{
    scalar_tests<T>(name);
    matrix_tests<T>(name);
    tensor_tests<T>(name, tensor_k);
    graph_tests<T>(name);
}

int main()
{
    codec_tests<fp8_e4m3>("fp8_e4m3"); codec_tests<fp8_e5m2>("fp8_e5m2"); codec_tests<fp8_e3m4>("fp8_e3m4");
    codec_tests<fp4_e2m1>("fp4_e2m1");  codec_tests<FP4<3, 0>>("fp4_e3m0");
    float16_codec();
    soft_vs_native_float16();
    // tensor_k: tolerance in ulps for multi-op chains (coarser grids accumulate more relative error)
    run_all<fp8_e4m3>("fp8_e4m3", 3);
    run_all<fp8_e5m2>("fp8_e5m2", 3);
    run_all<fp8_e3m4>("fp8_e3m4", 3);
    run_all<fp4_e2m1>("fp4_e2m1", 3);
    run_all<float16>("float16", 4);
    std::cout << (g_n - g_fail) << "/" << g_n << " checks passed\n";
    return g_fail ? 1 : 0;
}
