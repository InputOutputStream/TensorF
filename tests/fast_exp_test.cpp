// fast_exp_test: accuracy and edge cases of the optional vectorisable exp (-DTENSORF_FAST_EXP).
//
//   g++ -std=c++20 -O2 -DTENSORF_FAST_EXP -Iinclude -Iinclude/core -Iinclude/tools tests/fast_exp_test.cpp -pthread
//
// Without the define the default exp is std::exp and this test only checks that (bit-identical results, 0 ulp).
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// error of `got` against the correctly rounded float exp, in units of the last place of the true result
static double ulp_err(float got, double x) {
    const double ref = std::exp(x);
    const float rf = (float)ref;
    if (!(rf > 0) || std::isinf(rf)) return 0.0;
    int e; std::frexp(rf, &e);
    const double ulp = std::ldexp(1.0, e - 24);
    return std::fabs((double)got - ref) / ulp;
}

int main() {
#ifdef TENSORF_FAST_EXP
    std::printf("TENSORF_FAST_EXP is ON\n");
#else
    std::printf("TENSORF_FAST_EXP is off: default exp must be bit-identical to std::exp\n");
#endif
    // 1. dense sweep over the whole useful range, through the real kernel (Matrix::exponent)
    const size_t N = 2'000'003;
    std::vector<float> xs(N);
    for (size_t i = 0; i < N; ++i) xs[i] = -87.3f + (88.7f + 87.3f) * (float)i / (float)(N - 1);
    Matrix<float> X(xs, shape_t{N});
    Matrix<float> Y = X.exponent();
    double max_ulp = 0, sum_ulp = 0; size_t worst = 0; size_t bit_exact = 0;
    for (size_t i = 0; i < N; ++i) {
        const double u = ulp_err(Y.data[i], xs[i]);
        if (u > max_ulp) { max_ulp = u; worst = i; }
        sum_ulp += u;
        const float sref = std::exp(xs[i]);
        if (std::memcmp(&sref, &Y.data[i], sizeof(float)) == 0) ++bit_exact;
    }
    std::printf("  exponent() over [-87.3, 88.7], %zu points: max error %.3f ulp (at x=%g), mean %.4f ulp, %.1f%% bit-identical to std::exp\n",
                N, max_ulp, xs[worst], sum_ulp / (double)N, 100.0 * (double)bit_exact / (double)N);
#ifdef TENSORF_FAST_EXP
    CHECK(max_ulp < 2.0, "max error %.3f ulp exceeds 2", max_ulp);
#else
    CHECK(bit_exact == N, "default exp is not bit-identical to std::exp (%zu of %zu)", bit_exact, N);
#endif

    // 2. special values
    const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
    Matrix<float> S(std::vector<float>{0.0f, -0.0f, nan, inf, -inf, 100.0f, -100.0f, 88.7f, 89.0f, -87.0f, -88.0f, -200.0f, 1.0f, -1.0f}, shape_t{14});
    auto E = S.exponent();
    CHECK(E.data[0] == 1.0f && E.data[1] == 1.0f, "exp(+-0) == 1");
    CHECK(std::isnan(E.data[2]), "exp(NaN) is NaN");
    CHECK(std::isinf(E.data[3]) && E.data[3] > 0, "exp(+inf) = +inf");
    CHECK(E.data[4] == 0.0f, "exp(-inf) = 0");
    CHECK(std::isinf(E.data[5]) && std::isinf(E.data[8]), "overflow gives +inf");
    CHECK(E.data[6] >= 0.0f && E.data[6] < 1e-37f, "exp(-100) underflows to 0 or a denormal  got %g", E.data[6]);
    CHECK(std::fabs(E.data[7] / std::exp(88.7f) - 1.0f) < 1e-5f, "exp(88.7) near the top of the range");
    CHECK(std::fabs(E.data[12] / std::exp(1.0f) - 1.0f) < 2e-7f && std::fabs(E.data[13] / std::exp(-1.0f) - 1.0f) < 2e-7f, "exp(+-1)");
    CHECK(E.data[9] > 0 && E.data[10] >= 0, "small arguments stay non-negative");

    // 3. sigmoid on top of it: saturates cleanly, no NaN, within 1e-6 of the exact value
    std::vector<float> sx; for (float v = -120.f; v <= 120.f; v += 0.37f) sx.push_back(v);
    Matrix<float> SG = Matrix<float>(sx, shape_t{sx.size()}).sigmoid();
    double worst_abs = 0; bool nan_seen = false;
    for (size_t i = 0; i < sx.size(); ++i) {
        const double exact = 1.0 / (1.0 + std::exp(-(double)sx[i]));
        if (std::isnan(SG.data[i])) nan_seen = true;
        worst_abs = std::fmax(worst_abs, std::fabs((double)SG.data[i] - exact));
    }
    std::printf("  sigmoid over [-120, 120]: max |error| %.3g\n", worst_abs);
    CHECK(!nan_seen && worst_abs < 1e-6, "sigmoid accuracy / saturation (max abs error %.3g)", worst_abs);

    // 4. double and low precision never use it
    Matrix<double> D(std::vector<double>{0.5, -3.25, 20.0}, shape_t{3});
    auto DE = D.exponent();
    CHECK(DE.data[0] == std::exp(0.5) && DE.data[1] == std::exp(-3.25) && DE.data[2] == std::exp(20.0), "double exp is std::exp");

    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
