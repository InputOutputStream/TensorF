// elementwise_equiv_test: the element-wise ops and the autograd glue around them must be BIT-FOR-BIT stable.
//
// Two safety nets in one file:
//   1. every result is compared with an independent per-element scalar formula (the same rounding order the
//      original composite expressions had), including NaN / inf / -0 / denormals, awkward sizes (vector tails),
//      int / float / double / float16 / FP8, scalar and row broadcasting, and the exceptions that must survive;
//   2. every section prints a checksum ("sum <name> <hash>") so the output of two builds can be diffed:
//        ./a.out > new.txt ;  (build against the old headers) ./b.out > old.txt ;  diff old.txt new.txt
//
//   g++ -std=c++20 -O2 -Iinclude -Iinclude/core -Iinclude/tools tests/elementwise_equiv_test.cpp -pthread
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "DataStructures/Graph.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static int g_fail = 0, g_checks = 0;
#define CHECK(c, ...) do { ++g_checks; if (!(c)) { ++g_fail; std::printf("  FAIL %s:%d  ", __FILE__, __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

// ── hashing / bitwise comparison (all NaNs are one value) ──────────────────────────────────────────────────
template <class T> static bool same(const T& a, const T& b) {
    if constexpr (std::is_floating_point_v<T>) { if (std::isnan(a) && std::isnan(b)) return true; }
    else if constexpr (!std::is_integral_v<T>) { if (std::isnan((float)a) && std::isnan((float)b)) return true; }
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}
template <class T> static uint64_t hash_vec(const std::vector<T>& v) {
    uint64_t h = 1469598103934665603ull;
    for (const T& x : v) {
        T y = x;
        if constexpr (std::is_floating_point_v<T>) { if (std::isnan(y)) y = std::numeric_limits<T>::quiet_NaN(); }
        else if constexpr (!std::is_integral_v<T>) { if (std::isnan((float)y)) y = T(std::numeric_limits<float>::quiet_NaN()); }
        const unsigned char* p = (const unsigned char*)&y;
        for (size_t i = 0; i < sizeof(T); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    }
    return h;
}
template <class T> static void emit(const char* name, const std::vector<T>& v) {
    std::printf("sum %-38s %016llx\n", name, (unsigned long long)hash_vec(v));
}
template <class T> static bool all_same(const std::vector<T>& a, const std::vector<T>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) if (!same(a[i], b[i])) return false;
    return true;
}

// ── test data: mixed magnitudes + special values, deterministic ─────────────────────────────────────────────
template <class T> static std::vector<T> make_data(size_t n, uint32_t seed, bool special, bool nonzero = false) {
    std::mt19937 g(seed);
    std::uniform_real_distribution<float> u(-3.0f, 3.0f);
    std::vector<T> v(n);
    for (size_t i = 0; i < n; ++i) {
        float f = u(g);
        if constexpr (std::is_integral_v<T>) f = (float)((int)(f * 7));
        v[i] = (T)f;
    }
    if (special) {
        if constexpr (std::is_floating_point_v<T>) {
            const T sp[] = {T(0), T(-0.0), std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity(),
                            std::numeric_limits<T>::quiet_NaN(), std::numeric_limits<T>::denorm_min(), T(1e30), T(-1e30), T(88), T(-88), T(100)};
            for (size_t i = 0; i < n && i < 11 * 3; ++i) v[(i * 5 + 1) % n] = sp[i % 11];
        }
    }
    if (nonzero) for (size_t i = 0; i < n; ++i) if (v[i] == T(0)) v[i] = T(1);
    return v;
}
static const size_t kSizes[] = {0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 257, 1000, 4099};

// ── 1. vecmath / Matrix binary + scalar ops, same shape ─────────────────────────────────────────────────────
template <class T> static void test_binary(const char* tn, bool float_like) {
    for (size_t n : kSizes) {
        auto a = make_data<T>(n, 11u + (uint32_t)n, float_like), b = make_data<T>(n, 77u + (uint32_t)n, float_like);
        auto bnz = b; for (auto& x : bnz) if (x == T(0)) x = T(1);
        std::vector<T> ra(n), rs(n), rm(n), rd(n);
        for (size_t i = 0; i < n; ++i) { ra[i] = (T)(a[i] + b[i]); rs[i] = (T)(a[i] - b[i]); rm[i] = (T)(a[i] * b[i]); rd[i] = (T)(a[i] / bnz[i]); }
        CHECK(all_same(vecmath::add(a, b), ra), "%s add n=%zu", tn, n);
        CHECK(all_same(vecmath::sub(a, b), rs), "%s sub n=%zu", tn, n);
        CHECK(all_same(vecmath::mul(a, b), rm), "%s mul n=%zu", tn, n);
        CHECK(all_same(vecmath::div(a, bnz), rd), "%s div n=%zu", tn, n);
        shape_t sh{n == 0 ? size_t(0) : size_t(1), n};
        Matrix<T> A(a, sh), B(b, sh), Bz(bnz, sh);
        CHECK(all_same((A + B).data, ra) && all_same((A - B).data, rs) && all_same((A * B).data, rm) && all_same((A / Bz).data, rd), "%s Matrix ops n=%zu", tn, n);
        if (n == 1000) { emit((std::string("add_") + tn).c_str(), ra); emit((std::string("mul_") + tn).c_str(), rm); emit((std::string("div_") + tn).c_str(), rd); }

        // scalar-with-vector forms
        const T s = (T)1.75f;
        std::vector<T> r1(n), r2(n), r3(n), r4(n), r5(n), r6(n), r7(n);
        for (size_t i = 0; i < n; ++i) { r1[i] = (T)(a[i] + s); r2[i] = (T)(a[i] * s); r3[i] = (T)(a[i] - s); r4[i] = (T)(s - a[i]); r5[i] = (T)(a[i] / s); r6[i] = (T)(-a[i]); }
        CHECK(all_same(vecmath::add_s(a, s), r1) && all_same(vecmath::mul_s(a, s), r2) && all_same(vecmath::sub_s(a, s), r3) &&
              all_same(vecmath::s_sub(s, a), r4) && all_same(vecmath::div_s(a, s), r5) && all_same(vecmath::neg(a), r6), "%s scalar forms n=%zu", tn, n);
        auto anz = a; for (auto& x : anz) if (x == T(0)) x = T(1);
        for (size_t i = 0; i < n; ++i) r7[i] = (T)(s / anz[i]);
        CHECK(all_same(vecmath::s_div(s, anz), r7), "%s s_div n=%zu", tn, n);

        // in-place, including x += x (exact alias)
        auto ip = a; vecmath::add_inplace(ip, b);  CHECK(all_same(ip, ra), "%s add_inplace n=%zu", tn, n);
        ip = a; vecmath::sub_inplace(ip, b);       CHECK(all_same(ip, rs), "%s sub_inplace n=%zu", tn, n);
        ip = a; vecmath::mul_inplace(ip, b);       CHECK(all_same(ip, rm), "%s mul_inplace n=%zu", tn, n);
        ip = a; vecmath::add_inplace(ip, ip);
        std::vector<T> dbl(n); for (size_t i = 0; i < n; ++i) dbl[i] = (T)(a[i] + a[i]);
        CHECK(all_same(ip, dbl), "%s x += x n=%zu", tn, n);
        ip = a; vecmath::add_inplace_s(ip, s); CHECK(all_same(ip, r1), "%s add_inplace_s n=%zu", tn, n);
        ip = a; vecmath::mul_inplace_s(ip, s); CHECK(all_same(ip, r2), "%s mul_inplace_s n=%zu", tn, n);
    }
}

// ── 2. exceptions that must survive ─────────────────────────────────────────────────────────────────────────
static void test_exceptions() {
    std::vector<float> a(10, 1.0f), b(9, 1.0f), z(10, 1.0f); z[7] = 0.0f;
    bool t1 = false, t2 = false, t3 = false, t4 = false, t5 = false;
    try { (void)vecmath::add(a, b); } catch (const std::invalid_argument&) { t1 = true; }
    try { (void)vecmath::div(a, z); } catch (const std::runtime_error&) { t2 = true; }
    try { (void)vecmath::div_s(a, 0.0f); } catch (const std::runtime_error&) { t3 = true; }
    try { std::vector<float> x = a; vecmath::add_inplace(x, b); } catch (const std::invalid_argument&) { t4 = true; }
    std::vector<int> ia(40, 6), iz(40, 2); iz[39] = 0;
    try { (void)vecmath::div(ia, iz); } catch (const std::runtime_error&) { t5 = true; }
#ifndef MATRIX_IEEE_DIVISION
    CHECK(t1 && t2 && t3 && t4 && t5, "exceptions: size=%d div0=%d div_s0=%d inplace=%d intdiv0=%d", t1, t2, t3, t4, t5);
#else
    CHECK(t1 && t4 && t5, "exceptions (IEEE mode)");
#endif
}

// ── 3. Matrix broadcasting: scalar fast paths and the general path must agree with the numpy rules ──────────
static void test_broadcast() {
    for (size_t r : {size_t(1), size_t(3), size_t(17)}) for (size_t c : {size_t(1), size_t(5), size_t(64), size_t(129)}) {
        auto d = make_data<float>(r * c, 5u + (uint32_t)(r * c), true);
        Matrix<float> A(d, shape_t{r, c});
        const float sv = 2.5f;
        Matrix<float> S1(std::vector<float>{sv}, shape_t{1});           // rank 1 scalar-like
        Matrix<float> S2(std::vector<float>{sv}, shape_t{1, 1});        // rank 2 scalar-like
        for (const Matrix<float>* S : {&S1, &S2}) {
            auto m = A * *S, p = A + *S, q = *S - A, w = A - *S;
            CHECK(m.shape == A.shape && p.shape == A.shape && q.shape == A.shape, "scalar broadcast shape r=%zu c=%zu", r, c);
            std::vector<float> em(r * c), ep(r * c), eq(r * c), ew(r * c);
            for (size_t i = 0; i < r * c; ++i) { em[i] = (float)(d[i] * sv); ep[i] = (float)(d[i] + sv); eq[i] = (float)(sv - d[i]); ew[i] = (float)(d[i] - sv); }
            CHECK(all_same(m.data, em) && all_same(p.data, ep) && all_same(q.data, eq) && all_same(w.data, ew), "scalar broadcast values r=%zu c=%zu", r, c);
            auto lhs = *S * A;                                            // scalar on the left, same result as A * S
            CHECK(lhs.shape == A.shape && all_same(lhs.data, em), "left scalar r=%zu c=%zu", r, c);
        }
        // A / scalar and scalar / A (non-zero data)
        auto nz = make_data<float>(r * c, 9u + (uint32_t)(r * c), false, true);
        Matrix<float> N(nz, shape_t{r, c});
        auto q1 = N / S1, q2 = S1 / N;
        std::vector<float> e1(r * c), e2(r * c);
        for (size_t i = 0; i < r * c; ++i) { e1[i] = (float)(nz[i] / sv); e2[i] = (float)(sv / nz[i]); }
        CHECK(q1.shape == N.shape && q2.shape == N.shape && all_same(q1.data, e1) && all_same(q2.data, e2), "scalar div r=%zu c=%zu", r, c);
        // row-vector broadcast still takes the general path: {r,c} + {c}
        auto rowd = make_data<float>(c, 31, false);
        Matrix<float> R(rowd, shape_t{c});
        auto rb = A + R;
        CHECK(rb.shape == A.shape, "row broadcast shape");
        std::vector<float> er(r * c);
        for (size_t i = 0; i < r; ++i) for (size_t j = 0; j < c; ++j) er[i * c + j] = (float)(d[i * c + j] + rowd[j]);
        CHECK(all_same(rb.data, er), "row broadcast values r=%zu c=%zu", r, c);
        if (r == 3 && c == 64) { emit("broadcast_scalar_mul", (A * S1).data); emit("broadcast_row_add", rb.data); }
    }
    // rank rules: a {n} vector against a {1,1} scalar is {1,n} (numpy), not {n}
    Matrix<float> v(std::vector<float>(5, 2.0f), shape_t{5});
    Matrix<float> s2(std::vector<float>{3.0f}, shape_t{1, 1});
    auto vr = v * s2;
    CHECK(vr.shape == (shape_t{1, 5}), "rank rule {5}*{1,1} -> {1,5}");
    // zero-size operand with a scalar
    Matrix<float> e(std::vector<float>{}, shape_t{0});
    Matrix<float> s1(std::vector<float>{3.0f}, shape_t{1});
    auto er0 = e * s1;
    CHECK(er0.data.empty(), "empty * scalar");
}

// ── 4. unary: exponent / maximum / sqrt / ln / pow / comparisons ────────────────────────────────────────────
template <class T> static void test_unary(const char* tn) {
    for (size_t n : kSizes) {
        auto a = make_data<T>(n, 21u + (uint32_t)n, std::is_floating_point_v<T>);
        Matrix<T> A(a, shape_t{n == 0 ? size_t(0) : size_t(1), n});
        std::vector<T> re(n), rmx(n), rsq(n), rp2(n), rp3(n), rp4(n);
        for (size_t i = 0; i < n; ++i) {
            re[i] = scalar::exp<T>(a[i]);
            rmx[i] = (a[i] < T(0)) ? T(0) : a[i];
            rsq[i] = scalar::sqrt<T>(a[i]);
            rp2[i] = (T)(a[i] * a[i]); rp3[i] = (T)(a[i] * a[i] * a[i]); rp4[i] = (T)(a[i] * a[i] * a[i] * a[i]);
        }
        CHECK(all_same(A.exponent().data, re), "%s exponent n=%zu", tn, n);
        CHECK(all_same(A.maximum(T(0)).data, rmx), "%s maximum n=%zu", tn, n);
        CHECK(all_same(A.sqrt().data, rsq), "%s sqrt n=%zu", tn, n);
        CHECK(all_same(vecmath::exponent(a), re), "%s vecmath::exponent n=%zu", tn, n);
        CHECK(all_same(vecmath::pow_s(a, T(2)), rp2) && all_same(vecmath::pow_s(a, T(3)), rp3) && all_same(vecmath::pow_s(a, T(4)), rp4), "%s pow_s n=%zu", tn, n);
        // comparisons
        auto gt = vecmath::cmp_gt(a, T(0)); auto le = vecmath::cmp_le(a, T(0));
        bool okc = gt.size() == n && le.size() == n;
        for (size_t i = 0; okc && i < n; ++i) okc = (gt[i] == (a[i] > T(0))) && (le[i] == (a[i] <= T(0)));
        CHECK(okc, "%s comparisons n=%zu", tn, n);
        if (n == 1000) { emit((std::string("exp_") + tn).c_str(), re); emit((std::string("max_") + tn).c_str(), rmx); }
    }
}

// ── 5. NaN scan ─────────────────────────────────────────────────────────────────────────────────────────────
static void test_nan_scan() {
    for (size_t n : kSizes) {
        if (n == 0) continue;
        std::vector<float> v(n, 1.5f);
        CHECK(!Matrix<float>::hasNaN(Matrix<float>(v, shape_t{n})), "no NaN n=%zu", n);
        for (size_t pos : {size_t(0), n / 2, n - 1}) {
            auto w = v; w[pos] = std::numeric_limits<float>::quiet_NaN();
            CHECK(Matrix<float>::hasNaN(Matrix<float>(w, shape_t{n})), "NaN at %zu of %zu", pos, n);
        }
        auto inf = v; inf[n - 1] = std::numeric_limits<float>::infinity();
        CHECK(!Matrix<float>::hasNaN(Matrix<float>(inf, shape_t{n})), "inf is not NaN n=%zu", n);
    }
}

// ── 6. autograd ops: forward + backward against scalar formulas, bit for bit ────────────────────────────────
template <class T> static void test_autograd(const char* tn) {
    for (size_t n : {size_t(1), size_t(7), size_t(64), size_t(1000), size_t(4099)}) {
        auto xd = make_data<T>(n, 3u + (uint32_t)n, true), yd = make_data<T>(n, 5u + (uint32_t)n, false), gd = make_data<T>(n, 8u + (uint32_t)n, false);
        for (auto& v : xd) if (std::isnan(v)) v = T(0.25);           // a NaN input is a NaN output; tested above, not through backward()
        shape_t sh{n};
        auto X = make_tensor<T>(Matrix<T>(xd, sh)), Y = make_tensor<T>(Matrix<T>(yd, sh));
        Matrix<T> G(gd, sh);

        // sigmoid
        {
            auto out = X->sigmoid();
            std::vector<T> s(n), gin(n);
            for (size_t i = 0; i < n; ++i) {
                T neg = (T)(T(-1) * xd[i]); T e = scalar::exp<T>(neg); T d = (T)(e + T(1)); s[i] = (T)(T(1) / d);
                T m1 = (T)(T(-1) * s[i]); T m2 = (T)(T(1) + m1); T temp = (T)(s[i] * m2);
                gin[i] = (T)(gd[i] * temp);
            }
            CHECK(all_same(out->val.data, s), "%s sigmoid fwd n=%zu", tn, n);
            out->backward(G);
            CHECK(all_same(X->grad.data, gin), "%s sigmoid bwd n=%zu", tn, n);
            if (n == 1000) { emit((std::string("sigmoid_fwd_") + tn).c_str(), out->val.data); emit((std::string("sigmoid_bwd_") + tn).c_str(), X->grad.data); }
            X->zero_grad();
        }
        // relu
        {
            auto out = X->relu();
            std::vector<T> r(n), gin(n);
            for (size_t i = 0; i < n; ++i) { r[i] = (xd[i] < T(0)) ? T(0) : xd[i]; gin[i] = (T)(gd[i] * ((xd[i] > T(0)) ? T(1) : T(0))); }
            CHECK(all_same(out->val.data, r), "%s relu fwd n=%zu", tn, n);
            out->backward(G);
            CHECK(all_same(X->grad.data, gin), "%s relu bwd n=%zu", tn, n);
            if (n == 1000) { emit((std::string("relu_bwd_") + tn).c_str(), X->grad.data); }
            X->zero_grad();
        }
        // add, mul (same shape), two contributions accumulate into the same leaf
        {
            auto out = X + Y;
            std::vector<T> f(n);
            for (size_t i = 0; i < n; ++i) f[i] = (T)(xd[i] + yd[i]);
            CHECK(all_same(out->val.data, f), "%s add fwd n=%zu", tn, n);
            out->backward(G);
            CHECK(all_same(X->grad.data, gd) && all_same(Y->grad.data, gd), "%s add bwd n=%zu", tn, n);
            X->zero_grad(); Y->zero_grad();
            auto out2 = X * Y;
            std::vector<T> f2(n), gx(n), gy(n);
            for (size_t i = 0; i < n; ++i) { f2[i] = (T)(xd[i] * yd[i]); gx[i] = (T)(gd[i] * yd[i]); gy[i] = (T)(gd[i] * xd[i]); }
            CHECK(all_same(out2->val.data, f2), "%s mul fwd n=%zu", tn, n);
            out2->backward(G);
            CHECK(all_same(X->grad.data, gx) && all_same(Y->grad.data, gy), "%s mul bwd n=%zu", tn, n);
            X->zero_grad(); Y->zero_grad();
            // x used twice: x*x + x -> gradient accumulates two contributions plus one
            auto out3 = (X * X) + X;
            out3->backward(G);
            std::vector<T> g3(n);
            for (size_t i = 0; i < n; ++i) { T a1 = (T)(gd[i] * xd[i]); T a2 = (T)(gd[i] * xd[i]); T acc = (T)(a1 + a2); g3[i] = (T)(acc + gd[i]); }
            // contribution ORDER is the graph's business; the sum of exactly three terms must match to rounding
            bool near = true;
            for (size_t i = 0; i < n && near; ++i) { double d = std::fabs((double)X->grad.data[i] - (double)g3[i]); near = d <= 1e-4 * (1.0 + std::fabs((double)g3[i])) || std::isnan(d); }
            CHECK(near, "%s x*x+x grad n=%zu", tn, n);
            emit((std::string("shared_leaf_grad_") + tn).c_str(), X->grad.data);
            X->zero_grad();
        }
    }
    // an infinite gradient through relu / sigmoid: relu multiplies by the mask, so inf * 0 = NaN must survive
    {
        std::vector<T> xd = {T(1), T(-1), T(0.5), T(-0.5), T(0), T(2), T(-2), T(3)};
        std::vector<T> gd = {std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity(), T(1), T(1), std::numeric_limits<T>::infinity(), T(1), T(1), -std::numeric_limits<T>::infinity()};
        Matrix<T> Gm(gd, shape_t{8});
        // through the op classes (they exist before and after the fused kernels); backward() of a graph would reject the NaN
        auto Xr = make_tensor<T>(Matrix<T>(xd, shape_t{8}));
        auto relu_op = std::make_shared<ReluOperation<T>>(Xr); auto relu_out = relu_op->forward(); relu_op->backward(Gm);
        auto Xs = make_tensor<T>(Matrix<T>(xd, shape_t{8}));
        auto sig_op = std::make_shared<SigmoidOperation<T>>(Xs); auto sig_out = sig_op->forward(); sig_op->backward(Gm);
        Matrix<T> rb = Xr->grad, sb = Xs->grad;
        std::vector<T> rref(8), sref(8);
        for (size_t i = 0; i < 8; ++i) {
            rref[i] = (T)(gd[i] * ((xd[i] > T(0)) ? T(1) : T(0)));
            T s = (T)(T(1) / (T)(scalar::exp<T>((T)(T(-1) * xd[i])) + T(1)));
            T m1 = (T)(T(-1) * s), m2 = (T)(T(1) + m1); sref[i] = (T)(gd[i] * (T)(s * m2));
        }
        CHECK(all_same(rb.data, rref), "%s relu_backward with infinite gradients", tn);
        CHECK(all_same(sb.data, sref), "%s sigmoid_backward with infinite gradients", tn);
        CHECK(std::isnan(rb.data[1]) && std::isnan(rb.data[4]), "%s inf * 0 stays NaN in relu_backward", tn);
        emit((std::string("relu_bwd_inf_") + tn).c_str(), rb.data);
    }
    // bias-style broadcast: [r,c] + [c], gradient of the bias is the column sum
    {
        const size_t r = 6, c = 33;
        auto xd = make_data<T>(r * c, 42, false), bd = make_data<T>(c, 43, false), gd = make_data<T>(r * c, 44, false);
        auto X = make_tensor<T>(Matrix<T>(xd, shape_t{r, c})); auto B = make_tensor<T>(Matrix<T>(bd, shape_t{c}));
        auto out = X + B; out->backward(Matrix<T>(gd, shape_t{r, c}));
        emit((std::string("bias_bcast_fwd_") + tn).c_str(), out->val.data);
        emit((std::string("bias_bcast_grad_") + tn).c_str(), B->grad.data);
        CHECK(B->grad.shape == (shape_t{c}) && X->grad.shape == (shape_t{r, c}), "%s bias grad shapes", tn);
    }
}

// ── 6b. matmul gradients: the transposed-BLAS fast path against explicit transposes, bit for bit ───────────────
template <class T> static void test_matmul_grads(const char* tn) {
    struct Case { shape_t a, b, g; const char* what; };
    const Case cases[] = {
        {{32, 784}, {784, 128}, {32, 128}, "mlp layer"},
        {{7, 13}, {13, 5}, {7, 5}, "odd sizes"},
        {{1, 9}, {9, 1}, {1, 1}, "1xKx1"},
        {{16, 16}, {16, 16}, {16, 16}, "square"},
        {{9}, {9, 4}, {4}, "A is 1-D"},
        {{6, 9}, {9}, {6}, "B is 1-D"},
        {{3, 6, 9}, {9, 4}, {3, 6, 4}, "batched A, shared B"},
        {{6, 9}, {3, 9, 4}, {3, 6, 4}, "shared A, batched B"},
    };
    uint32_t seed = 100;
    for (const Case& c : cases) {
        auto ad = make_data<T>(mxd::numel(c.a), seed++, false), bd = make_data<T>(mxd::numel(c.b), seed++, false), gd = make_data<T>(mxd::numel(c.g), seed++, false);
        Matrix<T> A(ad, c.a), B(bd, c.b), G(gd, c.g);
        auto got = MatmulOperation<T>::matmul_grads(A, B, G);
        // reference: the formulas written with explicit transposes (what the code did before the fast path)
        Matrix<T> A2 = A, B2 = B;
        if (A.shape.size() == 1) A2 = A.reshape({1, A.shape[0]});
        if (B.shape.size() == 1) B2 = B.reshape({B.shape[0], 1});
        shape_t gs = G.shape;
        if (A.shape.size() == 1) gs.insert(gs.end() - (B.shape.size() == 1 ? 0 : 1), 1);
        if (B.shape.size() == 1) gs.push_back(1);
        Matrix<T> G2 = G.reshape(gs);
        Matrix<T> rA = sumGradForBroadcast(G2.matmul(mxd::swap_last2(B2)), A2.shape).reshape(A.shape);
        Matrix<T> rB = sumGradForBroadcast(mxd::swap_last2(A2).matmul(G2), B2.shape).reshape(B.shape);
        CHECK(got.first.shape == A.shape && got.second.shape == B.shape, "%s matmul_grads shapes (%s)", tn, c.what);
        CHECK(all_same(got.first.data, rA.data), "%s matmul gA bit-identical (%s)", tn, c.what);
        CHECK(all_same(got.second.data, rB.data), "%s matmul gB bit-identical (%s)", tn, c.what);
        emit((std::string("matmul_gA_") + tn + "_" + c.what).c_str(), got.first.data);
        emit((std::string("matmul_gB_") + tn + "_" + c.what).c_str(), got.second.data);
    }
    // through the autograd graph: accumulating into leaves that already hold a gradient
    {
        auto X = make_tensor<T>(Matrix<T>(make_data<T>(8 * 12, 7, false), shape_t{8, 12}));
        auto W = make_tensor<T>(Matrix<T>(make_data<T>(12 * 5, 8, false), shape_t{12, 5}));
        Matrix<T> G(make_data<T>(8 * 5, 9, false), shape_t{8, 5});
        auto out = X->matmul(W); out->backward(G); out->backward(G);          // second call adds a second contribution
        emit((std::string("matmul_graph_gX_") + tn).c_str(), X->grad.data);
        emit((std::string("matmul_graph_gW_") + tn).c_str(), W->grad.data);
    }
}

// ── 6c. inputs that do not require a gradient: skipped by the ops, but nobody may notice ──────────────────────
// For each binary op: run it with `a` requiring a gradient and with `a` not requiring one. `b`'s gradient must be
// identical in both runs, and the non-grad input must end with NO gradient (as before the skip existed).
template <class T> static void test_requires_grad_skip(const char* tn) {
    const shape_t sa{8, 12}, sb{12, 5};
    for (int op = 0; op < 4; ++op) {                       // 0 matmul, 1 mul, 2 sub, 3 div
        const char* names[] = {"matmul", "mul", "sub", "div"};
        const bool mm = op == 0;
        const shape_t A_shape = mm ? sa : shape_t{8, 12}, B_shape = mm ? sb : shape_t{8, 12};
        auto ad = make_data<T>(mxd::numel(A_shape), 301u + (uint32_t)op, false, true), bd = make_data<T>(mxd::numel(B_shape), 401u + (uint32_t)op, false, true);
        const shape_t out_shape = mm ? shape_t{8, 5} : shape_t{8, 12};
        Matrix<T> G(make_data<T>(mxd::numel(out_shape), 501u + (uint32_t)op, false), out_shape);
        auto run = [&](bool a_grad, bool b_grad, Matrix<T>& ga, Matrix<T>& gb) {
            auto A = make_tensor<T>(Matrix<T>(ad, A_shape)); auto B = make_tensor<T>(Matrix<T>(bd, B_shape));
            A->requires_grad = a_grad; B->requires_grad = b_grad;
            Tensor_t<T> out = op == 0 ? A->matmul(B) : op == 1 ? A * B : op == 2 ? A - B : A / B;
            out->backward(G);
            ga = A->grad; gb = B->grad;
        };
        Matrix<T> ga_all, gb_all, ga_a, gb_a, ga_b, gb_b;
        run(true, true, ga_all, gb_all);
        run(false, true, ga_a, gb_a);                      // `a` is data
        run(true, false, ga_b, gb_b);                      // `b` is data
        CHECK(ga_a.get_size() == 0, "%s %s: a without requires_grad must have no gradient", tn, names[op]);
        CHECK(gb_b.get_size() == 0, "%s %s: b without requires_grad must have no gradient", tn, names[op]);
        CHECK(all_same(gb_a.data, gb_all.data) && gb_a.shape == gb_all.shape, "%s %s: b's gradient unchanged when a is data", tn, names[op]);
        CHECK(all_same(ga_b.data, ga_all.data) && ga_b.shape == ga_all.shape, "%s %s: a's gradient unchanged when b is data", tn, names[op]);
        emit((std::string("skip_") + names[op] + "_ga_" + tn).c_str(), ga_all.data);
        emit((std::string("skip_") + names[op] + "_gb_" + tn).c_str(), gb_all.data);
    }
#ifndef EQUIV_OLD_CODE
    // matmul_grads with the need flags: the skipped side is empty, the other side is bit-identical
    {
        Matrix<T> A(make_data<T>(8 * 12, 1, false), shape_t{8, 12}), B(make_data<T>(12 * 5, 2, false), shape_t{12, 5}), G(make_data<T>(8 * 5, 3, false), shape_t{8, 5});
        auto full = MatmulOperation<T>::matmul_grads(A, B, G);
        auto no_a = MatmulOperation<T>::matmul_grads(A, B, G, false, true);
        auto no_b = MatmulOperation<T>::matmul_grads(A, B, G, true, false);
        CHECK(no_a.first.get_size() == 0 && all_same(no_a.second.data, full.second.data), "%s matmul_grads need_a=false", tn);
        CHECK(no_b.second.get_size() == 0 && all_same(no_b.first.data, full.first.data), "%s matmul_grads need_b=false", tn);
    }
#endif
}

// ── 7. a small training loop: the whole stack, loss + parameters must be unchanged ──────────────────────────
static void test_training_loop() {
    const size_t B = 16, I = 24, H = 32, O = 4;
    auto w1 = make_tensor<float>(Matrix<float>::randu(-0.3f, 0.3f, {I, H}, 1u));
    auto w2 = make_tensor<float>(Matrix<float>::randu(-0.3f, 0.3f, {H, O}, 2u));
    auto x  = make_tensor<float>(Matrix<float>::randu(-1.0f, 1.0f, {B, I}, 3u)); x->requires_grad = false;
    auto t  = make_tensor<float>(Matrix<float>::randu(-1.0f, 1.0f, {B, O}, 4u)); t->requires_grad = false;
    Matrix<float> seed(std::vector<float>{1.0f}, shape_t{1});
    std::vector<float> losses;
    for (int step = 0; step < 20; ++step) {
        auto h = x->matmul(w1)->relu();
        auto y = (h->matmul(w2))->sigmoid();
        auto d = y - t;
        auto loss = (d * d)->sum();
        losses.push_back(loss->val.data[0]);
        loss->backward(seed);
        for (auto* p : {w1.get(), w2.get()}) {
            for (size_t i = 0; i < p->val.data.size(); ++i) p->val.data[i] = (float)(p->val.data[i] - 0.05f * p->grad.data[i]);
        }
        w1->zero_grad(); w2->zero_grad();
    }
    emit("train_losses", losses);
    emit("train_w1", w1->val.data);
    emit("train_w2", w2->val.data);
    CHECK(losses.back() < losses.front(), "training loop reduces the loss (%g -> %g)", losses.front(), losses.back());
}

int main() {
    test_binary<float>("f32", true);
    test_binary<double>("f64", true);
    test_binary<int32_t>("i32", false);
    test_binary<float16>("f16", true);
    test_binary<FP8<4, 3>>("fp8", true);
    test_exceptions();
    test_broadcast();
    test_unary<float>("f32");
    test_unary<double>("f64");
    test_unary<float16>("f16");
    test_unary<FP8<4, 3>>("fp8");
    test_nan_scan();
    test_autograd<float>("f32");
    test_autograd<double>("f64");
    test_matmul_grads<float>("f32");
    test_matmul_grads<double>("f64");
    test_matmul_grads<float16>("f16");
    test_requires_grad_skip<float>("f32");
    test_requires_grad_skip<double>("f64");
    test_training_loop();
    std::printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
