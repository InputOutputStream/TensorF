// Ordered (topological) backprop: correctness on shared subgraphs, no exponential blow-up,
// accumulate/zero_grad semantics, log clamp, mse reduction.
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>   

using D = double;
static int g_fail = 0, g_n = 0;
#define CHECK(c) do { g_n++; if (!(c)) { g_fail++; std::printf("FAIL line %d: %s\n", __LINE__, #c); } } while (0)
static bool near(D a, D b, D tol = 1e-9) { return std::fabs(a - b) <= tol * (1 + std::fabs(b)); }
static Tensor_t<D> sc(D v) { return make_tensor<D>(Matrix<D>(std::vector<D>{v}, shape_t{1})); }
static Tensor_t<D> mat(std::vector<D> v, shape_t s) { return make_tensor<D>(Matrix<D>(v, s)); }

int main() {
    // 1. diamond chain: y = y*a + y*b (a+b = 1) 60 times. A per-path walk needs 2^60 visits.
    {
        auto t0 = std::chrono::steady_clock::now();
        auto x = sc(1.5); auto a = sc(0.3), b = sc(0.7);
        auto y = x;
        for (int i = 0; i < 60; i++) y = y * a + y * b;
        y->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        CHECK(near(y->val.data[0], 1.5, 1e-9));
        CHECK(near(x->grad.data[0], 1.0, 1e-9));       // every layer has derivative a + b = 1
        CHECK(ms < 2000);
        // zero_grad over the same graph must also be linear
        auto t1 = std::chrono::steady_clock::now();
        y->zero_grad();
        double ms2 = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
        CHECK(x->grad.get_size() == 0); CHECK(a->grad.get_size() == 0); CHECK(y->grad.get_size() == 0);
        CHECK(ms2 < 2000);
        std::printf("diamond x60: backward %.1f ms, zero_grad %.1f ms\n", ms, ms2);
    }
    // 2. y = y + y, 20 times: dy/dx = 2^20 (both contributions of a shared node must be summed)
    {
        auto x = sc(1.0); auto y = x;
        for (int i = 0; i < 20; i++) y = y + y;
        y->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(x->grad.data[0], 1048576.0));
    }
    // 3. leaf used many times: loss = sum(x*x + x) -> d/dx = 2x + 1
    {
        auto x = mat({1, 2, 3}, {3});
        auto loss = (x * x + x)->sum();
        loss->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(x->grad.data[0], 3) && near(x->grad.data[1], 5) && near(x->grad.data[2], 7));
    }
    // 4. no zero_grad -> contributions add up; zero_grad -> same result again
    {
        auto x = mat({1, 2}, {2});
        auto loss = (x * x)->sum();
        Matrix<D> one(std::vector<D>{1.0}, shape_t{1});
        loss->backward(one);
        D g0 = x->grad.data[0];
        loss->backward(one);
        CHECK(near(x->grad.data[0], 2 * g0));
        loss->zero_grad();
        CHECK(x->grad.get_size() == 0);
        loss->backward(one);
        CHECK(near(x->grad.data[0], g0));
    }
    // 5. requires_grad == false leaf receives nothing; the other one still does
    {
        auto x = sc(2.0), k = sc(5.0); k->requires_grad = false;
        auto y = x * k;
        y->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(x->grad.data[0], 5.0)); CHECK(k->grad.get_size() == 0);
    }
    // 6. concat: gradient reaches both inputs
    {
        auto a = mat({1, 2}, {2}), b = mat({3, 4, 5}, {3});
        auto c = Tensor<D>::concat({a, b}, 0);
        auto loss = (c * c)->sum();
        loss->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(a->grad.data[1], 4) && near(b->grad.data[2], 10));
    }
    // 7. ln: derivative is 1/x above the clamp and 0 at/below it (forward clamps at 1e-9)
    {
        auto x = mat({2.0, 0.0, -1.0, 0.5}, {4});
        auto y = x->ln()->sum();
        y->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(x->grad.data[0], 0.5)); CHECK(x->grad.data[1] == 0.0); CHECK(x->grad.data[2] == 0.0); CHECK(near(x->grad.data[3], 2.0));
        for (auto g : x->grad.data) CHECK(std::isfinite(g));
    }
    // 8. mse = mean over ALL elements; d/dpred = 2 (pred - true) / N
    {
        auto yt = mat({1, 2, 3, 4, 5, 6}, {2, 3}), yp = mat({2, 2, 5, 4, 3, 6}, {2, 3});
        auto loss = Tensor<D>::mse(yt, yp);
        CHECK(near(loss->val.data[0], (1 + 0 + 4 + 0 + 4 + 0) / 6.0));
        loss->backward(Matrix<D>(std::vector<D>{1.0}, shape_t{1}));
        CHECK(near(yp->grad.data[0], 2 * (2 - 1) / 6.0)); CHECK(near(yp->grad.data[2], 2 * (5 - 3) / 6.0)); CHECK(near(yp->grad.data[4], 2 * (3 - 5) / 6.0));
    }
    std::printf("%d/%d checks passed\n", g_n - g_fail, g_n);
    return g_fail ? 1 : 0;
}