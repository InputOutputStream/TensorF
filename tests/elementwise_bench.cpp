// elementwise_bench: where does the time go in TensorF's element-wise ops and autograd glue?
//
//   g++ -std=c++20 -O2 -Iinclude -Iinclude/core -Iinclude/tools tests/elementwise_bench.cpp -o bin/elementwise_bench -pthread
//   ./bin/elementwise_bench            (add -DTENSORF_USE_OPENBLAS + OpenBLAS flags to benchmark the real BLAS path)
//
// Each row is the median of several timed windows. "ns/elem" is per output element; "GB/s" counts the bytes an
// ideal implementation must move (2 reads + 1 write for a binary op) so a row can be compared with the memory roofline.
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "tools/tensorf_blas.h"
#include "DataStructures/Tensor.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <vector>

using Clock = std::chrono::steady_clock;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
template <class T> static void sink(const T& v) { asm volatile("" : : "r,m"(v) : "memory"); }

static double time_per_call(const std::function<void()>& f, double window = 0.08, int samples = 7) {
    for (int i = 0; i < 3; ++i) f();
    std::vector<double> v;
    for (int s = 0; s < samples; ++s) {
        long reps = 0; const double t0 = now(); double el = 0;
        do { f(); ++reps; el = now() - t0; } while (el < window);
        v.push_back(el / (double)reps);
    }
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static bool g_csv = false;
static void row(const char* name, size_t n, double sec, double bytes_per_elem) {
    const double ns = sec / (double)n * 1e9, gbs = bytes_per_elem * (double)n / sec / 1e9;
    if (g_csv) std::printf("%s,%zu,%.4f,%.3f\n", name, n, ns, gbs);
    else       std::printf("  %-34s n=%-8zu %9.2f us  %7.3f ns/elem  %7.2f GB/s\n", name, n, sec * 1e6, ns, gbs);
}

static Matrix<float> rnd(size_t r, size_t c) { return Matrix<float>::randu(-0.5f, 0.5f, {r, c}); }
static Matrix<float> rnd1(size_t n) { return Matrix<float>::randu(-0.5f, 0.5f, {n}); }

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) if (!std::strcmp(argv[i], "--csv")) g_csv = true;
    if (!g_csv) std::printf("BLAS backend: %s\n", tensorf::blas::backend());

    for (size_t n : {size_t(16384), size_t(1u << 22)}) {
        if (!g_csv) std::printf("\n[vector ops, n=%zu (%s)]\n", n, n < 100000 ? "cache resident" : "DRAM sized");
        std::vector<float> va(n, 1.5f), vb(n, 2.5f);
        Matrix<float> a = rnd1(n), b = rnd1(n), s(2.0f);
        Matrix<float> bnz = Matrix<float>::randu(0.5f, 1.5f, {n});          // divisor: never 0 (a random float hits exactly 0.0 about once in 2^24)
        row("vecmath::add (raw)", n, time_per_call([&] { auto r = vecmath::add(va, vb); sink(r[0]); }), 12);
        row("Matrix a + b", n, time_per_call([&] { Matrix<float> r = a + b; sink(r.data[0]); }), 12);
        row("Matrix a * b", n, time_per_call([&] { Matrix<float> r = a * b; sink(r.data[0]); }), 12);
        row("Matrix a * scalar-Matrix (broadcast)", n, time_per_call([&] { Matrix<float> r = a * s; sink(r.data[0]); }), 8);
        row("Matrix a / b", n, time_per_call([&] { Matrix<float> r = a / bnz; sink(r.data[0]); }), 12);
        row("Matrix a.exponent()", n, time_per_call([&] { Matrix<float> r = a.exponent(); sink(r.data[0]); }), 8);
        row("Matrix a += b", n, time_per_call([&] { Matrix<float> r = a; r += b; sink(r.data[0]); }), 16);
        row("Matrix copy (alloc+memcpy)", n, time_per_call([&] { Matrix<float> r = a; sink(r.data[0]); }), 8);
    }

    for (size_t n : {size_t(4096), size_t(16384), size_t(1u << 18)}) {
        if (!g_csv) std::printf("\n[tensor ops, n=%zu elements]\n", n);
        auto x = make_tensor<float>(rnd(n / 64, 64)); x->requires_grad = false;
        auto y = make_tensor<float>(rnd(n / 64, 64)); y->requires_grad = false;
        auto xg = make_tensor<float>(rnd(n / 64, 64));
        auto seed = make_tensor<float>(Matrix<float>(std::vector<float>(n, 1.0f), shape_t{n / 64, 64}));
        row("Tensor add fwd (no grad)", n, time_per_call([&] { auto r = x + y; sink(r->val.data[0]); }), 12);
        row("Tensor sigmoid fwd (no grad)", n, time_per_call([&] { auto r = x->sigmoid(); sink(r->val.data[0]); }), 8);
        row("Tensor relu fwd (no grad)", n, time_per_call([&] { auto r = x->relu(); sink(r->val.data[0]); }), 8);
        row("Tensor sigmoid fwd+bwd", n, time_per_call([&] { auto r = xg->sigmoid(); r->backward(seed->val); xg->zero_grad(); sink(r->val.data[0]); }), 20);
        row("Tensor relu fwd+bwd", n, time_per_call([&] { auto r = xg->relu(); r->backward(seed->val); xg->zero_grad(); sink(r->val.data[0]); }), 20);
        row("Tensor add fwd+bwd", n, time_per_call([&] { auto r = xg + y; r->backward(seed->val); xg->zero_grad(); sink(r->val.data[0]); }), 20);
        row("Tensor mul fwd+bwd", n, time_per_call([&] { auto r = xg * y; r->backward(seed->val); xg->zero_grad(); sink(r->val.data[0]); }), 28);
    }

    if (!g_csv) std::printf("\n[reference steps, the probe cases]\n");
    {   // rnn B=32 h=128 in=64 T=25 (forward)
        const size_t B = 32, in = 64, hid = 128, T = 25;
        auto Wx = make_tensor<float>(rnd(in, hid)); Wx->requires_grad = false;
        auto Wh = make_tensor<float>(rnd(hid, hid)); Wh->requires_grad = false;
        auto h0 = make_tensor<float>(rnd(B, hid)); h0->requires_grad = false;
        std::vector<Tensor_t<float>> xs; for (size_t t = 0; t < T; ++t) { xs.push_back(make_tensor<float>(rnd(B, in))); xs.back()->requires_grad = false; }
        row("rnn step B=32 h=128 T=25 (fwd)", B * hid * T, time_per_call([&] {
            auto h = h0; for (size_t t = 0; t < T; ++t) h = (xs[t]->matmul(Wx) + h->matmul(Wh))->sigmoid(); sink(h->val.data[0]); }), 1);
    }
    {   // rnn B=8 h=64 in=32 T=50
        const size_t B = 8, in = 32, hid = 64, T = 50;
        auto Wx = make_tensor<float>(rnd(in, hid)); Wx->requires_grad = false;
        auto Wh = make_tensor<float>(rnd(hid, hid)); Wh->requires_grad = false;
        auto h0 = make_tensor<float>(rnd(B, hid)); h0->requires_grad = false;
        std::vector<Tensor_t<float>> xs; for (size_t t = 0; t < T; ++t) { xs.push_back(make_tensor<float>(rnd(B, in))); xs.back()->requires_grad = false; }
        row("rnn step B=8 h=64 T=50 (fwd)", B * hid * T, time_per_call([&] {
            auto h = h0; for (size_t t = 0; t < T; ++t) h = (xs[t]->matmul(Wx) + h->matmul(Wh))->sigmoid(); sink(h->val.data[0]); }), 1);
    }
    for (size_t B : {size_t(32), size_t(128)}) {
        const size_t I = 784, H = B == 32 ? 128 : 256, O = 10;
        auto x = make_tensor<float>(rnd(B, I)); x->requires_grad = false;
        auto w1 = make_tensor<float>(Matrix<float>::randu(-0.05f, 0.05f, {I, H}));
        auto w2 = make_tensor<float>(Matrix<float>::randu(-0.05f, 0.05f, {H, O}));
        auto seed = make_tensor<float>(Matrix<float>(std::vector<float>{1.0f}, shape_t{1}));
        char nm[64]; std::snprintf(nm, sizeof nm, "mlp_train step B=%zu H=%zu", B, H);
        row(nm, B, time_per_call([&] {
            auto y = x->matmul(w1)->relu()->matmul(w2); auto loss = (y * y)->sum();
            loss->backward(seed->val); w1->zero_grad(); w2->zero_grad(); }), 1);
    }
    return 0;
}
