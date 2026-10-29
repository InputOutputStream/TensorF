// blas_check: proves WHICH BLAS this binary really uses, that it is numerically right, and how fast it is.
//
//   blas_check                      print backend, verify results, benchmark sgemm/dgemm
//   blas_check --expect=openblas    exit 1 unless the OpenBLAS backend is compiled in (also: shim, none)
//   blas_check --n=1024             benchmark size (default 512)
//
// Works unchanged on PC and on an Android device (scripts/run_android.sh runs it with --expect).
#include "DataStructures/Matrix.hpp"
#include "tools/tensorf_blas.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c, ...) do { if (!(c)) { ++g_fail; std::printf("  [FAIL] " __VA_ARGS__); std::printf("\n"); } } while (0)

template <class T> static double max_rel_err(const std::vector<T>& a, const std::vector<T>& b) {
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i)
        m = std::max(m, std::fabs((double)a[i] - (double)b[i]) / std::max(1.0, std::fabs((double)b[i])));
    return m;
}

template <class T> static void naive(const std::vector<T>& A, const std::vector<T>& B, std::vector<double>& C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) for (size_t j = 0; j < N; ++j) {
        double s = 0; for (size_t k = 0; k < K; ++k) s += (double)A[i * K + k] * (double)B[k * N + j];
        C[i * N + j] = s;
    }
}

template <class T> static void gemm_case(const char* name, size_t M, size_t N, size_t K, double tol) {
    std::vector<T> A(M * K), B(K * N), C(M * N, T(0));
    for (size_t i = 0; i < A.size(); ++i) A[i] = T(std::sin(0.37 * (double)i));
    for (size_t i = 0; i < B.size(); ++i) B[i] = T(std::cos(0.11 * (double)i));
    std::vector<double> ref(M * N); naive(A, B, ref, M, N, K);
    mxd::gemm<T>(A.data(), B.data(), C.data(), M, N, K);
    std::vector<double> got(C.begin(), C.end());
    const double e = max_rel_err(got, ref);
    CHECK(e < tol, "%s %zux%zux%zu max rel err %.3g (tol %.1g)", name, M, N, K, e, tol);
}

template <class T> static double bench_gemm(size_t n) {
    std::vector<T> A(n * n, T(1.01)), B(n * n, T(0.99)), C(n * n);
    mxd::gemm<T>(A.data(), B.data(), C.data(), n, n, n);                       // warm-up (OpenBLAS spins up its pool here)
    int reps = 0; double el = 0;
    auto t0 = std::chrono::steady_clock::now();
    do { mxd::gemm<T>(A.data(), B.data(), C.data(), n, n, n); ++reps;
         el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (el < 1.0 && reps < 200);
    return 2.0 * (double)n * n * n * reps / el / 1e9;
}

int main(int argc, char** argv) {
    std::string expect; size_t n = 512;
    for (int i = 1; i < argc; ++i) {
        if (!std::strncmp(argv[i], "--expect=", 9)) expect = argv[i] + 9;
        else if (!std::strncmp(argv[i], "--n=", 4))  n = (size_t)std::atol(argv[i] + 4);
    }

    std::printf("BLAS backend : %s\n", tensorf::blas::describe().c_str());
#if defined(__aarch64__)
    std::printf("target arch  : aarch64\n");
#elif defined(__arm__)
    std::printf("target arch  : arm32\n");
#elif defined(__x86_64__)
    std::printf("target arch  : x86_64\n");
#endif

    if (!expect.empty() && expect != tensorf::blas::backend()) {
        std::printf("  [FAIL] expected backend '%s' but this binary uses '%s'\n", expect.c_str(), tensorf::blas::backend());
        return 1;
    }

    // Correctness: odd sizes exercise edge/remainder kernels; tolerance is generous for float accumulation order.
    gemm_case<float >("sgemm", 1, 1, 1, 1e-5);
    gemm_case<float >("sgemm", 7, 13, 5, 1e-4);
    gemm_case<float >("sgemm", 65, 33, 129, 1e-4);
    gemm_case<double>("dgemm", 7, 13, 5, 1e-12);
    gemm_case<double>("dgemm", 65, 33, 129, 1e-12);

    // Level-1 calls the optimizer uses.
    {
        const int len = 1001; std::vector<float> x(len), y(len), y0;
        for (int i = 0; i < len; ++i) { x[i] = 0.01f * i; y[i] = 1.0f - 0.002f * i; }
        y0 = y;
        cblas_saxpy(len, 0.5f, x.data(), 1, y.data(), 1);
        double e = 0; for (int i = 0; i < len; ++i) e = std::max(e, (double)std::fabs(y[i] - (y0[i] + 0.5f * x[i])));
        CHECK(e < 1e-4, "saxpy err %.3g", e);
        std::vector<float> z = x; cblas_sscal(len, 3.0f, z.data(), 1);
        e = 0; for (int i = 0; i < len; ++i) e = std::max(e, (double)std::fabs(z[i] - 3.0f * x[i]));
        CHECK(e < 1e-4, "sscal err %.3g", e);
        double ref = 0; for (int i = 0; i < len; ++i) ref += (double)x[i] * y0[i];
        const double d = cblas_sdot(len, x.data(), 1, y0.data(), 1);
        CHECK(std::fabs(d - ref) / std::max(1.0, std::fabs(ref)) < 1e-4, "sdot %.6g vs %.6g", d, ref);
    }

    std::printf("correctness  : %s\n", g_fail ? "FAILED" : "ok");
    std::printf("sgemm %zu^3  : %7.2f GFLOP/s\n", n, bench_gemm<float >(n));
    std::printf("dgemm %zu^3  : %7.2f GFLOP/s\n", n, bench_gemm<double>(n));
    return g_fail ? 1 : 0;
}
