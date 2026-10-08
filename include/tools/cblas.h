// Minimal CBLAS replacement for platforms without OpenBLAS (Android NDK builds, quick ports).
// Implements exactly what TensorF calls: saxpy/daxpy, sscal/dscal, sdot/ddot, sgemm/dgemm,
// row-major only. Plain loops written so clang/gcc can auto-vectorise them (NEON on arm64).
// This is a portability shim, not a fast BLAS. For speed, build with OpenBLAS
// (-DTENSORF_USE_OPENBLAS=ON) or replace gemm below with a tiled/NEON kernel.
#ifndef TENSORF_CBLAS_SHIM_H
#define TENSORF_CBLAS_SHIM_H

#include <cstddef>
#include <cstdio>
#include <cstdlib>

enum CBLAS_ORDER     { CblasRowMajor = 101, CblasColMajor = 102 };
enum CBLAS_TRANSPOSE { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 };

namespace tensorf_cblas_shim {

template <typename T>
inline void axpy(int n, T a, const T* x, int incx, T* y, int incy) {
    if (incx == 1 && incy == 1) { for (int i = 0; i < n; i++) y[i] += a * x[i]; return; }
    for (int i = 0; i < n; i++) y[(size_t)i * incy] += a * x[(size_t)i * incx];
}
template <typename T>
inline void scal(int n, T a, T* x, int incx) {
    for (int i = 0; i < n; i++) x[(size_t)i * incx] *= a;
}
template <typename T>
inline T dot(int n, const T* x, int incx, const T* y, int incy) {
    T s = 0;
    for (int i = 0; i < n; i++) s += x[(size_t)i * incx] * y[(size_t)i * incy];
    return s;
}
// C(MxN) = alpha * op(A)(MxK) * op(B)(KxN) + beta * C, row-major.
template <typename T>
inline void gemm(CBLAS_ORDER order, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K,
                 T alpha, const T* A, int lda, const T* B, int ldb, T beta, T* C, int ldc) {
    if (order != CblasRowMajor) { std::fprintf(stderr, "cblas shim: only row-major supported\n"); std::abort(); }
    for (int i = 0; i < M; i++) {
        T* c = C + (size_t)i * ldc;
        if (beta == T(0)) for (int j = 0; j < N; j++) c[j] = 0;
        else if (beta != T(1)) for (int j = 0; j < N; j++) c[j] *= beta;
    }
    const bool aT = (ta != CblasNoTrans), bT = (tb != CblasNoTrans);
    for (int i = 0; i < M; i++) {
        T* c = C + (size_t)i * ldc;
        for (int k = 0; k < K; k++) {                       // i-k-j order: contiguous inner loop over j
            const T a = alpha * (aT ? A[(size_t)k * lda + i] : A[(size_t)i * lda + k]);
            if (a == T(0)) continue;
            if (!bT) { const T* b = B + (size_t)k * ldb; for (int j = 0; j < N; j++) c[j] += a * b[j]; }
            else     { for (int j = 0; j < N; j++) c[j] += a * B[(size_t)j * ldb + k]; }
        }
    }
}

} // namespace tensorf_cblas_shim

inline void cblas_saxpy(int n, float a, const float* x, int incx, float* y, int incy)    { tensorf_cblas_shim::axpy<float>(n, a, x, incx, y, incy); }
inline void cblas_daxpy(int n, double a, const double* x, int incx, double* y, int incy) { tensorf_cblas_shim::axpy<double>(n, a, x, incx, y, incy); }
inline void cblas_sscal(int n, float a, float* x, int incx)                               { tensorf_cblas_shim::scal<float>(n, a, x, incx); }
inline void cblas_dscal(int n, double a, double* x, int incx)                             { tensorf_cblas_shim::scal<double>(n, a, x, incx); }
inline float  cblas_sdot(int n, const float* x, int incx, const float* y, int incy)       { return tensorf_cblas_shim::dot<float>(n, x, incx, y, incy); }
inline double cblas_ddot(int n, const double* x, int incx, const double* y, int incy)     { return tensorf_cblas_shim::dot<double>(n, x, incx, y, incy); }
inline void cblas_sgemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K, float alpha,
                        const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc)
{ tensorf_cblas_shim::gemm<float>(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }
inline void cblas_dgemm(CBLAS_ORDER o, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int M, int N, int K, double alpha,
                        const double* A, int lda, const double* B, int ldb, double beta, double* C, int ldc)
{ tensorf_cblas_shim::gemm<double>(o, ta, tb, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc); }

#endif