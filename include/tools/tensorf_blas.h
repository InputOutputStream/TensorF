// BLAS backend selector. The only place that decides which cblas implementation TensorF uses.
//
//   -DTENSORF_USE_OPENBLAS   real OpenBLAS: #include <cblas.h> resolves to OpenBLAS's own header and
//                            the build must link libopenblas. Set by the Makefile / CMake for you.
//   (nothing defined)        the portable shim in tools/cblas_shim.h: plain loops, no library needed.
//   -DMATRIX_NO_BLAS         Matrix uses its own loops. The level-1 calls in the optimizer still go
//                            through whichever backend is selected above.
//
// Never put a file named cblas.h on the include path: it would shadow OpenBLAS's header.
#ifndef TENSORF_BLAS_H
#define TENSORF_BLAS_H

#include <string>

#if defined(TENSORF_USE_OPENBLAS)
    #include <cblas.h>
#else
    #include "cblas_shim.h"
#endif

namespace tensorf { namespace blas {

// "openblas", "shim" or "none" (none = Matrix GEMM/dot use portable loops, MATRIX_NO_BLAS).
inline const char* backend() {
#if defined(MATRIX_NO_BLAS)
    return "none";
#elif defined(TENSORF_USE_OPENBLAS)
    return "openblas";
#else
    return "shim";
#endif
}

// Human-readable detail. For OpenBLAS: build config, detected core and its thread count at runtime.
inline std::string describe() {
    std::string s = backend();
#if defined(TENSORF_USE_OPENBLAS)
    s += std::string(" [") + openblas_get_config() + ", core=" + openblas_get_corename()
       + ", threads=" + std::to_string(openblas_get_num_threads()) + "]";
#endif
    return s;
}

// Pins the BLAS thread count for a scope, then restores it. No-op for the shim (always one thread).
// OpenBLAS runs its own pthread pool, so anything that must measure or run single-threaded needs this.
class ScopedThreads {
#if defined(TENSORF_USE_OPENBLAS)
    int prev_;
public:
    explicit ScopedThreads(int n) : prev_(openblas_get_num_threads()) { openblas_set_num_threads(n < 1 ? 1 : n); }
    ~ScopedThreads() { openblas_set_num_threads(prev_); }
#else
public:
    explicit ScopedThreads(int) {}
#endif
    ScopedThreads(const ScopedThreads&) = delete;
    ScopedThreads& operator=(const ScopedThreads&) = delete;
};

}} // namespace tensorf::blas

#endif // TENSORF_BLAS_H
