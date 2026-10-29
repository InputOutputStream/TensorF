// libtensorf.a : the non-template part of TensorF (build information; add other .cpp files next to it).
#include "tensorf_blas.h"
#include <string>

#ifndef TENSORF_VERSION
#define TENSORF_VERSION "unknown"
#endif
#ifndef TENSORF_BLAS_NAME
#define TENSORF_BLAS_NAME "unknown"
#endif

namespace tensorf {
// Version of the installed library, and the BLAS it was configured for (the program must be built the same way).
extern "C" const char* tensorf_version() { return TENSORF_VERSION; }
extern "C" const char* tensorf_configured_blas() { return TENSORF_BLAS_NAME; }
}  // namespace tensorf
