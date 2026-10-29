cmake_minimum_required(VERSION 3.18)
project(TensorF LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)                  # never benchmark a Debug build
endif()

# Same include roots as the Makefile, so headers resolve exactly as in the native build.
set(TENSORF_INCLUDES
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include/core
  ${CMAKE_CURRENT_SOURCE_DIR}/include/nn
  ${CMAKE_CURRENT_SOURCE_DIR}/include/data
  ${CMAKE_CURRENT_SOURCE_DIR}/include/net
  ${CMAKE_CURRENT_SOURCE_DIR}/include/profiler
  ${CMAKE_CURRENT_SOURCE_DIR}/include/parallel
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tools)

# ── BLAS ─────────────────────────────────────────────────────────────────────
#   -DTENSORF_BLAS=AUTO      (default) native: OpenBLAS if found, else the portable shim
#                                      Android: build OpenBLAS from source for the target ABI
#   -DTENSORF_BLAS=OPENBLAS  require OpenBLAS (error if absent)
#   -DTENSORF_BLAS=SHIM      portable loops in include/tools/cblas_shim.h (slow, no library)
#   -DTENSORF_BLAS=NONE      Matrix uses its own loops (MATRIX_NO_BLAS)
# The choice reaches the code as -DTENSORF_USE_OPENBLAS (see include/tools/tensorf_blas.h).
#
# OpenBLAS location (native or Android):
#   -DTENSORF_OPENBLAS_ROOT=/path   an existing install (include/ + lib/); skips pkg-config and the source build
# Android-only OpenBLAS build knobs (built with the NDK compiler, as a static lib, inside the build dir):
#   -DTENSORF_OPENBLAS_VERSION=0.3.30        -DTENSORF_OPENBLAS_TARGET=ARMV8 (e.g. CORTEXA76, CORTEXX1)
#   -DTENSORF_OPENBLAS_DYNAMIC_ARCH=ON       pick the kernel per CPU at run time (bigger, slower to build)
#   -DTENSORF_OPENBLAS_NUM_THREADS=8         max BLAS threads compiled in;  -DTENSORF_OPENBLAS_THREADS=OFF for single-thread
set(TENSORF_BLAS "AUTO" CACHE STRING "BLAS backend: AUTO | OPENBLAS | SHIM | NONE")
set_property(CACHE TENSORF_BLAS PROPERTY STRINGS AUTO OPENBLAS SHIM NONE)
set(TENSORF_OPENBLAS_ROOT "" CACHE PATH "Existing OpenBLAS install (include/ + lib/)")
set(TENSORF_OPENBLAS_VERSION "0.3.30" CACHE STRING "OpenBLAS version built for Android")
set(TENSORF_OPENBLAS_TARGET "" CACHE STRING "OpenBLAS TARGET for the Android build (empty = per-ABI default)")
option(TENSORF_OPENBLAS_DYNAMIC_ARCH "Android OpenBLAS: DYNAMIC_ARCH runtime kernel selection" OFF)
option(TENSORF_OPENBLAS_THREADS "Android OpenBLAS: multithreaded" ON)
set(TENSORF_OPENBLAS_NUM_THREADS "8" CACHE STRING "Android OpenBLAS: max threads compiled in")

# Legacy switch (older scripts): ON -> OPENBLAS, OFF -> SHIM, only when TENSORF_BLAS was left at AUTO.
if(DEFINED TENSORF_USE_OPENBLAS AND TENSORF_BLAS STREQUAL "AUTO")
  if(TENSORF_USE_OPENBLAS)
    set(TENSORF_BLAS "OPENBLAS")
  else()
    set(TENSORF_BLAS "SHIM")
  endif()
endif()
string(TOUPPER "${TENSORF_BLAS}" _blas)

# A file named cblas.h on the include path shadows OpenBLAS's real header (this was the original bug).
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/cblas.h)
  message(FATAL_ERROR "include/tools/cblas.h shadows the real OpenBLAS <cblas.h>. "
                      "The shim is now include/tools/cblas_shim.h; remove or rename cblas.h.")
endif()

add_library(tensorf_deps INTERFACE)
target_include_directories(tensorf_deps BEFORE INTERFACE ${TENSORF_INCLUDES})

# --- native: find an installed OpenBLAS -------------------------------------
function(tensorf_find_native_openblas out_target)
  set(${out_target} "" PARENT_SCOPE)
  if(TENSORF_OPENBLAS_ROOT)
    find_path(TF_OB_INC cblas.h HINTS ${TENSORF_OPENBLAS_ROOT}/include ${TENSORF_OPENBLAS_ROOT}/include/openblas NO_DEFAULT_PATH)
    find_library(TF_OB_LIB NAMES openblas HINTS ${TENSORF_OPENBLAS_ROOT}/lib ${TENSORF_OPENBLAS_ROOT}/lib64 NO_DEFAULT_PATH)
  else()
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
      pkg_check_modules(TF_OPENBLAS QUIET IMPORTED_TARGET openblas)
      if(TF_OPENBLAS_FOUND)
        set(${out_target} PkgConfig::TF_OPENBLAS PARENT_SCOPE)
        return()
      endif()
    endif()
    find_path(TF_OB_INC cblas.h PATH_SUFFIXES openblas openblas-pthread openblas-base)
    find_library(TF_OB_LIB NAMES openblas)
  endif()
  if(TF_OB_INC AND TF_OB_LIB)
    add_library(tensorf_openblas_native UNKNOWN IMPORTED)
    set_target_properties(tensorf_openblas_native PROPERTIES IMPORTED_LOCATION "${TF_OB_LIB}"
                          INTERFACE_INCLUDE_DIRECTORIES "${TF_OB_INC}")
    set(${out_target} tensorf_openblas_native PARENT_SCOPE)
  endif()
endfunction()

# --- Android: build OpenBLAS with the NDK compiler -----------------------------
function(tensorf_android_openblas out_target)
  # Prebuilt install given: use it as is.
  if(TENSORF_OPENBLAS_ROOT)
    find_path(TF_OB_INC cblas.h HINTS ${TENSORF_OPENBLAS_ROOT}/include ${TENSORF_OPENBLAS_ROOT}/include/openblas NO_DEFAULT_PATH)
    find_library(TF_OB_LIB NAMES libopenblas.a openblas HINTS ${TENSORF_OPENBLAS_ROOT}/lib NO_DEFAULT_PATH)
    if(NOT (TF_OB_INC AND TF_OB_LIB))
      message(FATAL_ERROR "TENSORF_OPENBLAS_ROOT=${TENSORF_OPENBLAS_ROOT}: no cblas.h / libopenblas found for ${ANDROID_ABI}")
    endif()
    add_library(tensorf_openblas_android UNKNOWN IMPORTED GLOBAL)
    set_target_properties(tensorf_openblas_android PROPERTIES IMPORTED_LOCATION "${TF_OB_LIB}"
                          INTERFACE_INCLUDE_DIRECTORIES "${TF_OB_INC}" INTERFACE_LINK_LIBRARIES "m")
    set(${out_target} tensorf_openblas_android PARENT_SCOPE)
    return()
  endif()

  # Per-ABI compiler wrapper triple and OpenBLAS target.
  if(ANDROID_ABI STREQUAL "arm64-v8a")
    set(_triple aarch64-linux-android)
    set(_def_target ARMV8)
    set(_extra BINARY=64)
  elseif(ANDROID_ABI STREQUAL "armeabi-v7a")
    set(_triple armv7a-linux-androideabi)
    set(_def_target ARMV7)
    set(_extra BINARY=32 ARM_SOFTFP_ABI=1)
  elseif(ANDROID_ABI STREQUAL "x86_64")
    set(_triple x86_64-linux-android)
    set(_def_target NEHALEM)                  # SSE4.2: runs on any emulator
    set(_extra BINARY=64)
  else()
    message(FATAL_ERROR "OpenBLAS for Android ABI '${ANDROID_ABI}' is not supported here. Use -DTENSORF_BLAS=SHIM.")
  endif()
  if(TENSORF_OPENBLAS_TARGET)
    set(_target ${TENSORF_OPENBLAS_TARGET})
  else()
    set(_target ${_def_target})
  endif()

  if(DEFINED ANDROID_PLATFORM_LEVEL)
    set(_api ${ANDROID_PLATFORM_LEVEL})
  else()
    set(_api ${CMAKE_SYSTEM_VERSION})
  endif()
  get_filename_component(_ndk_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
  set(_cc "${_ndk_bin}/${_triple}${_api}-clang")
  if(NOT EXISTS "${_cc}")
    message(FATAL_ERROR "NDK C compiler wrapper not found: ${_cc}")
  endif()
  find_program(TENSORF_HOSTCC NAMES gcc cc clang REQUIRED)       # OpenBLAS compiles a small helper (getarch) for the build host
  find_program(TENSORF_HOST_MAKE NAMES gmake make REQUIRED)
  cmake_host_system_information(RESULT _jobs QUERY NUMBER_OF_LOGICAL_CORES)

  set(_args CC=${_cc} HOSTCC=${TENSORF_HOSTCC} AR=${CMAKE_AR} RANLIB=${CMAKE_RANLIB}
            TARGET=${_target} ${_extra}
            NOFORTRAN=1 NO_LAPACK=1 ONLY_CBLAS=1 NO_SHARED=1 NO_AFFINITY=1 USE_OPENMP=0
            NUM_THREADS=${TENSORF_OPENBLAS_NUM_THREADS})
  if(TENSORF_OPENBLAS_THREADS)
    list(APPEND _args USE_THREAD=1)
  else()
    list(APPEND _args USE_THREAD=0 USE_LOCKING=1)
  endif()
  if(TENSORF_OPENBLAS_DYNAMIC_ARCH)
    list(APPEND _args DYNAMIC_ARCH=1)
  endif()

  include(ExternalProject)
  set(_ts "")
  if(NOT CMAKE_VERSION VERSION_LESS 3.24)
    set(_ts DOWNLOAD_EXTRACT_TIMESTAMP TRUE)       # silences policy CMP0135
  endif()
  set(_prefix ${CMAKE_BINARY_DIR}/openblas-${ANDROID_ABI})
  set(_url_base https://github.com/OpenMathLib/OpenBLAS/releases/download/v${TENSORF_OPENBLAS_VERSION})
  set(_hash "")
  if(TENSORF_OPENBLAS_VERSION STREQUAL "0.3.30")
    set(_hash URL_HASH SHA256=27342cff518646afb4c2b976d809102e368957974c250a25ccc965e53063c95d)
  endif()
  ExternalProject_Add(tensorf_openblas_build
    URL ${_url_base}/OpenBLAS-${TENSORF_OPENBLAS_VERSION}.tar.gz ${_hash}
    ${_ts}
    PREFIX ${_prefix}
    BUILD_IN_SOURCE TRUE
    CONFIGURE_COMMAND ""
    BUILD_COMMAND   ${TENSORF_HOST_MAKE} -j${_jobs} ${_args}
    INSTALL_COMMAND ${TENSORF_HOST_MAKE} ${_args} PREFIX=${_prefix}/install install
    BUILD_BYPRODUCTS ${_prefix}/install/lib/libopenblas.a)

  # The imported target needs its include dir to exist at configure time; the build fills it.
  file(MAKE_DIRECTORY ${_prefix}/install/include)
  add_library(tensorf_openblas_android STATIC IMPORTED GLOBAL)
  set_target_properties(tensorf_openblas_android PROPERTIES
      IMPORTED_LOCATION ${_prefix}/install/lib/libopenblas.a
      INTERFACE_INCLUDE_DIRECTORIES ${_prefix}/install/include
      INTERFACE_LINK_LIBRARIES "m")
  add_dependencies(tensorf_openblas_android tensorf_openblas_build)
  set(${out_target} tensorf_openblas_android PARENT_SCOPE)
endfunction()

# --- choose ------------------------------------------------------------------
set(_ob_target "")
if(_blas STREQUAL "AUTO")
  if(ANDROID)
    set(_blas OPENBLAS)                       # no system OpenBLAS on a phone: build it
  else()
    tensorf_find_native_openblas(_ob_target)
    if(_ob_target)
      set(_blas OPENBLAS)
    else()
      message(WARNING "OpenBLAS not found: using the slow portable shim. Install libopenblas-dev, "
                      "or pass -DTENSORF_OPENBLAS_ROOT=... / -DTENSORF_BLAS=SHIM to silence this.")
      set(_blas SHIM)
    endif()
  endif()
endif()

if(_blas STREQUAL "OPENBLAS")
  if(NOT _ob_target)
    if(ANDROID)
      tensorf_android_openblas(_ob_target)
    else()
      tensorf_find_native_openblas(_ob_target)
    endif()
  endif()
  if(NOT _ob_target)
    message(FATAL_ERROR "TENSORF_BLAS=OPENBLAS but OpenBLAS was not found. Install libopenblas-dev "
                        "or pass -DTENSORF_OPENBLAS_ROOT=/path/to/OpenBLAS.")
  endif()
  target_link_libraries(tensorf_deps INTERFACE ${_ob_target})
  target_compile_definitions(tensorf_deps INTERFACE TENSORF_USE_OPENBLAS)
elseif(_blas STREQUAL "NONE")
  target_compile_definitions(tensorf_deps INTERFACE MATRIX_NO_BLAS)
elseif(NOT _blas STREQUAL "SHIM")
  message(FATAL_ERROR "TENSORF_BLAS must be AUTO, OPENBLAS, SHIM or NONE (got '${TENSORF_BLAS}')")
endif()
message(STATUS "TensorF BLAS backend: ${_blas}")

find_package(Threads REQUIRED)
target_link_libraries(tensorf_deps INTERFACE Threads::Threads)

set(TENSORF_OUT ${CMAKE_BINARY_DIR}/bin)
foreach(t blas_check gcntest spmmtest coratrain basic_tests algorithm_profiler_test memory_profiler_test profiler_memory_test device_benchmark_test profiler_advisor_test hyperparam_advisor_test profiler_memory_nohook_test hw_fingerprint_test profiler_capability_test profiler_probe_test profiler_workload_test )
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE tensorf_deps)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TENSORF_OUT})
endforeach()

# algorithm_profiler_test  gpt2                      scope_test
# backprop_test            hw_fingerprint_test       smollm
# basic_tests              lowprec_tests             spmmtest
# contract                 matrix_tests              support_tests
# coratrain                memory_profiler_test      tensorf-gateway
# device_benchmark_test    profiler_capability_test  transformer
# diff_ops                 profiler_probe_test
# gcntest                  profiler_workload_test


# echo "===  ==="; adb shell "$REMOTE/profiler_advisor_test"
# echo "===   ==="; adb shell "$REMOTE/hyperparam_advisor_test"
# echo "===  ==="; adb shell "$REMOTE/profiler_memory_test"
# echo "===  ==="; adb shell "$REMOTE/profiler_memory_nohook_test"
