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
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tools)

# BLAS. Default: the self-contained shim in include/tools/cblas.h (no OpenBLAS needed).
option(TENSORF_USE_OPENBLAS "Link a real OpenBLAS instead of the shim" OFF)
set(CBLAS_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/include/tools/cblas.h)

add_library(tensorf_deps INTERFACE)
target_include_directories(tensorf_deps BEFORE INTERFACE ${TENSORF_INCLUDES})
if(TENSORF_USE_OPENBLAS)
  find_library(OPENBLAS_LIB openblas REQUIRED)
  target_link_libraries(tensorf_deps INTERFACE ${OPENBLAS_LIB})
else()
  # include/tools/cblas.h must be the shim (it defines tensorf_cblas_shim), not OpenBLAS's real header
  set(_shim_ok FALSE)
  if(EXISTS ${CBLAS_HEADER})
    file(READ ${CBLAS_HEADER} _cblas_txt)
    string(FIND "${_cblas_txt}" "tensorf_cblas_shim" _pos)
    if(NOT _pos EQUAL -1)
      set(_shim_ok TRUE)
    endif()
  endif()
  if(NOT _shim_ok)
    message(FATAL_ERROR "include/tools/cblas.h is not the TensorF CBLAS shim. "
      "Replace it with the shim, or configure with -DTENSORF_USE_OPENBLAS=ON.")
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(tensorf_deps INTERFACE Threads::Threads)

set(TENSORF_OUT ${CMAKE_BINARY_DIR}/bin)
foreach(t gcntest spmmtest coratrain basic_tests)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE tensorf_deps)
  target_compile_options(${t} PRIVATE -Wall -Wextra -Wno-unused-parameter)
  set_target_properties(${t} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${TENSORF_OUT})
endforeach()