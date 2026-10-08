#pragma once

#include <memory>
#include <vector>
#include <string>
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <cmath>

#include "shape.hpp"
#include "scalar.hpp"   // pulls in fp8.hpp, fp4.hpp, float16, scalar::*

// H1: these four typedefs are (misleadingly) UNSIGNED. Kept for API compat.
//     Correctly-named aliases are provided right below.
typedef unsigned char          int8;
typedef unsigned short int     int16;
typedef unsigned int           int32;
typedef unsigned long long int int64;

// Correctly-named fixed-width aliases (built from <cstdint>).
using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;
using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

// H3: `defined(_Float32)` / `defined(_Float64)` were never true — types, not macros.
using float32 = float;
using float64 = double;

// (float16 / float16_is_native are defined in Types/scalar.hpp.)

// fp8 aliases
using fp8_e3m4 = FP8<3, 4>;
using fp8_e4m3 = FP8<4, 3>;
using fp8_e5m2 = FP8<5, 2>;

// fp4 aliases (fp4_e1m2 stays declared but fails when instantiated — see fp4.hpp).
using fp4_e2m1 = FP4<2, 1>;
using fp4_e1m2 = FP4<1, 2>;

// Legacy traits (kept for backward compatibility; the canonical ones are
// scalar::is_fp8_v / scalar::is_fp4_v).
template <typename T> struct is_fp8 : std::false_type {};
template <int E, int M> struct is_fp8<FP8<E, M>> : std::true_type {};

template <typename T> struct is_fp4 : std::false_type {};
template <unsigned short E, unsigned short M> struct is_fp4<FP4<E, M>> : std::true_type {};

// ─── forward declarations ────────────────────────────────────────────────────
template <typename T> class Tensor;
template <typename T> class Graph;
template <typename T> class CSR;
template <typename T> class GGUFLoader;
template <typename T, template<typename> class LinearT> class GPTGGUFLoader;
template <typename T, template<typename> class LinearT> class LlamaGGUFLoader;
template <typename T> class Operation;
template <typename T> class Module;

template <typename T> using Operation_t = std::shared_ptr<Operation<T>>;
template <typename T> using Tensor_t    = std::shared_ptr<Tensor<T>>;