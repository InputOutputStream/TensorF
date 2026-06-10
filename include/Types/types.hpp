#pragma once

    #include <memory>
    #include <vector>
    #include <string>
    #include <ctype.h>
    #include <cmath>

    #include "fp8.hpp"
    #include "fp4.hpp"

    using shape_t = std::vector<size_t>;

    template <typename T>
    class Tensor;

    template <typename T>
    class GGUFLoader;

    template <typename T>
    class GPTGGUFLoader;
    
    template <typename T>
    class LlamaGGUFLoader;

    template <typename T>
    class Operation;

    template <typename T>
    class Module;


    template <typename T>
    using Operation_t=std::shared_ptr<Operation<T>>;

    template <typename T>
    using Tensor_t=std::shared_ptr<Tensor<T>>;

    typedef unsigned char int8;
    typedef unsigned short int int16;
    typedef unsigned int int32;
    typedef unsigned long long int int64;

    // ── Float types — fallback si le compilateur ne supporte pas _FloatXX ──
    #if defined(__FLT16_MAX__)
        typedef _Float16 float16;
    #else
        typedef uint16_t float16;   // stockage 16-bit, ops promues en float32
    #endif

    #if defined(__FLT32_MAX__) && defined(_Float32)
        typedef _Float32 float32;
    #else
        typedef float float32;
    #endif

    #if defined(__FLT64_MAX__) && defined(_Float64)
        typedef _Float64 float64;
    #else
        typedef double float64;
    #endif
    
    using fp8_e3m4 = FP8<3,4>;
    using fp8_e4m3 = FP8<4,3>;
    using fp8_e5m2 = FP8<5,2>;

    using fp4_e2m1 = FP4<2,1>;
    using fp4_e1m2 = FP4<1,2>;
    

    template<typename T> struct is_fp8 : std::false_type {};
    template<int E, int M> struct is_fp8<FP8<E,M>> : std::true_type {};

    template<typename T> struct is_fp4 : std::false_type {};
    template<unsigned short E, unsigned short M> struct is_fp4<FP4<E,M>> : std::true_type {};

