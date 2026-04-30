#ifndef __TYPES__
#define __TYPES__

    #include <memory>
    #include <vector>
    #include <ctype.h>
    #include "fp8.hpp"

    using shape_t = std::vector<size_t>;

    template <typename T>
    class Tensor;

    template <typename T>
    class Operation;

    template <typename T>
    class Module;


    template <typename T>
    using Operation_t=std::shared_ptr<Operation<T>>;

    // template <typename T>
    // using module_t = std::shared_ptr<Module<T>>;

    template <typename T>
    using Tensor_t=std::shared_ptr<Tensor<T>>;

    typedef unsigned char int8;
    typedef unsigned short int int16;
    typedef unsigned int int32;
    typedef unsigned long long int int64;


    typedef unsigned char int8;
    typedef _Float16 float16;
    typedef _Float32 float32;
    typedef _Float64 float64;

    using fp8_e3m4 = FP8<3,4>;
    using fp8_e4m3 = FP8<4,3>;
    using fp8_e5m2 = FP8<5,2>;




#endif // !__TYPES__
