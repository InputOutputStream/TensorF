#ifndef __RELU__HPP_
#define __RELU__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>
    #include <memory>
    #include <vector>


    template <typename T>
    class Relu {

        public:
            Tensor_t<T> forward(Tensor_t<T> x){
               return x->relu();
            }
    };

#endif