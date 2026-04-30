#ifndef __SOFTMAX__HPP_
#define __SOFTMAX__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    template <typename T>
    class Softmax : public Module<T>{

        public:
            Tensor_t<T> forward(Tensor_t<T> x){
               return x->softmax();
            }
    };

#endif