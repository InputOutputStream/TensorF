#ifndef __LINEAR__HPP_
#define __LINEAR__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>
    #include <memory>
    #include <vector>


    template <typename T>
    class Linear : Module<T>{
        protected:

        public:
            Tensor_t<T> weight;
            Tensor_t<T> bias;

            Linear(size_t in_features, size_t out_features){
                this->weight = make_tensor<T>(*Matrix<T>().random(in_features, out_features));
                this->bias = make_tensor<T>(*Matrix<T>().random(out_features));

                this->register_parameter(weight);
                this->register_parameter(bias);
            }

            Tensor_t<T> forward(Tensor_t<T> x){
               return x->matmul(weight) + bias;
            }

    };

#endif