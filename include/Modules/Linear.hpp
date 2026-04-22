#ifndef __LINEAR__HPP_
#define __LINEAR__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>

    template <typename T>
    class Linear : public Module<T>{

        public:
            Tensor_t<T> weight;
            Tensor_t<T> bias;
            bool sbias;

            Linear(long in_features, long out_features, bool sbias = true){
                    this->weight = make_tensor<T>(Matrix<T>::random({in_features, out_features}));
                    this->register_parameter(weight);
                    this->sbias = sbias;

                    if(sbias)
                    {
                        this->bias = make_tensor<T>(Matrix<T>::zeros({out_features}));
                        this->register_parameter(bias);
                    }
                }

            Tensor_t<T> forward(Tensor_t<T> x){
                if(sbias)
                   {
                        return x->matmul(weight) + bias;
                    }
                else
                    {
                        return x->matmul(weight);
                    }
            }

    };

#endif