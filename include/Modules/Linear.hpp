#ifndef __LINEAR__HPP_
#define __LINEAR__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>

    template <typename T>
    class Linear : public Module<T>{

        public:
            Tensor_t<T> weight = nullptr;
            Tensor_t<T> bias = nullptr;
            bool sbias;

            Linear(size_t in_features, size_t out_features, bool sbias = true) 
            {
                // Glorot 
                auto limit = std::sqrt((T)0.01 / (T)(in_features + out_features));
                this->weight = make_tensor<T>(Matrix<T>::randu(-limit, limit, {in_features, out_features}));
                this->register_parameter(this->weight);
                this->sbias = sbias;
                this->bias = make_tensor<T>(Matrix<T>::zeros({out_features}));
                if (sbias)
                    this->register_parameter(this->bias);
            }
            

            Tensor_t<T> forward(Tensor_t<T> x){
                Tensor_t<T> res;
                if(sbias)
                   {
                        res = x->matmul(weight) + bias;
                   }
                else
                    {
                        res = x->matmul(weight);
                    }
                // std::cerr << " Linear out val : "<< res->val<< "\n";
                return res;
            }

    };

#endif