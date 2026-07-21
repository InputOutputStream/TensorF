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

            Linear(size_t out_features, size_t in_features, bool sbias = true) 
            {
                // Glorot 
                auto limit = std::sqrt((T)6.0 / (T)(in_features + out_features));
                this->weight = make_tensor<T>(Matrix<T>::randu(-limit, limit, {out_features, in_features}));
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
                        res = x->matmul(weight->transpose()) + bias;
                   }
                else
                    {
                        res = x->matmul(weight->transpose());
                    }
                // std::cerr << " Linear out val : "<< res->val<< "\n";
                return res;
            }

            // Weight/bias transfer used by GPT::load_backbone_from() when both the
            // pretrained and target model are dense (LinearT == Linear on both
            // sides). 
            void load_pretrained(Linear<T>& src) {
                weight->val.copy_from(src.weight->val);
                if (sbias && src.sbias)
                    bias->val.copy_from(src.bias->val);
            }

    friend class GGUFLoader<T>;
    friend class GPTGGUFLoader<T>; 
    friend class LlamaGGUFLoader<T>; 

};

#endif