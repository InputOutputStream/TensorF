


#ifndef __OPTIMIZER__HPP_
#define __OPTIMIZER__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>
    #include <memory>
    #include <vector>

    typedef enum Optimizer_t{
        SGD,
        ADAM,
        ADAMw
    };

    template <typename T>
    class Optimizer{
        protected:
            Optimizer_t optimizer = SGD;
            T lr;
            std::vector<Tensor_t<T>> parameters;

            void sgd(){
                for(auto p : parameters)
                {
                    p->data = p->data - this->lr * p->grad;
                }
            }

        public:

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim){
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            void step(){
                switch(this->optim){
                    case SGD:
                        this->sgd();
                        break;
                }

            }

    };

#endif