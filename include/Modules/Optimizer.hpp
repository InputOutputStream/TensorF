#ifndef __OPTIMIZER__HPP_
#define __OPTIMIZER__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>
    #include <vector>
    #include <cmath>

    enum Optimizer_t{
        SGD,
        ADAM,
        // ADAMw
    };

    template <typename T>
    class Optimizer{
        protected:
            Optimizer_t optimizer;
            T lr;
            std::vector<Tensor_t<T>> parameters;

            // Adam parameters
            T t = 0;
            Matrix<T> m0;
            Matrix<T> v0;
            T b1 = 0.9;
            T b2 = 0.999;
            T eps = 1e-8;

            void sgd(){
                for(auto p : parameters)
                {
                    p->data = p->data - this->lr * p->grad;
                }
            }

            void Adam(){
                for(auto p : parameters)
                {
                    m0 = Matrix<T>::zeros(p->data.shape);
                    v0 = Matrix<T>::zeros(p->data.shape);
                    
                    t++;
                    m0 = b1 * m0 + (1 - b1) * p->grad;
                    v0 = b2 * v0 + (1 - b2) * (p->grad^2);
                    m0 = m0/(1 - pow(b1,t));
                    v0 = v0/(1 - pow(b2,t));

                    p->data = p->data - (lr*m0)/(v0.sqrt() + eps);
                }

                t=0;
            }

        public:

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim){
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps){
                if(optim == ADAM){
                    this->b1 = beta1;
                    this->b2 = beta2;
                    this->eps = eps;
                }
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            void zero_grad(){
                for(auto p : parameters)
                    p->zero_grad();
            }

            void step(){
                switch(this->optimizer){
                    case SGD:
                        this->sgd();
                        break;
                    case ADAM:
                        this->Adam();
                        break;
                }

            }
    };

#endif