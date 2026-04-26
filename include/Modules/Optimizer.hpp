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
        ADAMw
    };

    template <typename T>
    class Optimizer{
        protected:
            Optimizer_t optimizer;
            T lr;
            std::vector<Tensor_t<T>> parameters;

            // Adam parameters
            T t = 0;
            Matrix<T> v0;
            Matrix<T> m0;
            bool initialized = false;
            std::vector<Matrix<T>> m; 
            std::vector<Matrix<T>> v; 
            T b1 = 0.9;
            T b2 = 0.999;
            T eps = 1e-8;

            // AdamW
            T lambda = 1e-4;

            void sgd(){
                for(auto p : parameters)
                {
                    p->data = p->data - this->lr * p->grad;
                }
            }

            void Adam()
            {
                if(!initialized){
                    for(auto p : parameters){
                        m.push_back(Matrix<T>::zeros(p->data.shape));
                        v.push_back(Matrix<T>::zeros(p->data.shape));
                    }

                    initialized = true;
                }

                t++;

                for(int i = 0; i < parameters.size(); i++){
                    auto p = parameters[i];
                    m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    v[i] = b2 * v[i] + (1 - b2) * (p->grad^2);

                    auto m_hat = m[i] / (1 - pow(b1, t));
                    auto v_hat = v[i] / (1 - pow(b2, t));

                    p->data = p->data - (lr * m_hat) / (v_hat.sqrt() + eps);
                }
            }

            void AdamW()
            {
                if(!initialized){
                    for(auto p : parameters){
                        m.push_back(Matrix<T>::zeros(p->data.shape));
                        v.push_back(Matrix<T>::zeros(p->data.shape));
                    }

                    initialized = true;
                }

                t++;

                for(int i = 0; i < parameters.size(); i++){
                    auto p = parameters[i];
                    m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    v[i] = b2 * v[i] + (1 - b2) * (p->grad^2);

                    auto m_hat = m[i] / (1 - pow(b1, t));
                    auto v_hat = v[i] / (1 - pow(b2, t));

                    p->data = p->data * (1 - lr * lambda);  // weight decay first
                    p->data = p->data - (lr * m_hat) / (v_hat.sqrt() + eps);
                }
            }
        public:

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim){
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps, T lambda){
                if(optim == ADAM){
                    this->b1 = beta1;
                    this->b2 = beta2;
                    this->eps = eps;
                    this->lambda = lambda;
                }
                
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