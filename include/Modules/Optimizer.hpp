#ifndef __OPTIMIZER__HPP_
#define __OPTIMIZER__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

    #include <iostream>
    #include <vector>
    #include <cmath>
    #include <cblas.h>

    enum Optimizer_t{
        SGD,
        ADAM,
        ADAMw
    };

    template <typename T>
    class Optimizer{
        protected:
            Optimizer_t optimizer;
            T lr = 0.1;
            std::vector<Tensor_t<T>> parameters;

            // Adam parameters
            size_t t = 0;
            bool initialized = false;
            std::vector<Matrix<T>> m; 
            std::vector<Matrix<T>> v; 
            T b1 = 0.9;
            T b2 = 0.999;
            T eps = 1e-8;

            // AdamW
            T lambda = 1e-4;

            void sgd() {
                for(auto p : this->parameters) {
                    if(p->grad.get_size() == 0) continue;

                    if constexpr (std::is_same_v<T, float>)
                        cblas_saxpy(p->val.get_size(), -this->lr,
                            p->grad.data.data(), 1,
                            p->val.data.data(),  1);

                    else if constexpr (std::is_same_v<T, double>)
                        cblas_daxpy(p->val.get_size(), -this->lr,
                            p->grad.data.data(), 1,
                            p->val.data.data(),  1);
                    else
                        p->val = p->val - this->lr * p->grad;
                }
            }

            void Adam() {
                if(!initialized) {
                    for(auto p : parameters) {
                        if(p->grad.get_size() == 0) continue;
                        m.push_back(Matrix<T>::zeros(p->val.shape));
                        v.push_back(Matrix<T>::zeros(p->val.shape));
                    }
                    initialized = true;
                }

                t++;
                T b1_corr = 1 - std::pow(b1, t);   // bias correction scalars
                T b2_corr = 1 - std::pow(b2, t);

                for(size_t i = 0; i < parameters.size(); i++) {
                    auto p = parameters[i];
                    if(p->grad.get_size() == 0) continue;

                    size_t n = p->val.get_size();

                    // m = b1*m + (1-b1)*grad  →  saxpy: m = (1-b1)*grad + b1*m
                    // step 1: scale m by b1 in place
                    if constexpr (std::is_same_v<T, float>) {
                        cblas_sscal(n, b1, m[i].data.data(), 1);
                        cblas_saxpy(n, (1 - b1), p->grad.data.data(), 1, m[i].data.data(), 1);
                    } else if constexpr (std::is_same_v<T, double>) {
                        cblas_dscal(n, b1, m[i].data.data(), 1);
                        cblas_daxpy(n, (1 - b1), p->grad.data.data(), 1, m[i].data.data(), 1);
                    } else {
                        m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    }

                    // v = b2*v + (1-b2)*grad^2  — need grad^2 as temp buffer
                    std::vector<T> grad_sq(n);
                    for(size_t k = 0; k < n; k++)
                        grad_sq[k] = p->grad.data[k] * p->grad.data[k];

                    if constexpr (std::is_same_v<T, float>) {
                        cblas_sscal(n, b2, v[i].data.data(), 1);
                        cblas_saxpy(n, (1 - b2), grad_sq.data(), 1, v[i].data.data(), 1);
                    } else if constexpr (std::is_same_v<T, double>) {
                        cblas_dscal(n, b2, v[i].data.data(), 1);
                        cblas_daxpy(n, (1 - b2), grad_sq.data(), 1, v[i].data.data(), 1);
                    } else {
                        Matrix<T> grad_sq_mat(grad_sq, p->grad.shape);  
                        v[i] = b2 * v[i] + (1 - b2) * grad_sq_mat;
                    }

                    // param update: p = p - lr * (m/b1_corr) / (sqrt(v/b2_corr) + eps)
                    // compute step vector = (m_hat) / (sqrt(v_hat) + eps) * lr
  
                    std::vector<T> step(n);
                    for(size_t k = 0; k < n; k++)
                        step[k] = (lr * m[i].data[k] / b1_corr) / 
                                (std::sqrt(v[i].data[k] / b2_corr) + eps);

                    if constexpr (std::is_same_v<T, float>)
                        cblas_saxpy(n, -1.0f, step.data(), 1, p->val.data.data(), 1);
                    else if constexpr (std::is_same_v<T, double>)
                        cblas_daxpy(n, -1.0,  step.data(), 1, p->val.data.data(), 1);
                    else
                        p->val.data = p->val.data - step;
                }
            }

            void AdamW() {
                if(!initialized) {
                    for(auto p : parameters) {
                        if(p->grad.get_size() == 0) continue;
                        m.push_back(Matrix<T>::zeros(p->val.shape));
                        v.push_back(Matrix<T>::zeros(p->val.shape));
                    }
                    initialized = true;
                }

                t++;
                T b1_corr = 1 - std::pow(b1, t);
                T b2_corr = 1 - std::pow(b2, t);

                for(size_t i = 0; i < parameters.size(); i++) {
                    auto p = parameters[i];
                    if(p->grad.get_size() == 0) continue;

                    size_t n = p->val.get_size();

                    // weight decay — scale p->val by (1 - lr*lambda) before Adam step
                    if constexpr (std::is_same_v<T, float>)
                        cblas_sscal(n, (1 - lr * lambda), p->val.data.data(), 1);
                    else if constexpr (std::is_same_v<T, double>)
                        cblas_dscal(n, (1 - lr * lambda), p->val.data.data(), 1);
                    else
                        {
                            p->val = p->val * (1 - lr * lambda);
                        }

                    // m update
                    if constexpr (std::is_same_v<T, float>) {
                        cblas_sscal(n, b1, m[i].data.data(), 1);
                        cblas_saxpy(n, (1 - b1), p->grad.data.data(), 1, m[i].data.data(), 1);
                    } else if constexpr (std::is_same_v<T, double>) {
                        cblas_dscal(n, b1, m[i].data.data(), 1);
                        cblas_daxpy(n, (1 - b1), p->grad.data.data(), 1, m[i].data.data(), 1);
                    } else {
                        m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    }

                    // v update
                    std::vector<T> grad_sq(n);
                    for(size_t k = 0; k < n; k++)
                        grad_sq[k] = p->grad.data[k] * p->grad.data[k];


                    if constexpr (std::is_same_v<T, float>) {
                        cblas_sscal(n, b2, v[i].data.data(), 1);
                        cblas_saxpy(n, (1 - b2), grad_sq.data(), 1, v[i].data.data(), 1);
                    } else if constexpr (std::is_same_v<T, double>) {
                        cblas_dscal(n, b2, v[i].data.data(), 1);
                        cblas_daxpy(n, (1 - b2), grad_sq.data(), 1, v[i].data.data(), 1);
                    } else {
                        Matrix<T> grad_sq_mat(grad_sq, p->grad.shape);  
                        v[i] = b2 * v[i] + (1 - b2) * grad_sq_mat;

                        
                    }

                    // param update — identical to Adam after weight decay
            
                    std::vector<T> step(n);
                    for(size_t k = 0; k < n; k++)
                        step[k] = (lr * m[i].data[k] / b1_corr) / 
                                (std::sqrt(v[i].data[k] / b2_corr) + eps);

                    if constexpr (std::is_same_v<T, float>)
                        cblas_saxpy(n, -1.0f, step.data(), 1, p->val.data.data(), 1);
                    else if constexpr (std::is_same_v<T, double>)
                        cblas_daxpy(n, -1.0,  step.data(), 1, p->val.data.data(), 1);
                    else
                        p->val.data = p->val.data - step;
                }
            }
        public:
            
        
            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim){ 
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps){
                this->b1 = beta1;
                this->b2 = beta2;
                this->eps = eps;
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps, T lambda){
                this->b1 = beta1;
                this->b2 = beta2;
                this->eps = eps;
                this->lambda = lambda;                
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
            }

            
            void zero_grad(){
                for(auto p : this->parameters)
                    p->zero_grad();
            }

            void clip_grad_norm(T max_norm = 1.0) {
                T total_norm = 0;

                for (auto p : parameters) {
                    if (p->grad.get_size() == 0) continue;

                    if constexpr (std::is_same_v<T, float>)
                        total_norm += cblas_sdot(p->grad.get_size(),
                                                p->grad.data.data(), 1,
                                                p->grad.data.data(), 1);
                    else if constexpr (std::is_same_v<T, double>)
                        total_norm += cblas_ddot(p->grad.get_size(),
                                                p->grad.data.data(), 1,
                                                p->grad.data.data(), 1);
                    else
                        for (auto g : p->grad.data)
                            total_norm += g * g;
                }

                total_norm = std::sqrt(total_norm);

                if (total_norm > max_norm) {
                    T scale = max_norm / (total_norm + T(1e-6));
                    for (auto p : parameters) {
                        if (p->grad.get_size() == 0) continue;

                        if constexpr (std::is_same_v<T, float>)
                            cblas_sscal(p->grad.get_size(), scale, p->grad.data.data(), 1);
                        else if constexpr (std::is_same_v<T, double>)
                            cblas_dscal(p->grad.get_size(), scale, p->grad.data.data(), 1);
                        else
                            p->grad = p->grad * scale;
                    }
                }
            }
            void step(){
                
                clip_grad_norm(1.0);
                switch(this->optimizer){
                    case SGD:
                        this->sgd();
                        break;
                    case ADAM:
                        this->Adam();
                        break;
                    case ADAMw:
                        this->AdamW();
                        break;
                }

            }
    };

#endif