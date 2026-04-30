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

            void sgd(){
                for(auto p : this->parameters)
                {
                    if(p->grad.get_size() == 0) continue; // skip if no gradient

                    if constexpr (std::is_same_v<T, float>)
                        cblas_saxpy(
                            p->val.get_size(),    // number of elements
                            -this->lr,              // alpha = -lr
                            p->grad.data.data(), 1, // x = gradient
                            p->val.data.data(), 1  // y = parameters (modified in place)
                        );
                    else if constexpr (std::is_same_v<T, double>)
                        cblas_daxpy(
                            p->val.get_size(),
                            -this->lr,
                            p->grad.data.data(), 1,
                            p->val.data.data(), 1
                        );
                    else
                        p->val = p->val - this->lr * p->grad; 

                }
            }

            void Adam()
            {
                
                if(!initialized){
                    for(auto p : parameters){
                        if(p->grad.get_size() == 0) continue;
                        m.push_back(Matrix<T>::zeros(p->val.shape));
                        v.push_back(Matrix<T>::zeros(p->val.shape));
                    }

                    initialized = true;
                }

                t++;
                for(size_t i = 0; i < parameters.size(); i++){
    
                    auto p = parameters[i];
                    if(p->grad.get_size() == 0) continue;

                    m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    v[i] = b2 * v[i] + (1 - b2) * (p->grad*p->grad);

                    auto m_hat = m[i] / (1 - std::pow(b1, t));
                    auto v_hat = v[i] / (1 - std::pow(b2, t));

                    p->val = p->val - ((lr * m_hat) / (v_hat.sqrt() + Matrix<T>(eps)));

                    // if constexpr (std::is_same_v<T, float>)
                    //     cblas_saxpy(
                    //         p->val.data.size(),    // number of elements
                    //         -al.data[0],              // alpha = -lr
                    //         p->grad.data.data(), 1, // x = gradient
                    //         p->val.data.data(), 1  // y = parameters (modified in place)
                    //     );
                    // else if constexpr (std::is_same_v<T, double>)
                    //     cblas_daxpy(
                    //         p->val.data.size(),
                    //         -al.data[0],
                    //         p->grad.data.data(), 1,
                    //         p->val.data.data(), 1
                    //     );
                }
            }

            void AdamW()
            {
                if(!initialized){
                    for(auto p : parameters){
                        if(p->grad.get_size() == 0) continue;
                        m.push_back(Matrix<T>::zeros(p->val.shape));
                        v.push_back(Matrix<T>::zeros(p->val.shape));
                    }

                    initialized = true;
                }

                t++;

                for(size_t i = 0; i < parameters.size(); i++){
                    auto p = parameters[i];
                    if(p->grad.get_size() == 0) continue;

                    m[i] = b1 * m[i] + (1 - b1) * p->grad;
                    v[i] = b2 * v[i] + (1 - b2) * (p->grad*p->grad);

                    auto m_hat = m[i] / (1 - std::pow(b1, t));
                    auto v_hat = v[i] / (1 - std::pow(b2, t));

                    p->val = p->val * (1 - lr * lambda);  // weight decay first
                    p->val = p->val - ((lr * m_hat) / (v_hat.sqrt() + Matrix<T>(eps)));

                    // if constexpr (std::is_same_v<T, float>)
                    //     cblas_saxpy(
                    //         p->val.data.size(),    // number of elements
                    //         -al.data[0],              // alpha = -lr
                    //         p->grad.data.data(), 1, // x = gradient
                    //         p->val.data.data(), 1  // y = parameters (modified in place)
                    //     );
                    // else if constexpr (std::is_same_v<T, double>)
                    //     cblas_daxpy(
                    //         p->val.data.size(),
                    //         -al.data[0],
                    //         p->grad.data.data(), 1,
                    //         p->val.data.data(), 1
                    //     );
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
                    for (auto g : p->grad.data)
                        total_norm += g * g;
                }
                total_norm = std::sqrt(total_norm);
                if (total_norm > max_norm) {
                    T scale = max_norm / (total_norm + T(1e-6));
                    for (auto p : parameters) {
                        if (p->grad.get_size() == 0) continue;
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

// #ifndef __OPTIMIZER__HPP_
// #define __OPTIMIZER__HPP_

// #include "../Types/types.hpp"
// #include "../DataStructures/Tensor.hpp"

// #include "Module.hpp"

//     #include <iostream>
//     #include <vector>

//     enum Optimizer_t{
//         SGD,
//         // ADAM,
//         // ADAMw
//     };

//     template <typename T>
//     class Optimizer{
//         protected:
//             Optimizer_t optimizer = SGD;
//             T lr;
//             std::vector<Tensor_t<T>> parameters;

//             void sgd(){
//                 for(auto p : parameters)
//                 {
//                     p->val = p->val - this->lr * p->grad;
//                 }
//             }

//         public:

//             Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim){
//                 this->parameters = params;
//                 this->lr = lr;
//                 this->optimizer = optim;
//             }

//             void zero_grad(){
//                 for(auto p : parameters)
//                     p->zero_grad();
//             }

//             void step(){
//                 switch(this->optimizer){
//                     case SGD:
//                         this->sgd();
//                         break;
//                 }

//             }
//     };

// #endif