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
        ADAMw,
        ADAFACTOR
    };

    template <typename T>
    class Optimizer{
        protected:
            Optimizer_t optimizer;
            T lr = 0.1;
            std::vector<Tensor_t<T>> parameters;
            bool requires_grad;

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

            // Adafactor state
            std::vector<std::vector<T>> af_row;  // per-param row accumulator R_t, only for 2D params
            std::vector<std::vector<T>> af_col;  // per-param col accumulator C_t, only for 2D params
            std::vector<Matrix<T>> af_v;          // full second moment, only for non-2D params (bias etc.)
            T af_decay          = (T)0.8;   // decay exponent: rho_t = 1 - t^-af_decay
            T af_eps1           = (T)1e-30; // floor added to grad^2 before accumulating
            T af_eps2           = (T)1e-3;  // floor on parameter RMS for the relative step size
            T af_clip_threshold = (T)1.0;   // RMS clipping threshold on the raw update


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
                    t = 0;
                    for(auto p : parameters) {
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

                    // v = b2*v + (1-b2)*pow(grad,2)  — need pow(grad,2) as temp buffer
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
                    t = 0;
                    for(auto p : parameters) {
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

            void adafactor() {
                if (!initialized) {
                    t = 0;
                    af_row.resize(parameters.size());
                    af_col.resize(parameters.size());
                    af_v.resize(parameters.size());
                    for (size_t i = 0; i < parameters.size(); i++) {
                        auto& shape = parameters[i]->val.shape;
                        if (shape.size() == 2) {
                            af_row[i].assign(shape[0], (T)0);
                            af_col[i].assign(shape[1], (T)0);
                        } else {
                            af_v[i] = Matrix<T>::zeros(shape);
                        }
                    }
                    initialized = true;
                }

                t++;
                // Adafactor's default schedule (no user-tunable beta2 — decays toward 1)
                T rho_t = (T)1.0 - std::pow((T)t, -af_decay);

                for (size_t i = 0; i < parameters.size(); i++) {
                    auto p = parameters[i];
                    if (p->grad.get_size() == 0) continue;

                    auto& shape = p->val.shape;
                    size_t n = p->val.get_size();

                    // relative step size: scale lr by the parameter's own RMS
                    // (this is Adafactor's default "relative step" behaviour —
                    // if you want a plain fixed step instead, just set af_eps2
                    // very large so max(af_eps2, param_rms) == af_eps2 == your lr scale)
                    T param_rms = (T)0;
                    for (auto x : p->val.data) param_rms += x * x;
                    param_rms = std::sqrt(param_rms / (T)n);
                    T step_size = this->lr * std::max(af_eps2, param_rms);

                    std::vector<T> update(n);
                    T update_rms_sq = 0;

                    if (shape.size() == 2) {
                        size_t rows = shape[0], cols = shape[1];

                        std::vector<T> row_mean(rows, (T)0), col_mean(cols, (T)0);
                        for (size_t r = 0; r < rows; r++) {
                            for (size_t c = 0; c < cols; c++) {
                                T g2 = p->grad.data[r * cols + c] * p->grad.data[r * cols + c] + af_eps1;
                                row_mean[r] += g2;
                                col_mean[c] += g2;
                            }
                        }
                        for (size_t r = 0; r < rows; r++) row_mean[r] /= (T)cols;
                        for (size_t c = 0; c < cols; c++) col_mean[c] /= (T)rows;

                        T row_total = 0;
                        for (size_t r = 0; r < rows; r++) {
                            af_row[i][r] = rho_t * af_row[i][r] + ((T)1 - rho_t) * row_mean[r];
                            row_total += af_row[i][r];
                        }
                        for (size_t c = 0; c < cols; c++)
                            af_col[i][c] = rho_t * af_col[i][c] + ((T)1 - rho_t) * col_mean[c];

                        T row_avg = row_total / (T)rows;

                        // rank-1 reconstruction: V_hat[r][c] = R[r]*C[c] / mean(R)
                        for (size_t r = 0; r < rows; r++) {
                            for (size_t c = 0; c < cols; c++) {
                                T v_hat = (af_row[i][r] * af_col[i][c]) / std::max(row_avg, af_eps1);
                                T u = p->grad.data[r * cols + c] / std::sqrt(v_hat);
                                update[r * cols + c] = u;
                                update_rms_sq += u * u;
                            }
                        }
                    } else {
                        // 1D (bias/gamma/beta) or other-rank tensors: no factoring possible,
                        // fall back to a plain per-element second moment.
                        for (size_t k = 0; k < n; k++) {
                            T g2 = p->grad.data[k] * p->grad.data[k] + af_eps1;
                            af_v[i].data[k] = rho_t * af_v[i].data[k] + ((T)1 - rho_t) * g2;
                        }
                        for (size_t k = 0; k < n; k++) {
                            T u = p->grad.data[k] / std::sqrt(af_v[i].data[k]);
                            update[k] = u;
                            update_rms_sq += u * u;
                        }
                    }

                    // update clipping (paper's RMS clip, distinct from the optimizer's own
                    // global clip_grad_norm )
                    T update_rms = std::sqrt(update_rms_sq / (T)n);
                    T clip = std::max((T)1.0, update_rms / af_clip_threshold);

                    for (size_t k = 0; k < n; k++)
                        p->val.data[k] -= step_size * (update[k] / clip);
                }
            }

            void set_adafactor_params(T decay, T eps1, T eps2, T clip_threshold) {
                af_decay = decay; af_eps1 = eps1; af_eps2 = eps2; af_clip_threshold = clip_threshold;
            }

        public:
            
        
            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, bool requires_grad){ 
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
                this->requires_grad = requires_grad;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps, bool requires_grad){
                this->b1 = beta1;
                this->b2 = beta2;
                this->eps = eps;
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
                this->requires_grad = requires_grad;
            }

            Optimizer(std::vector<Tensor_t<T>> params, T lr, Optimizer_t optim, T beta1, T beta2, T eps, T lambda, bool requires_grad){
                this->b1 = beta1;
                this->b2 = beta2;
                this->eps = eps;
                this->lambda = lambda;                
                this->parameters = params;
                this->lr = lr;
                this->optimizer = optim;
                this->requires_grad = requires_grad;
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

                if (this->optimizer != ADAFACTOR)
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
                    case ADAFACTOR:
                        this->adafactor();
                        break;
                }

            }
    };

#endif