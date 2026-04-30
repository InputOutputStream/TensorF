#include "../Types/types.hpp"
#include "Operation.hpp"
#include <cmath>

#ifndef __SOFTMAX_OPP_INCLUDED__
#define __SOFTMAX_OPP_INCLUDED__

/**
 * SoftmaxOperation  (row-wise, axis = last dim)
 *
 * forward:
 *   For each row i:
 *     s[i,j] = exp(x[i,j] - max(x[i,:])) / sum_k(exp(x[i,k] - max(x[i,:])))
 *
 * backward (analytic Jacobian-vector product):
 *   Given upstream grad G (same shape as output S):
 *   dL/dx[i,j] = s[i,j] * (G[i,j] - sum_k(G[i,k] * s[i,k]))
 *              = s[i,j] * (G[i,j] - dot(G[i,:], s[i,:]))
 *
 *   This is the standard softmax backward that avoids building the full
 *   N×C×C Jacobian.
 */
template <typename T>
class SoftmaxOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        Matrix<T>   output;   // saved forward output S, needed for backward

    SoftmaxOperation(Tensor_t<T> t1)
        : t1(t1)
    {}

    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;
    void zero_grad() override;
    void reset_graph() override;

    void to_string() override {
        std::cout << "Softmax Operation\n";
    }
};

// ── forward ──────────────────────────────────────────────────────────────────

template <typename T>
Tensor_t<T> SoftmaxOperation<T>::forward()
{
    const auto& x     = this->t1->val;
    size_t      ndims = x.shape.size();

    // Works for any ND input — treats last dim as the class axis
    size_t N = 1;
    for (size_t i = 0; i < ndims - 1; ++i) N *= x.shape[i];
    size_t C = x.shape[ndims - 1];   // number of classes

    std::vector<T> out(N * C);

    for (size_t i = 0; i < N; ++i) {
        // 1. row max for numerical stability
        T max_val = x.data[i * C];
        for (size_t j = 1; j < C; ++j)
            if (x.data[i * C + j] > max_val)
                max_val = x.data[i * C + j];

        // 2. exp(x - max)
        T sum = T(0);
        for (size_t j = 0; j < C; ++j) {
            out[i * C + j] = std::exp(x.data[i * C + j] - max_val);
            sum += out[i * C + j];
        }

        // 3. normalise
        for (size_t j = 0; j < C; ++j)
            out[i * C + j] /= sum;
    }

    this->output = Matrix<T>(out, x.shape);
    return std::make_shared<Tensor<T>>(this->output, this->shared_from_this());
}

// ── backward ─────────────────────────────────────────────────────────────────

template <typename T>
void SoftmaxOperation<T>::backward(Matrix<T> grad)
{
    const Matrix<T>& S = this->output;
    size_t ndims = S.shape.size();

    size_t N = 1;
    for (size_t i = 0; i < ndims - 1; ++i) N *= S.shape[i];
    size_t C = S.shape[ndims - 1];

    std::vector<T> dx(N * C);

    for (size_t i = 0; i < N; ++i) {
        // dot(G[i,:], S[i,:])
        T dot = T(0);
        for (size_t j = 0; j < C; ++j)
            dot += grad.data[i * C + j] * S.data[i * C + j];

        // dL/dx[i,j] = S[i,j] * (G[i,j] - dot)
        for (size_t j = 0; j < C; ++j)
            dx[i * C + j] = S.data[i * C + j] * (grad.data[i * C + j] - dot);
    }

    this->t1->backward(Matrix<T>(dx, S.shape));
}

template <typename T>
void SoftmaxOperation<T>::zero_grad() {
    this->t1->zero_grad();
}

template <typename T>
void SoftmaxOperation<T>::reset_graph() {
    this->t1->reset_graph();
}

#endif // __SOFTMAX_OPP_INCLUDED__