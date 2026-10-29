#include "Types/types.hpp"
#include "Operation.hpp"
#include <cmath>

#ifndef __LOG_SOFTMAX_OPP_INCLUDED__
#define __LOG_SOFTMAX_OPP_INCLUDED__

// Stable log-softmax along the last axis.
//   y[i,j] = x[i,j] - m[i] - log(sum_k exp(x[i,k] - m[i])),   m[i] = max_k x[i,k]
// The max shift is mathematically a no-op for the gradient and is treated as a
// constant (detached) here, so dL/dx has the clean closed form:
//   dx[i,j] = G[i,j] - softmax(y)[i,j] * sum_k G[i,k]
template <typename T>
class LogSoftmaxOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
    Tensor_t<T> t1;
    Matrix<T>   output;

    LogSoftmaxOperation(Tensor_t<T> t1) : t1(t1) {}

    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;
    void zero_grad() override { this->t1->zero_grad(); }
    void reset_graph() override { if (this->t1) { this->t1->reset_graph(); this->t1 = nullptr; } }
    void to_string() override { std::cout << "LogSoftmax Operation\n"; }
};

template <typename T>
Tensor_t<T> LogSoftmaxOperation<T>::forward()
{
    const auto& x = this->t1->val;
    const size_t nd = x.shape.size();
    size_t N = 1;
    for (size_t i = 0; i + 1 < nd; ++i) N *= x.shape[i];
    const size_t C = x.shape[nd - 1];

    using A = scalar::acc_t<T>;
    std::vector<T> m(N), out(N * C);
    for (size_t i = 0; i < N; ++i) {
        T mx = x.data[i * C];
        for (size_t j = 1; j < C; ++j) if (x.data[i * C + j] > mx) mx = x.data[i * C + j];
        m[i] = mx;
        A sum = A(0);
        for (size_t j = 0; j < C; ++j) sum += std::exp(scalar::to_acc<T>(x.data[i * C + j]) - scalar::to_acc<T>(mx));
        const A log_sum = std::log(sum);
        for (size_t j = 0; j < C; ++j)
            out[i * C + j] = scalar::from_acc<T>(
                scalar::to_acc<T>(x.data[i * C + j]) - scalar::to_acc<T>(mx) - log_sum);
    }
    this->output = Matrix<T>(std::move(out), x.shape);
    return std::make_shared<Tensor<T>>(this->output, this->shared_from_this());
}

template <typename T>
void LogSoftmaxOperation<T>::backward(Matrix<T> grad)
{
    if (!this->t1->requires_grad) return;
    const Matrix<T>& S = this->output;
    const size_t nd = S.shape.size();
    size_t N = 1;
    for (size_t i = 0; i + 1 < nd; ++i) N *= S.shape[i];
    const size_t C = S.shape[nd - 1];

    using A = scalar::acc_t<T>;
    std::vector<T> dx(N * C);
    for (size_t i = 0; i < N; ++i) {
        A gsum = A(0);
        for (size_t j = 0; j < C; ++j) gsum += scalar::to_acc<T>(grad.data[i * C + j]);
        for (size_t j = 0; j < C; ++j) {
            const A sm = std::exp(scalar::to_acc<T>(S.data[i * C + j]));   // = softmax
            dx[i * C + j] = scalar::from_acc<T>(
                scalar::to_acc<T>(grad.data[i * C + j]) - sm * gsum);
        }
    }
    this->t1->accumulate(Matrix<T>(std::move(dx), S.shape));
}

#endif