#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __MAX_AXIS_OPP_INCLUDED__
#define __MAX_AXIS_OPP_INCLUDED__

// Per-axis max with autograd. Keeps the reduced axis as size 1 (same convention
// as SumAxisOperation), so {N, C}.max(1) -> {N, 1}. Gradient flows only to the
// argmax position (first occurrence on ties).
template <typename T>
class MaxAxisOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
    Tensor_t<T> t1;
    size_t axis;
    shape_t input_shape;
    shape_t out_shape;
    std::vector<size_t> argmax_idx;

    MaxAxisOperation(Tensor_t<T> t1, size_t axis)
        : t1(t1), axis(axis), input_shape(t1->val.shape) {}

    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;
    void zero_grad() override { this->t1->zero_grad(); }
    void reset_graph() override { if (this->t1) { this->t1->reset_graph(); this->t1 = nullptr; } }
    void to_string() override { std::cout << "MaxAxis Operation (axis=" << axis << ")\n"; }
};

template <typename T>
Tensor_t<T> MaxAxisOperation<T>::forward()
{
    const auto& x = this->t1->val;
    const size_t nd = input_shape.size();
    if (axis >= nd) throw std::out_of_range("MaxAxis: axis out of range");
    size_t outer = 1, n = input_shape[axis], inner = 1;
    for (size_t d = 0; d < axis; d++) outer *= input_shape[d];
    for (size_t d = axis + 1; d < nd; d++) inner *= input_shape[d];

    out_shape = input_shape;
    out_shape[axis] = 1;
    std::vector<T> out(outer * inner);
    argmax_idx.resize(outer * inner);

    for (size_t o = 0; o < outer; o++)
        for (size_t i = 0; i < inner; i++) {
            size_t best_k = 0;
            T best_v = x.data[(o * n) * inner + i];
            for (size_t k = 1; k < n; k++) {
                const T v = x.data[(o * n + k) * inner + i];
                if (v > best_v) { best_v = v; best_k = k; }
            }
            out[o * inner + i] = best_v;
            argmax_idx[o * inner + i] = best_k;
        }
    return std::make_shared<Tensor<T>>(Matrix<T>(std::move(out), out_shape), this->shared_from_this());
}

template <typename T>
void MaxAxisOperation<T>::backward(Matrix<T> grad)
{
    if (!this->t1->requires_grad) return;
    const size_t nd = input_shape.size();
    size_t outer = 1, n = input_shape[axis], inner = 1;
    for (size_t d = 0; d < axis; d++) outer *= input_shape[d];
    for (size_t d = axis + 1; d < nd; d++) inner *= input_shape[d];

    size_t total = 1;
    for (auto d : input_shape) total *= d;
    std::vector<T> dx(total, T(0));
    for (size_t o = 0; o < outer; o++)
        for (size_t i = 0; i < inner; i++) {
            const size_t k = argmax_idx[o * inner + i];
            dx[(o * n + k) * inner + i] = grad.data[o * inner + i];
        }
    this->t1->accumulate(Matrix<T>(std::move(dx), input_shape));
}

#endif