#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __GATHER_OPP_INCLUDED__
#define __GATHER_OPP_INCLUDED__

// Gather: out[i] = logits[i, labels[i]], per position along the last axis.
// logits: [..., C] (rank >= 2), labels: logits.shape[:-1] (same T, integer valued).
// result: labels.shape.  Backward scatters grad back into logits at the label positions.
template <typename T>
class GatherOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
    Tensor_t<T> t1;         // logits
    Tensor_t<T> labels;     // indices, integer-valued T
    shape_t  orig_shape;
    size_t   num_elems   = 0;
    size_t   num_classes = 0;

    GatherOperation(Tensor_t<T> logits, Tensor_t<T> labels)
        : t1(logits), labels(labels) {}

    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;
    void zero_grad() override { this->t1->zero_grad(); if (this->labels) this->labels->zero_grad(); }
    void reset_graph() override {
        if (this->t1)     { this->t1->reset_graph();     this->t1 = nullptr; }
        if (this->labels) { this->labels->reset_graph(); this->labels = nullptr; }
    }
    void to_string() override { std::cout << "Gather Operation\n"; }
};

template <typename T>
Tensor_t<T> GatherOperation<T>::forward()
{
    const auto& lg = this->t1->val;
    if (lg.shape.size() < 2) throw std::invalid_argument("Gather: logits must have rank >= 2");
    this->num_classes = lg.shape.back();
    this->num_elems   = lg.data.size() / this->num_classes;
    if (this->labels->val.data.size() != this->num_elems)
        throw std::invalid_argument("Gather: labels size must equal logits.numel / num_classes");

    std::vector<T> out(this->num_elems);
    for (size_t i = 0; i < this->num_elems; ++i) {
        double v = std::round(static_cast<double>(scalar::to_acc<T>(this->labels->val.data[i])));
        if (!(v >= 0) || v >= static_cast<double>(this->num_classes))
            throw std::out_of_range("Gather: label " + std::to_string(v) + " outside [0, " +
                                    std::to_string(this->num_classes) + ")");
        out[i] = lg.data[i * this->num_classes + static_cast<size_t>(v)];
    }
    this->orig_shape = lg.shape;
    shape_t out_shape = this->labels->val.shape;
    return std::make_shared<Tensor<T>>(Matrix<T>(std::move(out), std::move(out_shape)), this->shared_from_this());
}

template <typename T>
void GatherOperation<T>::backward(Matrix<T> grad)
{
    if (!this->t1->requires_grad) return;
    
    if (grad.get_size() != this->num_elems)
        throw std::invalid_argument("Gather::backward: grad size mismatch");
    std::vector<T> dx(this->num_elems * this->num_classes, T(0));
    for (size_t i = 0; i < this->num_elems; ++i) {
        double v = std::round(static_cast<double>(scalar::to_acc<T>(this->labels->val.data[i])));
        size_t j = static_cast<size_t>(v);
        dx[i * this->num_classes + j] = grad.data[i];
    }
    this->t1->accumulate(Matrix<T>(std::move(dx), this->orig_shape));
}

#endif