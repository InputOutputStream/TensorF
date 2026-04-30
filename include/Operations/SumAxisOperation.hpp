#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __SUM_AXIS_OPP_INCLUDED__
#define __SUM_AXIS_OPP_INCLUDED__

/**
 * SumAxisOperation
 *
 * forward:  reduces input along `axis`, squeezing that dimension.
 *           e.g. {N, C} --axis=1--> {N}
 *
 * backward: the upstream grad has the reduced shape {N}.
 *           We broadcast it back to the original shape {N, C}
 *           by repeating it along the summed axis — identical to
 *           what SumOperation does for the scalar case but per-row/col.
 */
template <typename T>
class SumAxisOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        size_t axis;
        shape_t input_shape;   // saved for backward broadcast

    SumAxisOperation(Tensor_t<T> t1, size_t axis)
        : t1(t1), axis(axis), input_shape(t1->val.shape)
    {}

    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;
    void zero_grad() override;
    void reset_graph() override;

    void to_string() override {
        std::cout << "SumAxis Operation (axis=" << axis << ")\n";
    }
};

// ── forward ──────────────────────────────────────────────────────────────────

template <typename T>
Tensor_t<T> SumAxisOperation<T>::forward()
{
    Matrix<T> result = this->t1->val.sum(this->axis);
    return std::make_shared<Tensor<T>>(result, this->shared_from_this());
}

// ── backward ─────────────────────────────────────────────────────────────────
//
// grad has the reduced shape (axis dimension squeezed out).
// We need to broadcast it back to input_shape by expanding along `axis`
// and repeating the values — equivalent to tiling the grad slice `axis_size` times.
//
// Strategy: rebuild a full-size gradient buffer using the same index arithmetic
// as broadcastTo, treating the axis dimension as having size 1 in grad.

template <typename T>
void SumAxisOperation<T>::backward(Matrix<T> grad)
{
    // Build the shape grad *would* have if we kept the axis dim as 1
    shape_t kept_shape = this->input_shape;
    kept_shape[this->axis] = 1;          // e.g. {N,1} when axis=1

    // Strides for the kept shape and the full input shape
    auto computeStrides = [](const shape_t& s) {
        shape_t st(s.size());
        size_t p = 1;
        for (int i = (int)s.size()-1; i >= 0; --i) {
            st[i] = p;
            p *= s[i];
        }
        return st;
    };

    shape_t kept_strides  = computeStrides(kept_shape);
    shape_t input_strides = computeStrides(this->input_shape);

    size_t total = 1;
    for (auto d : this->input_shape) total *= d;

    std::vector<T> out(total);

    for (size_t i = 0; i < total; ++i) {
        // decompose flat index i into multi-index in input_shape
        shape_t idx(this->input_shape.size());
        size_t rem = i;
        for (size_t d = 0; d < this->input_shape.size(); ++d) {
            idx[d] = rem / input_strides[d];
            rem    = rem % input_strides[d];
        }

        // clamp the axis dimension to 0 to index into grad
        idx[this->axis] = 0;

        // compute flat index into grad
        size_t grad_idx = 0;
        for (size_t d = 0; d < kept_shape.size(); ++d)
            grad_idx += idx[d] * kept_strides[d];

        out[i] = grad.data[grad_idx];
    }

    this->t1->backward(Matrix<T>(out, this->input_shape));
}

template <typename T>
void SumAxisOperation<T>::zero_grad() {
    this->t1->zero_grad();
}

template <typename T>
void SumAxisOperation<T>::reset_graph() {
    this->t1->reset_graph();
}

#endif // __SUM_AXIS_OPP_INCLUDED__