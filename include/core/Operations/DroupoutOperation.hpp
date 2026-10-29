#include "Types/types.hpp"
#include "Operation.hpp"
#include <cstdint>
#include <optional>
#include <random>
#include <vector>

#ifndef __DROPOUT_OPP_INCLUDED__
#define __DROPOUT_OPP_INCLUDED__

// Inverted dropout as an autograd operation.
//   forward:  y[i] = x[i] * (keep ? 1/(1-p) : 0)
//   backward: dx[i] = grad[i] * (keep ? 1/(1-p) : 0)     [only if x requires grad]
//
// The mask is stored bit-packed (n/8 bytes vs 4n bytes for a float mask),
// forward fuses mask generation + scaling into one pass, and backward is
// skipped entirely when the input doesn't require a gradient.
template <typename T>
class DropoutOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
    Tensor_t<T> t1;
    float p_keep_;
    std::vector<uint32_t> mask_bits_;

    // The caller passes the RNG (same convention as coratrain's dropout()).
    // Reseeding the global RNG from inside an op would surprise other users
    // of mxd::rng(), so the reference is taken instead.
    DropoutOperation(Tensor_t<T> t1, float p, std::mt19937& rng)
        : t1(t1), p_keep_(1.0f - p), rng_(rng)
    {
        if (p < 0.0f || p >= 1.0f)
            throw std::invalid_argument("Dropout: p must be in [0, 1)");
    }

    Tensor_t<T> forward() override
    {
        const size_t n = this->t1->val.data.size();
        this->mask_bits_.assign((n + 31) / 32, 0u);
        std::vector<T> out(n);

        const T scale = T(1.0 / static_cast<double>(this->p_keep_));
        constexpr double inv = 1.0 / 4294967296.0;
        const double p_keep = static_cast<double>(this->p_keep_);

        for (size_t i = 0; i < n; ++i) {
            const bool keep = (static_cast<double>(this->rng_()) * inv < p_keep);
            if (keep) {
                this->mask_bits_[i >> 5] |= (1u << (i & 31));
                out[i] = this->t1->val.data[i] * scale;
            } else {
                out[i] = T(0);
            }
        }
        return std::make_shared<Tensor<T>>(
            Matrix<T>(std::move(out), this->t1->val.shape),
            this->shared_from_this());
    }

    void backward(Matrix<T> grad) override
    {
        // Constant input (X->requires_grad = false): skip the [N] scatter entirely.
        if (!this->t1 || !this->t1->requires_grad) return;

        const size_t n = grad.data.size();
        const T scale = T(1.0 / static_cast<double>(this->p_keep_));
        std::vector<T> dx(n);
        for (size_t i = 0; i < n; ++i)
            dx[i] = (this->mask_bits_[i >> 5] & (1u << (i & 31)))
                        ? (grad.data[i] * scale)
                        : T(0);
        this->t1->accumulate(Matrix<T>(std::move(dx), this->t1->val.shape));
    }

    void zero_grad() override { if (this->t1) this->t1->zero_grad(); }

    void reset_graph() override {
        if (this->t1) { this->t1->reset_graph(); this->t1 = nullptr; }
        this->mask_bits_.clear();
    }

    void to_string() override {
        std::cout << "Dropout Operation (p_keep=" << this->p_keep_ << ")\n";
    }

private:
    std::mt19937& rng_;
};

#endif