#ifndef HEAD__HPP
#define HEAD__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../../Module.hpp"

template <typename T, template<typename> class LinearT>
class Head : public Module<T>{
    private:

        size_t input_dim;
        size_t seq_length;
        size_t head_size;

        LinearT<T> K;
        LinearT<T> Q;
        LinearT<T> V;

        Matrix<T> mask;

    public:

    template <typename... Args>
    Head(size_t head_size, size_t input_dim, size_t sequence_length, Args&&... args):
      K(head_size, input_dim, args...),
      Q(head_size, input_dim, args...),
      V(head_size, input_dim, args...)
    {
        this->input_dim = input_dim;
        this->seq_length = sequence_length;
        this->head_size = head_size;

        this->register_module(&Q);
        this->register_module(&K);
        this->register_module(&V);

    }

    Head(Head&& other)
        : Module<T>(), // fresh, empty submodules list
        input_dim(other.input_dim),
        seq_length(other.seq_length),
        head_size(other.head_size),
        K(std::move(other.K)),
        Q(std::move(other.Q)),
        V(std::move(other.V)),
        mask(other.mask)
    {
        this->register_module(&Q);
        this->register_module(&K);
        this->register_module(&V);
    }

    Head(const Head&) = delete;

    void set_mask(size_t seq) {
        if (this->mask.get_size() > 0 && this->mask.shape[1] == seq)
            return;
        Matrix<T> tmp = Matrix<T>::tril(Matrix<T>::ones({seq, seq}));
        Matrix<bool> tmp2(tmp == 0);
        this->mask = Matrix<T>::where(tmp2, -Matrix<T>::inf(), (T)0.0);
    }

    std::pair<Tensor_t<T>, Tensor_t<T>> scaled_dot_product_attention(Tensor_t<T> q, Tensor_t<T> k, Tensor_t<T> v, bool apply_mask)
    {
        size_t d_k = this->head_size;

        auto scaled_scores = q->matmul(k->transpose({0, 2, 1})) / (T)std::sqrt((double)d_k);

        if (apply_mask) {
            size_t actual_seq = q->shape[1];   // ← runtime seq length, e.g. 22
            this->set_mask(actual_seq);        // ← rebuild mask only when seq changes
            scaled_scores = scaled_scores + make_tensor<T>(this->mask);
        }

        auto attention = scaled_scores->softmax();
        return {attention->matmul(v), attention};
    }

    LinearT<T>& get_Q() { return Q; }
    LinearT<T>& get_K() { return K; }
    LinearT<T>& get_V() { return V; }

    void load_pretrained(Head<T, Linear>& src) {
        Q.load_pretrained(src.get_Q());
        K.load_pretrained(src.get_K());
        V.load_pretrained(src.get_V());
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask=true){
        // """Forward pass through attention head"""

        // std::cerr << " head input x val : "<< x->val<< "\n";

        // Compute Q, K, V projections
        Tensor_t<T> q = this->Q.forward(x);  // (B, T, head_size)
        Tensor_t<T> k = this->K.forward(x);  // (B, T, head_size)
        Tensor_t<T> v = this->V.forward(x);  // (B, T, head_size)

        // std::cerr << " q val : "<< q->val<< "\n";
        // std::cerr << " k val : "<< k->val<< "\n";
        // std::cerr << " v val : "<< v->val<< "\n";

        // Apply scaled dot-product attention
        auto res = this->scaled_dot_product_attention(q, k, v, apply_mask);

        Tensor_t<T> values = res.first;
        Tensor_t<T> attention = res.second;
        // std::cerr << " scaled dot product values and attention val : "<< values->val<< attention->val << "\n";

        return values;        
    }
    
    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
};

#endif // !HEAD__HPP