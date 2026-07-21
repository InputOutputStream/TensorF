#ifndef LLAMA_HEAD__HPP
#define LLAMA_HEAD__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../../Module.hpp"
#include "../../RotationalPositionalEncoding.hpp"

template <typename T, template<typename> class LinearT>
class Head : public Module<T>{
    private:

        size_t input_dim;
        size_t seq_length;
        size_t head_size;
        RotationalPositionalEncoding<T> rope;

        LinearT<T> K;
        LinearT<T> Q;
        LinearT<T> V;

        Matrix<T> mask;

    public:

    template <typename... Args>
    Head(size_t head_size, size_t input_dim, size_t sequence_length, Args&&... args):
        input_dim(input_dim),
        seq_length(sequence_length),
        head_size(head_size),
        rope(head_size, sequence_length),
        K(head_size, input_dim, args...),
        Q(head_size, input_dim, args...),
        V(head_size, input_dim, args...)
    {
        this->register_module(&Q);
        this->register_module(&K);
        this->register_module(&V);
    }

    Head(Head&& other)
        : Module<T>(),
        input_dim(other.input_dim),
        seq_length(other.seq_length),
        head_size(other.head_size),
        rope(std::move(other.rope)),
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

    LinearT<T>& get_Q() { return Q; }
    LinearT<T>& get_K() { return K; }
    LinearT<T>& get_V() { return V; }

    // Pretrained source is always dense (Linear<T>) — see LlamaGGUFLoader
    void load_pretrained(Head<T, Linear>& src) {
        Q.load_pretrained(src.get_Q());
        K.load_pretrained(src.get_K());
        V.load_pretrained(src.get_V());
    }

    std::pair<Tensor_t<T>, Tensor_t<T>> scaled_dot_product_attention(
        Tensor_t<T> q, Tensor_t<T> k, Tensor_t<T> v, bool apply_mask)
    {
        size_t d_k = this->head_size;

        auto scaled_scores = q->matmul(k->transpose({0, 2, 1})) / (T)std::sqrt((double)d_k);

        if (apply_mask) {
            size_t actual_seq = q->shape[1];
            this->set_mask(actual_seq);
            scaled_scores = scaled_scores + make_tensor<T>(this->mask);
        }

        auto attention = scaled_scores->softmax();
        return {attention->matmul(v), attention};
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask=true){

        // Q, K, V projections
        Tensor_t<T> q = this->Q.forward(x);  // (B, T, head_size)
        Tensor_t<T> k = this->K.forward(x);  // (B, T, head_size)
        Tensor_t<T> v = this->V.forward(x);  // (B, T, head_size)

        // Apply positional encoding to Q and K 

        size_t T_len = x->shape[1];
        Tensor_t<T> pos_indices = make_tensor<T>(Matrix<T>::arrange(T_len).reshape({1, T_len}));  // {1, T}

        Tensor_t<T> pe = this->rope.forward(pos_indices);  // {T, head_size}
        // pe broadcasts over batch dimension when added to {B, T, head_size}
        q = q + pe;
        k = k + pe;

        auto res = this->scaled_dot_product_attention(q, k, v, apply_mask);

        return res.first;
    }
    
    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;

};

#endif // !LLAMA_HEAD__HPP