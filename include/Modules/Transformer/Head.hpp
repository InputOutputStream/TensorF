#ifndef HEAD__HPP
#define HEAD__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../Module.hpp"
#include "../Linear.hpp"

template <typename T>
class Head : public Module<T>{
    private:
        bool bias;

        size_t input_dim;
        size_t seq_length;
        size_t head_size;
    
        Linear<T> K;
        Linear<T> Q;
        Linear<T> V;

        Tensor_t<T> scaled_scores;
        Tensor_t<T> attention;
        Matrix<T> mask;

    public:

    Head(size_t head_size, size_t input_dim, size_t sequence_length): 
      K(input_dim, head_size, true),
      Q(input_dim, head_size, true),
      V(input_dim, head_size, true)
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
        bias(other.bias),
        input_dim(other.input_dim),
        seq_length(other.seq_length),
        head_size(other.head_size),
        K(std::move(other.K)),
        Q(std::move(other.Q)),
        V(std::move(other.V)),
        scaled_scores(other.scaled_scores),
        attention(other.attention),
        mask(other.mask)
    {
        this->register_module(&Q);
        this->register_module(&K);
        this->register_module(&V);
    }

Head(const Head&) = delete;

    void set_mask(shape_t scores_shape) {
        if (this->mask != nullptr && this->mask.shape == scores_shape)
            return;

        //size_t seq = this->seq_length;
        Matrix<T> tmp = Matrix<T>::tril(Matrix<T>::ones(scores_shape));
        Matrix<bool> tmp2(tmp == 0);
        this->mask = Matrix<T>::where(tmp2, -Matrix<T>::inf(), (T)0.0);
    }

    std::pair<Tensor_t<T>, Tensor_t<T>> scaled_dot_product_attention(
        Tensor_t<T> q, Tensor_t<T> k, Tensor_t<T> v, bool apply_mask)
    {
        size_t d_k = this->head_size;

        this->scaled_scores = q->matmul(k->transpose()) / (T)std::sqrt((double)d_k);

        if (apply_mask) {
            if (this->mask.get_size() == 0)
                this->set_mask(this->scaled_scores->shape);  
            this->scaled_scores = this->scaled_scores + make_tensor<T>(this->mask);
        }

        this->attention = this->scaled_scores->softmax();
        Tensor_t<T> output = this->attention->matmul(v);

        return {output, this->attention};
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask=true){
        // """Forward pass through attention head"""
    
        std::cerr << " head input x val : "<< x->val<< "\n";

        // Compute Q, K, V projections
        Tensor_t<T> q = this->Q.forward(x);  // (B, T, head_size)
        Tensor_t<T> k = this->K.forward(x);  // (B, T, head_size) 
        Tensor_t<T> v = this->V.forward(x);  // (B, T, head_size)

        std::cerr << " q val : "<< q->val<< "\n";
        std::cerr << " k val : "<< k->val<< "\n";
        std::cerr << " v val : "<< v->val<< "\n";

        // Apply scaled dot-product attention
        auto res = this->scaled_dot_product_attention(q, k, v, apply_mask);

        Tensor_t<T> values = res.first;
        this->attention = res.second;
        std::cerr << " scaled dot product values and attention val : "<< values->val<< attention->val << "\n";

        return values;        
    }

};

#endif // !HEAD__HPP