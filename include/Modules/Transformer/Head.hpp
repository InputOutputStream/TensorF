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
        Tensor_t<T> mask;

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

        this->set_mask();
    }

    void set_mask(){
        if (this->mask != nullptr)
            return;
            
        // Create lower triangular mask for causal attention
        Matrix<T> tmp = Matrix<T>::tril(Matrix<T>::ones({this->seq_length, this->seq_length}));
        
        // Convert to attention mask: 0 -> -inf, 1 -> 0
        this->mask = make_tensor<T>(Matrix<T>::where(tmp == 0, -Matrix<T>::inf(), 0.0f));
    }

    std::pair<Tensor_t<T>, Tensor_t<T>> scaled_dot_product_attention(Tensor_t<T> q, Tensor_t<T> k, Tensor_t<T> v, bool apply_mask){
        size_t batch_size = q->shape[0];
        size_t d_k = this->head_size;

        // Compute attention scores: Q @ K^T / sqrt(dk)
        this->scaled_scores = q->matmul(k->transpose()) / std::sqrt(d_k);
        if (this->mask != nullptr && apply_mask)
            this->scaled_scores += this->mask;
        
        this->attention = this->scaled_scores->softmax();

        Tensor_t<T> output = this->attention->matmul(v);
        
        return std::pair(output, this->attention);
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask=true){
        // """Forward pass through attention head"""
        
        // Compute Q, K, V projections
        Tensor_t<T> q = this->Q.forward(x);  // (B, T, head_size)
        Tensor_t<T> k = this->K.forward(x);  // (B, T, head_size) 
        Tensor_t<T> v = this->V.forward(x);  // (B, T, head_size)

        // Apply scaled dot-product attention
        auto res = this->scaled_dot_product_attention(q, k, v, apply_mask);
                
        Tensor_t<T> values = res.first;
        this->attention = res.second;

        return values;        
    }

};

#endif // !HEAD__HPP