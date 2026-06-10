#ifndef __MHA__HPP
#define __MHA__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../../Module.hpp"
#include "../../Linear.hpp"

#include "Head.hpp"

template <typename T>
class MultiHeadAttention : public Module<T>{
    private:
        size_t n_heads;
        Linear<T> proJ;
        std::vector<std::unique_ptr<Head<T>>> Heads;

    public:

    MultiHeadAttention(size_t head_size, size_t input_dim, size_t sequence_length, size_t n_heads): 
    proJ(head_size * n_heads, input_dim, true)
    {
        this->n_heads = n_heads;
        
        Heads.reserve(n_heads);
        for(size_t i = 0; i < n_heads; i++) {
            Heads.push_back(std::make_unique<Head<T>>(head_size, input_dim, sequence_length));
            this->register_module(Heads.back().get());
        }
        this->register_module(&proJ);
    }

    MultiHeadAttention(MultiHeadAttention&& other)
        : Module<T>(), // fresh, empty submodules list
        n_heads(other.n_heads),
        proJ(std::move(other.proJ)),
        Heads(std::move(other.Heads))
    {
        for(auto& h : Heads)
            this->register_module(h.get());
        this->register_module(&proJ);
    }

    MultiHeadAttention(const MultiHeadAttention&) = delete;


    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        std::vector<Tensor_t<T>> head_outs;
        
        for(auto& head : this->Heads)
            head_outs.push_back(head->forward(x, apply_mask));

        // concat on last axis: {B,T,head_size} * n_heads -> {B,T,n_heads*head_size}
        Tensor_t<T> out = Tensor<T>::concat(head_outs, 2);
        // std::cerr << " mha concat out val : "<< out->val<< "\n";
    
        return this->proJ.forward(out);
    }

    friend class GGUFLoader<T>;
    friend class GPTGGUFLoader<T>; 
};

#endif // !__MHA__HPP