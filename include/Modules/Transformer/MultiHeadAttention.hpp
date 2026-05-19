#ifndef __MHA__HPP
#define __MHA__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../Module.hpp"
#include "../Linear.hpp"

#include "Head.hpp"

template <typename T>
class MultiHeadAttention : public Module<T>{
    private:
        size_t n_heads;
        Linear<T> proJ;
        std::vector<Head<T>> Heads;

    public:

    MultiHeadAttention(size_t head_size, size_t input_dim, size_t sequence_length, size_t n_heads): 
    proJ(head_size*n_heads, input_dim, true)
    {
        this->n_heads = n_heads;

        for(size_t i = 0; i < n_heads; i++)
                Heads.emplace_back(head_size, input_dim, sequence_length);
        for(auto& b : Heads)
            this->register_module(&b);

        this->register_module(&proJ);
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        std::vector<Matrix<T>> res;
        
        for(auto& head: this->Heads)
        {
            Tensor_t<T> out = head.forward(x, apply_mask);
            res.push_back(out->val);
        }
        Tensor_t<T> out = make_tensor<T>(Matrix<T>::stack(res, 2));
        out  = this->proJ.forward(out);
        return out;
    }

};

#endif // !__MHA__HPP