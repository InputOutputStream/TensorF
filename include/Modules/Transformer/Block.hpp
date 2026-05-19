#ifndef __BLOCK__HPP
#define __BLOCK__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../Module.hpp"
#include "../FeedForward.hpp"
#include "../LayerNorm.hpp"
#include "../Linear.hpp"

#include "MultiHeadAttention.hpp"
#include "Head.hpp"

template <typename T>
class Block : public Module<T>{
    private:
        MultiHeadAttention<T> mha;
        FeedForward<T> ffwd;

        LayerNorm<T> ln1; 
        LayerNorm<T> ln2;

    public:

    Block(size_t input_dim, size_t sequence_length, size_t n_heads): 
        mha(input_dim / n_heads, input_dim, sequence_length, n_heads),
        ffwd(input_dim, 4 * input_dim, input_dim),
        ln1({input_dim}),  
        ln2({input_dim})  
    {

        this->register_module(&mha);
        this->register_module(&ffwd);
        this->register_module(&ln1);
        this->register_module(&ln2);

    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        Tensor_t<T> y = x + this->mha.forward(this->ln1.forward(x), apply_mask);
        Tensor_t<T> z = y + this->ffwd.forward(this->ln2.forward(y));
        return z;
    }

};

#endif // !__BLOCK__HPP