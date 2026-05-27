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

        static size_t checked_head_size(size_t input_dim, size_t n_heads) {
            if (input_dim % n_heads != 0)
                throw std::runtime_error("Block: input_dim must be divisible by n_heads");
            return input_dim / n_heads;
        }

    public:

    Block(size_t input_dim, size_t sequence_length, size_t n_heads): 
        mha(checked_head_size(input_dim, n_heads), input_dim, sequence_length, n_heads),
        ffwd(input_dim, 4 * input_dim, input_dim),
        ln1({input_dim}),  
        ln2({input_dim})  
    {
        if (input_dim % n_heads != 0)
            throw std::runtime_error("Block: input_dim must be divisible by n_heads");

        this->register_module(&mha);
        this->register_module(&ffwd);
        this->register_module(&ln1);
        this->register_module(&ln2);

    }

    Block(Block&& other)
        : Module<T>(),          // fresh, empty submodules list
        mha(std::move(other.mha)),
        ffwd(std::move(other.ffwd)),
        ln1(std::move(other.ln1)),
        ln2(std::move(other.ln2))
    {
        // re-register at NEW addresses
        this->register_module(&mha);
        this->register_module(&ffwd);
        this->register_module(&ln1);
        this->register_module(&ln2);
    }

    Block(const Block&) = delete; 

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        // std::cerr << " block input x val : "<< x->val<< "\n";
        Tensor_t<T> lna = this->ln1.forward(x);
        // std::cerr << " layer norm 1 val : "<< lna->val<< "\n";

        Tensor_t<T> y = x + this->mha.forward(lna, apply_mask);
        // std::cerr << " mha out plus x val : "<< y->val<< "\n";

        Tensor_t<T> lnb = this->ln2.forward(y);
        // std::cerr << " layer norm 2 val : "<< lnb->val<< "\n";

        Tensor_t<T> z = y + this->ffwd.forward(lnb);

        // std::cerr << " block out z val : "<< z->val<< "\n";
        return z;
    }

};

#endif // !__BLOCK__HPP