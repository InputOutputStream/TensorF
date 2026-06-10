#ifndef __LLAMA_BLOCK__HPP
#define __LLAMA_BLOCK__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../../Module.hpp"
#include "../../RMSNorm.hpp"
#include "../../Linear.hpp"

#include "MultiHeadAttention.hpp"
#include "Head.hpp"
#include "FeedForward.hpp"

template <typename T>
class Block : public Module<T>{
    private:
        MultiHeadAttention<T> mha;
        FeedForward<T> ffwd;
        RMSNorm<T> rms1;
        RMSNorm<T> rms2;
        
        static size_t checked_head_size(size_t input_dim, size_t n_heads) {
            if (input_dim % n_heads != 0)
                throw std::runtime_error("Block: input_dim must be divisible by n_heads");
            return input_dim / n_heads;
        }

    public:

    Block(size_t input_dim, size_t sequence_length, size_t n_heads, size_t ffn_hidden)
    : mha(checked_head_size(input_dim, n_heads), input_dim, sequence_length, n_heads),
      ffwd(input_dim, ffn_hidden, input_dim),  
      rms1({input_dim}),
      rms2({input_dim})
   {
        this->register_module(&mha);
        this->register_module(&ffwd);
        this->register_module(&rms1);
        this->register_module(&rms2);
    }

    Block(Block&& other)
        : Module<T>(),
        mha(std::move(other.mha)),
        ffwd(std::move(other.ffwd)),
        rms1(std::move(other.rms1)),
        rms2(std::move(other.rms2))
    {
        this->register_module(&mha);
        this->register_module(&ffwd);
        this->register_module(&rms1);
        this->register_module(&rms2);
    }

    Block(const Block&) = delete;

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        Tensor_t<T> rms_a = this->rms1.forward(x);

        Tensor_t<T> y = x + this->mha.forward(rms_a, apply_mask);

        Tensor_t<T> rms_b = this->rms2.forward(y);

        Tensor_t<T> z = y + this->ffwd.forward(rms_b);

        return z;
    }

    friend class GGUFLoader<T>;
    friend class LlamaGGUFLoader<T>; 

};

#endif // !__LLAMA_BLOCK__HPP