#ifndef __LLAMA_BLOCK__HPP
#define __LLAMA_BLOCK__HPP

#include <vector>
#include <memory>
#include <algorithm>

#include "../../Module.hpp"
#include "../../RMSNorm.hpp"

#include "MultiHeadAttention.hpp"
#include "Head.hpp"
#include "FeedForward.hpp"

template <typename T, template<typename> class LinearT>
class Block : public Module<T>{
    private:
        MultiHeadAttention<T, LinearT> mha;
        FeedForward<T, LinearT> ffwd;
        RMSNorm<T> rms1;
        RMSNorm<T> rms2;
        
        static size_t checked_head_size(size_t input_dim, size_t n_heads) {
            if (input_dim % n_heads != 0)
                throw std::runtime_error("Block: input_dim must be divisible by n_heads");
            return input_dim / n_heads;
        }

    public:

    template <typename... Args>
    Block(size_t input_dim, size_t sequence_length, size_t n_heads, size_t ffn_hidden, Args&&... args)
    : mha(checked_head_size(input_dim, n_heads), input_dim, sequence_length, n_heads, args...),
      ffwd(input_dim, ffn_hidden, input_dim, args...),  
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

    MultiHeadAttention<T, LinearT>& get_mha()  { return mha; }
    FeedForward<T, LinearT>&        get_ffwd() { return ffwd; }

    void load_pretrained(Block<T, Linear>& src) {
        mha.load_pretrained(src.get_mha());
        ffwd.load_pretrained(src.get_ffwd());
    }

    Tensor_t<T> forward(Tensor_t<T> x, bool apply_mask){

        Tensor_t<T> rms_a = this->rms1.forward(x);

        Tensor_t<T> y = x + this->mha.forward(rms_a, apply_mask);

        Tensor_t<T> rms_b = this->rms2.forward(y);

        Tensor_t<T> z = y + this->ffwd.forward(rms_b);

        return z;
    }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;

};

#endif // !__LLAMA_BLOCK__HPP