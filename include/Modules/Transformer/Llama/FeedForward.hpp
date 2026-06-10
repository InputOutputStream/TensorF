#ifndef __LLAMA_FEED_FORWARD_H
#define __LLAMA_FEED_FORWARD_H

#include "../../../Types/types.hpp"
#include "../../Linear.hpp"
#include "../../Optimizer.hpp"
#include "../../Module.hpp"

// LLaMA FFN uses SwiGLU:
//   hidden = silu(gate_proj(x)) * up_proj(x)
//   out    = down_proj(hidden)
// where silu(x) = x * sigmoid(x)

template <typename T>
class FeedForward: public Module<T>{
            
    private:
        Linear<T> gate_proj;   // W_gate : in  -> hidden
        Linear<T> up_proj;     // W_up   : in  -> hidden
        Linear<T> down_proj;   // W_down : hidden -> out

    public:

    FeedForward(size_t in_features, size_t hidden, size_t out_features)
    : gate_proj(in_features, hidden, false),   
      up_proj  (in_features, hidden, false),
      down_proj(hidden, out_features, false)
    {
        this->register_module(&gate_proj);
        this->register_module(&up_proj);
        this->register_module(&down_proj);
    }

    FeedForward(FeedForward&& other)
        : Module<T>(),
        gate_proj(std::move(other.gate_proj)),
        up_proj  (std::move(other.up_proj)),
        down_proj(std::move(other.down_proj))
    {
        this->register_module(&gate_proj);
        this->register_module(&up_proj);
        this->register_module(&down_proj);
    }

    FeedForward(const FeedForward&) = delete;

    Tensor_t<T> forward(Tensor_t<T> x) {
        Tensor_t<T> gate_raw = gate_proj.forward(x);
        Tensor_t<T> gate = gate_raw * gate_raw->sigmoid();  // SiLU

        Tensor_t<T> up = up_proj.forward(x);

        // Element-wise gating
        Tensor_t<T> hidden = gate * up;

        return down_proj.forward(hidden);
    }

    friend class GGUFLoader<T>;
    friend class LlamaGGUFLoader<T>; 

};

#endif // !__LLAMA_FEED_FORWARD_H