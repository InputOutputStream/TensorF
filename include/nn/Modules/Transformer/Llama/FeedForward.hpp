#ifndef __LLAMA_FEED_FORWARD_H
#define __LLAMA_FEED_FORWARD_H

#include "Types/types.hpp"
#include "../../Optimizer.hpp"
#include "../../Module.hpp"

// LLaMA FFN uses SwiGLU:
//   hidden = silu(gate_proj(x)) * up_proj(x)
//   out    = down_proj(hidden)
// where silu(x) = x * sigmoid(x)
//

template <typename T, template<typename> class LinearT>
class FeedForward: public Module<T>{
            
    private:
        LinearT<T> gate_proj;   // W_gate : in  -> hidden
        LinearT<T> up_proj;     // W_up   : in  -> hidden
        LinearT<T> down_proj;   // W_down : hidden -> out

        // Constructs a LinearT<T> with bias forced off, regardless of what
        // Args the enclosing Block/Llama was instantiated with. Two overloads:
        // one for LinearT's that take a bias bool (Linear), one for LinearT's
        // that don't (LoRALinear takes rank/alpha only — no bias flag exists
        // to force off, so it's a plain forward).
        template <typename... Args>
        static LinearT<T> make_linear(size_t out, size_t in, Args&&... args) {
            if constexpr (std::is_constructible_v<LinearT<T>, size_t, size_t, bool>) {
                return LinearT<T>(out, in, false);   // Linear: force bias=false
            } else {
                return LinearT<T>(out, in, args...); // LoRALinear etc: no bias concept, pass through
            }
        }

    public:

    template <typename... Args>
    FeedForward(size_t in_features, size_t hidden, size_t out_features, Args&&... args)
    : gate_proj(make_linear(hidden, in_features, args...)),
      up_proj  (make_linear(hidden, in_features, args...)),
      down_proj(make_linear(out_features, hidden, args...))
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

    LinearT<T>& get_gate() { return gate_proj; }
    LinearT<T>& get_up()   { return up_proj; }
    LinearT<T>& get_down() { return down_proj; }

    void load_pretrained(FeedForward<T, Linear>& src) {
        gate_proj.load_pretrained(src.get_gate());
        up_proj.load_pretrained(src.get_up());
        down_proj.load_pretrained(src.get_down());
    }

    Tensor_t<T> forward(Tensor_t<T> x) {
        Tensor_t<T> gate_raw = gate_proj.forward(x);
        Tensor_t<T> gate = Tensor<T>::SiLU(gate_raw);  // SiLU
        Tensor_t<T> up = up_proj.forward(x);

        // Element-wise gating
        Tensor_t<T> hidden = gate * up;
        return down_proj.forward(hidden);
    }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;

};

#endif // !__LLAMA_FEED_FORWARD_H

//  attention/lm_head get bias=true, FFN is always bias-free regardless
// Llama<float, Linear> model(vocab_size, d_model, block_size, n_head, n_layer, ffn_hidden, /*bias=*/true);