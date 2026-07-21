#ifndef __LORA_LINEAR__HPP_
#define __LORA_LINEAR__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

#include <iostream>


template <typename T>
class LoRALinear : public Module<T> {
public:
    Tensor_t<T> weight;   
    Tensor_t<T> bias;     
    bool sbias;
    Tensor_t<T> A;        // {in, rank} — trained
    Tensor_t<T> B;        // {rank, out} — trained
    T scale;              // alpha / rank

    LoRALinear(size_t out, size_t in, size_t rank, T alpha, bool sbias = true) {
        // NOTE: weight layout is {out, in} — matches Linear::weight so a pretrained
        // dense weight can be copied straight in via load_pretrained() below,
        // and transposed the same way Linear does in forward().
        weight = make_tensor<T>(Matrix<T>::zeros({out, in})); // load pretrained here
        this->sbias = sbias;
        bias = make_tensor<T>(Matrix<T>::zeros({out}));       // frozen, optional
        A = make_tensor<T>(Matrix<T>::randn({in, rank}));
        B = make_tensor<T>(Matrix<T>::zeros({rank, out}));
        scale = alpha / (T)rank;
        
        // only A and B are parameters — weight/bias are frozen
        this->register_parameter(A);
        this->register_parameter(B);
    }

    Tensor_t<T> forward(Tensor_t<T> x) {
        auto base = sbias ? (x->matmul(weight->transpose()) + bias)
                           : x->matmul(weight->transpose());   // frozen path
        auto lora = x->matmul(A)->matmul(B);                  // trained path
        return base + scale * lora;
    }

    // Copies a pretrained dense weight (and bias, if both sides have one) into
    // the frozen path. A/B are deliberately left alone — that's the whole point
    // of LoRA: the backbone weight is frozen at its pretrained value, only the
    // low-rank adapter trains.
    void load_pretrained(Linear<T>& src) {
        weight->val.copy_from(src.weight->val);
        if (sbias && src.sbias)
            bias->val.copy_from(src.bias->val);
    }
    
    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;
};

#endif