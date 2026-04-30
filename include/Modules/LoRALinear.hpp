#ifndef __LORA_LINEAR__HPP_
#define __LORA_LINEAR__HPP_

#include "../Types/types.hpp"
#include "../DataStructures/Tensor.hpp"

#include "Module.hpp"

#include <iostream>


template <typename T>
class LoRALinear : public Module<T> {
public:
    Tensor_t<T> weight;   // frozen — not registered as parameter
    Tensor_t<T> A;        // {in, rank} — trained
    Tensor_t<T> B;        // {rank, out} — trained
    T scale;              // alpha / rank

    LoRALinear(size_t in, size_t out, size_t rank, T alpha) {
        weight = make_tensor<T>(Matrix<T>::zeros({in, out})); // load pretrained here
        A = make_tensor<T>(Matrix<T>::randn({in, rank}));
        B = make_tensor<T>(Matrix<T>::zeros({rank, out}));
        scale = alpha / (T)rank;
        
        // only A and B are parameters — weight is frozen
        this->register_parameter(A);
        this->register_parameter(B);
    }

    Tensor_t<T> forward(Tensor_t<T> x) {
        auto base = x->matmul(weight);          // frozen path
        auto lora = x->matmul(A)->matmul(B);    // trained path
        return base + scale * lora;
    }
};

#endif