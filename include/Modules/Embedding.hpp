#ifndef __EMBEDDING_H_
#define __EMBEDDING_H_

#include <cmath>

#include "../Types/types.hpp"
#include "Linear.hpp"
#include "Optimizer.hpp"
#include "Module.hpp"

template< typename T>
class Embedding: Module<T>{
    public:
    Tensor_t<T> embeddings;
    size_t vocab_size;
    size_t input_dim;

    Embedding(size_t vocab_size, size_t input_dim){

        // Initialize embeddings with Glorot uniform initialization
        auto limit = std::sqrt((T)6.0 / (T)(vocab_size + input_dim));
        this->embeddings = make_tensor<T>(Matrix<T>::randu(-limit, limit, {vocab_size, input_dim}));
        this->register_parameter (this->embeddings);
    }  

    Tensor_t<T> forward(Tensor_t<T> indices){
        return make_tensor<T>(this->embeddings->elemsAt(indices));
    }
};

#endif // !__EMBEDDING_H_