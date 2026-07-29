#ifndef _POSITIONAL_ENCODING_H
#define _POSITIONAL_ENCODING_H

#include <vector>
#include <cmath>
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "Module.hpp"

template <typename T>
class PositionalEncoding : public Module<T> {
    public:
        size_t input_dim;
        size_t max_sequence_length;
        Tensor_t<T> weight;  // shape: {max_sequence_length, input_dim}

    PositionalEncoding(size_t input_dim, size_t max_sequence_length)
        : input_dim(input_dim), max_sequence_length(max_sequence_length)
    {
        auto limit = std::sqrt((T)6.0 / (T)(max_sequence_length + input_dim));
        this->weight = make_tensor<T>(Matrix<T>::randu(-limit, limit, {max_sequence_length, input_dim}));
        this->register_parameter(this->weight);
    }

    // move constructor so it can live in GPT's member list
    PositionalEncoding(PositionalEncoding&& other)
        : Module<T>(),
          input_dim(other.input_dim),
          max_sequence_length(other.max_sequence_length),
          weight(std::move(other.weight))
    {
        this->register_parameter(this->weight);
    }

    PositionalEncoding(const PositionalEncoding&) = delete;

    Tensor_t<T> forward(Tensor_t<T> indices) {
        // std::cerr << " PositionalEncoding this->weight->embed(indices) val : "<< this->weight->embed(indices)->val<< "\n";
        return this->weight->embed(indices);
    }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
    template <typename, template<typename> class> friend class LlamaGGUFLoader; 

};

#endif