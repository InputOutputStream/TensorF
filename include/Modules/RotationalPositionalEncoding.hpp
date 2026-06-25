#ifndef _ROTATIONAL_POSITIONAL_ENCODING_H
#define _ROTATIONAL_POSITIONAL_ENCODING_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"

template <typename T>
class RotationalPositionalEncoding{
    public:
        size_t input_dim;
        size_t max_sequence_length;
        Tensor_t<T> PE;

    RotationalPositionalEncoding(size_t input_dim, size_t max_sequence_length){
        this->input_dim = input_dim;
        this->max_sequence_length = max_sequence_length;
    }

    Tensor_t<T> forward(Tensor_t<T> index){
        auto even_i = Matrix<T>::arrange(0, this->input_dim, 2);
        std::vector<T> tmp;
        for(auto i: even_i.data){
            tmp.push_back(std::pow(100000, (T)i/(T)input_dim));
        }
        Matrix<T> denominator(tmp);
        auto position = index->reshape({index->val.shape[index->ndims - 1], 1})->val;
        auto even_PE = Matrix<T>::sin(position/denominator);
        auto odd_PE = Matrix<T>::cos(position/denominator);
        auto stacked = Matrix<T>::concat({even_PE, odd_PE}, 1);
        this->PE = make_tensor<T>(stacked.reshape({index->val.get_size(), input_dim}));
        return this->PE;
    }
};

#endif 