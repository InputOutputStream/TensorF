#ifndef _POSITIONAL_ENCODING_H
#define _POSITIONAL_ENCODING_H

#include <vector>
#include <algorithm>
#include <cmath>
#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"

template <typename T>
class PositionalEncoding{
    public:
        size_t input_dim;
        size_t max_sequence_length;
        Tensor_t<T> PE;

    PositionalEncoding(size_t input_dim, size_t max_sequence_length){
        this->input_dim = input_dim;
        this->max_sequence_length = max_sequence_length;
    }

    Tensor_t<T> forward(Tensor_t<T> index){
        auto even_i = Matrix<T>::arrange(0, this->input_dim, 2);
        std::vector<T> denominator;
        for(auto i: even_i.data){
            denominator.push_back(std::pow(10000, (T)i/(T)input_dim));
        }
        auto position = index->val.reshape({index->val.shape[index->val.ndims - 1], 1});
        auto even_PE = Matrix<T>::sin(position/denominator);
        auto odd_PE = Matrix<T>::cos(position/denominator);
        auto stacked = Matrix<T>::stack({even_PE, odd_PE}, 2);
        this->PE = make_tensor<T>(Matrix<T>::reshape(stacked, {index->val.get_size(), input_dim}));
        return this->PE;
    }
};

#endif 