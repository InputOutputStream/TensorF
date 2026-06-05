

#ifndef __LAYER__NORM__
#define __LAYER__NORM__

#include "../Types/types.hpp"
#include "Module.hpp"
#include "../DataStructures/Matrix.hpp"

#include <vector>
#include <memory>
#include <cmath>


template <typename T>
class LayerNorm : public Module<T>{

    public:

        T eps = 1e-5;
        T tol = 1e-9;
        shape_t  normalized_shape;
        Tensor_t<T> gamma;
        Tensor_t<T> beta;

    LayerNorm(shape_t normalized_shape, T eps = 1e-5, T tol = 1e-9)
    {
        this->eps = eps;
        this->tol = tol;
        this->normalized_shape = normalized_shape;
        this->gamma = make_tensor<T>(Matrix<T>::ones(normalized_shape)) ;
        this->beta = make_tensor<T>(Matrix<T>::zeros(normalized_shape));

        this->register_parameter(gamma);
        this->register_parameter(beta);
    }

    LayerNorm(std::initializer_list<size_t> inshape, T eps = 1e-5, T tol = 1e-7)
    {
        shape_t normalized_shape = Matrix<T>::getShape(inshape);
        this->eps = eps;
        this->tol = tol;
        this->normalized_shape = normalized_shape;
        this->gamma = make_tensor<T>(Matrix<T>::ones(normalized_shape)) ;
        this->beta = make_tensor<T>(Matrix<T>::zeros(normalized_shape));

        this->register_parameter(gamma);
        this->register_parameter(beta);
    }

    Tensor_t<T> forward(Tensor_t<T> x) {
        size_t axis = x->ndims - 1;
        auto mu    = x->mean(axis);                     // {B,T,1}
        auto var   = x->var(axis);                      // {B,T,1}
        auto stdev = (var + this->eps)->sqrt();  // uses operator+(Tensor_t<T>, const T)
        auto xhat  = (x - mu) / stdev;                  // {B,T,D}
        return this->gamma * xhat + this->beta;
    }

    friend class GGUFLoader<T>;
};

 #endif // !__LAYER__NORM__



/**
 * LayerNormOperation
 *
 * Normalises over the LAST axis (the feature/embedding dim), which is the
 * standard for transformer layer-norm.  gamma (scale) and beta (shift) are
 * learnable Tensors with the same shape as the last dim.
 *
 * forward:
 *   mu    = mean(x, axis=-1)              shape: [..., 1]
 *   sigma = sqrt(var(x, axis=-1) + eps)   shape: [..., 1]
 *   x_hat = (x - mu) / sigma              shape: same as x
 *   y     = gamma * x_hat + beta          shape: same as x
 */