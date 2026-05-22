

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
    private:
        Tensor_t<T> mean;
        Tensor_t<T> diff;
        Tensor_t<T> var;
        Tensor_t<T> stdev;
        Tensor_t<T> y;

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

    LayerNorm(std::initializer_list<size_t> inshape, T eps = 1e-5, T tol = 1e-9)
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
        size_t D = x->val.shape.back();
        this->mean = x->sum(x->ndims - 1) / make_tensor<T>((T)D);
        this->diff = x - mean;                          // {B,T,D} - {B,T,1} broadcasts
        this->var = pow(diff, (T)2)->sum(x->ndims - 1) / make_tensor<T>((T)D);
        this->stdev = (var + make_tensor<T>(this->eps))->sqrt();  // sqrt(var + eps)
        this->y = diff / stdev;                      // {B,T,D} / {B,T,1} broadcasts
        std::cerr <<"ln shape: " << this->y->shape;
        return this->gamma * y + this->beta;
    }
};

 #endif // !__LAYER__NORM__