

#ifndef __LAYER__NORM__
#define __LAYER__NORM__

#include "../Types/types.hpp"
#include "Module.hpp"
#include "../DataStructures/Matrix.hpp"

#include <vector>
#include <memory>
#include <cmath>


/**

LayerNorm.hpp — subtle correctness issue:

x->sum(0) gives a sum, not a mean, 
and dividing by shape[0] only works for 1D or if you're 
normalizing over the first axis. Standard LayerNorm normalizes 
over the last dimension (the feature dim), not the batch/sequence dim.

*/

template <typename T>
class LayerNorm : public Module<T>{
    private:
        Tensor_t<T> mean;
        Tensor_t<T> diff;
        Tensor_t<T> var;
        Tensor_t<T> std;
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

    Tensor_t<T> forward(Tensor_t<T> x)
    {

        this->mean = x->sum(x->ndims - 1) / make_tensor<T>((T)x->val.shape[x->ndims-1]);
        this->diff = x - mean;
        this->var  = (diff ^ (T)2)->sum(x->ndims - 1) / make_tensor<T>((T)x->val.shape[x->ndims-1]);
        this->std  = var->sqrt() + make_tensor<T>(this->eps);
        this->y    = diff / std;
        return this->gamma * y + this->beta;
    }
};

 #endif // !__LAYER__NORM__