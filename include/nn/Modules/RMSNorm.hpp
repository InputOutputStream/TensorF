#ifndef __RMS_NORM__HPP
#define __RMS_NORM__HPP

#include "Types/types.hpp"
#include "Module.hpp"
#include "DataStructures/Matrix.hpp"

#include <vector>
#include <memory>
#include <cmath>

// RMSNorm:  out = (x / rms(x)) * gamma
//   where   rms(x) = sqrt( mean(x²) + eps )
//
// Key differences from LayerNorm:
//   • does NOT subtract the mean  (no centring step)
//   • has NO beta / bias parameter

template <typename T>
class RMSNorm : public Module<T>{

    public:

        T eps;
        shape_t normalized_shape;
        Tensor_t<T> gamma;   // learnable scale 

    RMSNorm(shape_t normalized_shape, T eps = 1e-5)
        : eps(eps), normalized_shape(normalized_shape)
    {
        this->gamma = make_tensor<T>(Matrix<T>::ones(normalized_shape));
        this->register_parameter(gamma);
    }

    RMSNorm(std::initializer_list<size_t> inshape, T eps = 1e-5)
        : eps(eps)
    {
        this->normalized_shape = Matrix<T>::getShape(inshape);
        this->gamma = make_tensor<T>(Matrix<T>::ones(this->normalized_shape));
        this->register_parameter(gamma);
    }

    RMSNorm(RMSNorm&& other)
        : Module<T>(),
          eps(other.eps),
          normalized_shape(std::move(other.normalized_shape)),
          gamma(std::move(other.gamma))
    {
        this->register_parameter(gamma);
    }

    RMSNorm(const RMSNorm&) = delete;

    Tensor_t<T> forward(Tensor_t<T> x) {
        size_t axis = x->ndims - 1;

       
        // RMSNorm skips the mean-centring step entirely:
        //   rms = sqrt( mean(x²) + eps )
        //   out = gamma * (x / rms)

        auto x_sq  = x * x;                          // x²
        auto ms    = x_sq->mean(axis);               // mean(x²)  {B,T,1}
        auto rms   = (ms + this->eps)->sqrt();        // rms       {B,T,1}
        auto x_hat = x / rms;                        // normalise {B,T,D}
        return this->gamma * x_hat;                  // scale     {B,T,D}
    }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;
};

#endif // !__RMS_NORM__HPP