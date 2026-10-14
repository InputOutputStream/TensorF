#include "Types/types.hpp"
#include "Operation.hpp"


#ifndef __LOG_OPP_INCLUDED__
#define __LOG_OPP_INCLUDED__


template <typename T>
class LogOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
        Tensor_t<T> t1;
        
    LogOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }  

    void backward(Matrix<T> grad) override;

    Tensor_t<T> forward() override; 
    void zero_grad() override;


    void reset_graph() override;
    void to_string() override{
        std::cout << "Log Operation \n";
    }
};


/**
 * Natural Log function Implementation
*/

    template <typename T>
    void LogOperation<T>::backward(Matrix<T> grad)
    {
        // forward clamps x to >= 1e-9 for float-like T, so the derivative there is 0 (not 1/x)
        const auto& x = this->t1->val;
        using A = scalar::acc_t<T>;
        std::vector<T> d(x.data.size());
        for (size_t i = 0; i < d.size(); i++) {
            A v = scalar::to_acc<T>(x.data[i]);
            if constexpr (mxd::is_float_like_v<T>) d[i] = (v > static_cast<A>(1e-9)) ? scalar::from_acc<T>(static_cast<A>(1) / v) : T(0);
            else d[i] = scalar::from_acc<T>(static_cast<A>(1) / v);
        }
        this->t1->accumulate(grad * Matrix<T>(std::move(d), x.shape));
    }

    template <typename T>
    Tensor_t<T> LogOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.ln(), this->shared_from_this());
    }

    template <typename T>
    void LogOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void LogOperation<T>::reset_graph() {
         if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
    }

#endif