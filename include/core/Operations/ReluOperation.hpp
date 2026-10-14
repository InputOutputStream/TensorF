#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __RELU_OPP_INCLUDED__
#define __RELU_OPP_INCLUDED__


template <typename T>
class ReluOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
        Tensor_t<T> t1;

    //..............................................................................................................

    ReluOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad)override;

    Tensor_t<T> forward()override;

    void zero_grad()override;
    void reset_graph()override;

    void to_string() override {
        std::cout << "Relu Operation \n";
    }
      
};


/**
 * Relu Operation Implementation
*/

    template <typename T>
    void ReluOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->accumulate(grad * (this->t1->val > (T)0));
    }

    template<typename T>
    Tensor_t<T> ReluOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.maximum(0), this->shared_from_this());
    }

    template <typename T>
    void ReluOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void ReluOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
    }

#endif