#include "Types/types.hpp"
#include "Operation.hpp"


#ifndef __SIGMOID_OPP_INCLUDED__
#define __SIGMOID_OPP_INCLUDED__


template <typename T>
class SigmoidOperation : public Operation<T>
{
    protected:
    Tensor_t<T> tmp;
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
        Tensor_t<T> t1;

    //..............................................................................................................

    SigmoidOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad)override;

    Tensor_t<T> forward()override;

    void zero_grad()override;
    void reset_graph()override;

    void to_string() override {
        std::cout << "Sigmoid Operation \n";
    }
      
};


/**
 * Sigmoid Operation Implementation
*/

    template <typename T>
    void SigmoidOperation<T>::backward(Matrix<T> grad)
    {
        if (this->t1->requires_grad) this->t1->accumulate(grad * (this->tmp->val * (Matrix<T>(1) + (Matrix<T>(-1)*this->tmp->val))));
    }

    template<typename T>
    Tensor_t<T> SigmoidOperation<T>::forward()
    {
        auto temp = (Matrix<T>(1) / ((T)1 + (Matrix<T>(-1)*this->t1->val).exponent()));
        this->tmp = std::make_shared<Tensor<T>>(temp);
        return std::make_shared<Tensor<T>>(temp, this->shared_from_this());
    }

    template <typename T>
    void SigmoidOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        if (this->tmp) this->tmp->zero_grad();
    }

    template <typename T>
    void SigmoidOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }

        if (this->tmp) {
            this->tmp->reset_graph();
            this->tmp = nullptr; // Drop strong reference
        }
    }

#endif