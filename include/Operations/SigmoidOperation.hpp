#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __SIGMOID_OPP_INCLUDED__
#define __SIGMOID_OPP_INCLUDED__


template <typename T>
class SigmoidOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;

    //..............................................................................................................

    SigmoidOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Sigmoid Operation \n";
    }
      
};


/**
 * Sigmoid Operation Implementation
*/

    template <typename T>
    void SigmoidOperation<T>::backward(Matrix<T> grad)
    {
        Matrix<T> temp = this->t1->data * (Matrix<T>(1) + (Matrix<T>(-1)*this->t1->data));
        this->t1->backward(grad * temp);
    }

    template<typename T>
    Tensor_t<T> SigmoidOperation<T>::forward()
    {
        auto temp = ((T)1 / ((T)1 + ((T)-1*this->t1->data).exponent()));
        return std::make_shared<Tensor<T>>(temp, this->shared_from_this());
    }

    template <typename T>
    void SigmoidOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void SigmoidOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }


#endif