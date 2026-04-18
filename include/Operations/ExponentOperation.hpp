#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __EXP_OPP_INCLUDED__
#define __EXP_OPP_INCLUDED__


template <typename T>
class ExponentOperation : public Operation<T>
{
    protected: 
        Tensor_t<T> tmp;

    public:
        Tensor_t<T> t1;
        
    ExponentOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    void zero_grad();


    void reset_graph();
    void to_string(){
        std::cout << "Exp Operation \n";
    }
};


/**
 * Exponential function Implementation
*/

    template <typename T>
    void ExponentOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad * this->tmp->data); 
    }

    template <typename T>
    Tensor_t<T> ExponentOperation<T>::forward()
    {
        this->tmp = std::make_shared<Tensor<T>>(this->t1->data.exponent());
        return std::make_shared<Tensor<T>>(this->t1->data.exponent(), this->shared_from_this());
    }

    template <typename T>
    void ExponentOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->tmp->zero_grad();
    }

    template <typename T>
    void ExponentOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }


#endif