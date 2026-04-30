#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __DIVIDE_OPP_INCLUDED__
#define __DIVIDE_OPP_INCLUDED__


template <typename T>
class DivisionOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;
        
    DivisionOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    void zero_grad();

    void reset_graph();

    void to_string(){
        std::cout << "Divide Operation \n";
    }
};


/**
 * Divide Operation Implementation
*/
 // forward is :x/y
 // backward will be the derivative of both wrt to x and wrt y
 // wrt x: grad/y
 // wrt y: (grad *x * (-1)) / y^2

    template <typename T>
    void DivisionOperation<T>::backward(Matrix<T> grad)
    {
        Matrix<T> grad1 = sumGradForBroadcast(grad / this->t2->val, this->t1->val.shape);
        Matrix<T> numerator = Matrix<T>(-1) * grad * this->t1->val;
        Matrix<T> t2_sq = this->t2->val * this->t2->val;
        Matrix<T> grad2 = sumGradForBroadcast(numerator / t2_sq, this->t2->val.shape);

        this->t1->backward(grad1);
        this->t2->backward(grad2);
    }

    template <typename T>
    void DivisionOperation<T>::zero_grad(){
        this->t1->zero_grad();
        this->t2->zero_grad();
    }

    template <typename T>
    void DivisionOperation<T>::reset_graph(){
        this->t1->reset_graph();
        this->t2->reset_graph();
    }

    template <typename T>
    Tensor_t<T> DivisionOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>((this->t1->val/this->t2->val), this->shared_from_this());
    }

#endif