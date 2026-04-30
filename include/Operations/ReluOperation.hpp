#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __RELU_OPP_INCLUDED__
#define __RELU_OPP_INCLUDED__


template <typename T>
class ReluOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;

    //..............................................................................................................

    ReluOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Relu Operation \n";
    }
      
};


/**
 * Relu Operation Implementation
*/

    template <typename T>
    void ReluOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad * (this->t1->val > (T)0));
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
        this->t1->reset_graph(); 
    }


#endif