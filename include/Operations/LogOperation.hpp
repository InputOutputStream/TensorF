#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __LOG_OPP_INCLUDED__
#define __LOG_OPP_INCLUDED__


template <typename T>
class LogOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        
    LogOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    void zero_grad();


    void reset_graph();
    void to_string(){
        std::cout << "Log Operation \n";
    }
};


/**
 * Naperian Log function Implementation
*/

    template <typename T>
    void LogOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad * (1/this->t1->data)); 
    }

    template <typename T>
    Tensor_t<T> LogOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data, this->shared_from_this());
    }

    template <typename T>
    void LogOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void LogOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }


#endif