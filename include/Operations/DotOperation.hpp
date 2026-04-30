#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __DOT_OPP_INCLUDED__
#define __DOT_OPP_INCLUDED__


template <typename T>
class DotOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    DotOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Dot Operation \n";
    }
      
};


/**
 * Dot Operation Implementation
*/

    template <typename T>
    void DotOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad.dot(this->t2->val.transpose()));
        this->t2->backward(this->t1->val.transpose().dot(grad));
    }

    template<typename T>
    Tensor_t<T> DotOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.dot(this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void DotOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad(); 
    }

    template <typename T>
    void DotOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph(); 
    }


#endif