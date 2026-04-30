#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __MATMUL_OPP_INCLUDED__
#define __MATMUL_OPP_INCLUDED__


template <typename T>
class MatmulOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    MatmulOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Matmul Operation \n";
    }
      
};



/**
 * Matmul Operation Implementation
*/

    template <typename T>
    void MatmulOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad.matmul(this->t2->val.transpose()));
        this->t2->backward(this->t1->val.transpose().matmul(grad));
    }

    template<typename T>
    Tensor_t<T> MatmulOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.matmul(this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void MatmulOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad(); 
    }

    template <typename T>
    void MatmulOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph(); 
    }


#endif