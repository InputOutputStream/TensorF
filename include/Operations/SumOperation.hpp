#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __SUM_OPP_INCLUDED__
#define __SUM_OPP_INCLUDED__


template <typename T>
class SumOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;

    //..............................................................................................................

    SumOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Sum Operation \n";
    }
      
};



/**
 * Sum Operation Implementation
*/

    template <typename T>
    void SumOperation<T>::backward(Matrix<T> grad)
    {
        Matrix<T> ones = Matrix<T>::ones(this->t1->data.shape); 
        this->t1->backward(grad.data[0] * ones);
    }

    template<typename T>
    Tensor_t<T> SumOperation<T>::forward()
    {
        T s = this->t1->data.sum();
        return std::make_shared<Tensor<T>>(Matrix<T>({s}), this->shared_from_this());
    }

    template <typename T>
    void SumOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void SumOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }


#endif