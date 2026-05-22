#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __RESHAPE_OPP_INCLUDED__
#define __RESHAPE_OPP_INCLUDED__


template <typename T>
class ReshapeOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        shape_t orig_shape, n_shape;

        
    ReshapeOperation(Tensor_t<T> t1, shape_t new_shape)
    {
        this->t1 = t1;
        n_shape = new_shape;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    
    void zero_grad();
    void reset_graph();
    void to_string(){
        std::cout << "Reshape Operation \n";
    }
};


template <typename T>
Tensor_t<T> ReshapeOperation<T>::forward()
{
    this->orig_shape = this->t1->val.shape;
    return std::make_shared<Tensor<T>>(Matrix<T>(this->t1->val.data, this->n_shape), this->shared_from_this());
}


template <typename T>
void ReshapeOperation<T>::backward(Matrix<T> grad)
{
    this->t1->backward(Matrix<T>(grad.data, this->orig_shape));
}

template <typename T>
void ReshapeOperation<T>::zero_grad(){
    this->t1->zero_grad(); 
}

template <typename T>
void ReshapeOperation<T>::reset_graph(){
    this->t1->reset_graph(); 
}

#endif