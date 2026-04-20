#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __MULTIPLY_OPP_INCLUDED__
#define __MULTIPLY_OPP_INCLUDED__


template <typename T>
class MultiplyOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;
        
    MultiplyOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 

    void zero_grad();

    void reset_graph();
    void to_string(){
        std::cout << "Multiply Operation \n";
    }
};



/**
 * Multiply Operation Implementation
*/

    template <typename T>
    void MultiplyOperation<T>::backward(Matrix<T> grad)
    {
        // Switching Gradients when carrying out product
        if(grad.shape == this->t1->data.shape)
        {
            this->t1->backward(grad);
            this->t2->backward(grad);   
        }else{
            Matrix<T> grad1 = sumGradForBroadcast(grad * this->t2->data, t1->data.shape);
            Matrix<T> grad2 = sumGradForBroadcast(grad * this->t1->data, t2->data.shape);

            this->t1->backward(grad1); 
            this->t2->backward(grad2);
        }
    }

    template <typename T>
    Tensor_t<T> MultiplyOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data*this->t2->data, this->shared_from_this());
    }

    template <typename T>
    void MultiplyOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void MultiplyOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }




#endif//............................................................................................................
