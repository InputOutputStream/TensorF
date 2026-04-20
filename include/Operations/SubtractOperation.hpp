#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __SUBTRACT_OPP_INCLUDED__
#define __SUBTRACT_OPP_INCLUDED__


template <typename T>
class SubtractOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    SubtractOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();
    void zero_grad();

    void reset_graph();

    void to_string(){
        std::cout << "Subtract Operation \n";
    }
      
};



/**
 * Subtract Operation Implementation
*/

    template <typename T>
    void SubtractOperation<T>::backward(Matrix<T> grad)
    {
        if(grad.shape == this->t1->data.shape)
        {

            this->t1->backward(grad);
            this->t2->backward(-grad);
        }else
        {
            Matrix<T> grad1 = sumGradForBroadcast(grad, t1->data.shape);
            Matrix<T> grad2 = sumGradForBroadcast(-grad, t2->data.shape);
            
            this->t1->backward(grad1);
            this->t2->backward(grad2);
        }
        
        // Distributing Gradients when carrying out subtraction
    }

    
    template <typename T>
    void SubtractOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void SubtractOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }

    template<typename T>
    Tensor_t<T> SubtractOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data-this->t2->data, this->shared_from_this());
    }


#endif