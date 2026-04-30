#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __ADD_OPP_INCLUDED__
#define __ADD_OPP_INCLUDED__


template <typename T>
class AddOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    AddOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Add Operation \n";
    }
      
};



/**
 * Add Operation Implementation
*/

    template <typename T>
    void AddOperation<T>::backward(Matrix<T> grad)
    {
        // Distributing Gradients when carrying out addition
        
        Matrix<T> grad1 = sumGradForBroadcast(grad, t1->val.shape);       
        Matrix<T> grad2 = sumGradForBroadcast(grad, t2->val.shape);

        this->t1->backward(grad1);
        this->t2->backward(grad2);
    }

    template<typename T>
    Tensor_t<T> AddOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val+this->t2->val, this->shared_from_this());
    }

    template <typename T>
    void AddOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void AddOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }




#endif