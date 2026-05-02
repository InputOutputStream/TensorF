#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __POWER_OPP_INCLUDED__
#define __POWER_OPP_INCLUDED__


template <typename T>
class PowerOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;
        
    PowerOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 

    void zero_grad();

    void reset_graph();
    void to_string(){
        std::cout << "Power Operation \n";
    }
};



/**
 * Power Operation Implementation
*/

    template <typename T>
    void PowerOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad * this->t2->val * (this->t1->val ^ (this->t2->val - Matrix<T>(1))));
    }

    template <typename T>
    Tensor_t<T> PowerOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val ^ (this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void PowerOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void PowerOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }




#endif//............................................................................................................
