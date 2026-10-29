#include "Types/types.hpp"
#include "Operation.hpp"


#ifndef __POWER_OPP_INCLUDED__
#define __POWER_OPP_INCLUDED__


template <typename T>
class PowerOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
        Tensor_t<T> t1, t2;
        Tensor_t<T> tmp;     // ← saved forward output, needed for t2's gradient

    PowerOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }  

    void backward(Matrix<T> grad)override;

    Tensor_t<T> forward()override; 

    void zero_grad()override;

    void reset_graph()override;
    void to_string() override{
        std::cout << "Power Operation \n";
    }
};



/**
 * Power Operation Implementation
*/

    template <typename T>
    void PowerOperation<T>::backward(Matrix<T> grad)
    {
         // d/da (a^b) = b · a^(b-1)
        if (this->t1->requires_grad)
            this->t1->accumulate(grad * this->t2->val
                                 * pow(this->t1->val, this->t2->val - Matrix<T>(1)));

        // d/db (a^b) = a^b · ln(a)   — real-valued only where a > 0
        if (this->t2->requires_grad)
            this->t2->accumulate(grad * this->tmp->val * this->t1->val.ln());
    }

    template <typename T>
    Tensor_t<T> PowerOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(pow(this->t1->val, this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void PowerOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void PowerOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
        if (this->t2) {
            this->t2->reset_graph();
            this->t2 = nullptr; // Drop strong reference
        }        
        if (this->tmp) { this->tmp->reset_graph(); this->tmp = nullptr; }

    }


#endif//............................................................................................................
