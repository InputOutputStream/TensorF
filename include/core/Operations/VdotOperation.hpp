#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __VDOT_OPP_INCLUDED__
#define __VDOT_OPP_INCLUDED__


template <typename T>
class VdotOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1, this->t2}; }
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    VdotOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad) override;

    Tensor_t<T> forward() override;

    void zero_grad() override;
    void reset_graph() override;

    void to_string() override{
        std::cout << "Vdot Operation \n";
    }
      
};


/**
 * Vdot Operation Implementation
*/

    template <typename T>
    void VdotOperation<T>::backward(Matrix<T> grad)
    {
        const Matrix<T>& A = this->t1->val;
        const Matrix<T>& B = this->t2->val;
        T g = grad.data.at(0);
        this->t1->accumulate((B * g).reshape(A.shape));
        this->t2->accumulate((A * g).reshape(B.shape));
    }

    template<typename T>
    Tensor_t<T> VdotOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.vdot(this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void VdotOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad(); 
    }

    template <typename T>
    void VdotOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
        if (this->t2) {
            this->t2->reset_graph();
            this->t2 = nullptr; // Drop strong reference
        }
    }

#endif