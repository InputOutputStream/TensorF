#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __SUM_OPP_INCLUDED__
#define __SUM_OPP_INCLUDED__


template <typename T>
class SumOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1}; }
        Tensor_t<T> t1;

    //..............................................................................................................

    SumOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }

    void backward(Matrix<T> grad) override;

    Tensor_t<T> forward() override;

    void zero_grad() override;
    void reset_graph() override ;

    void to_string() override{ 
        std::cout << "Sum Operation \n";
    }
      
};



/**
 * Sum Operation Implementation
*/

    template <typename T>
    void SumOperation<T>::backward(Matrix<T> grad)
    {
        if (this->t1->requires_grad) this->t1->accumulate(grad.data[0] * Matrix<T>::ones(this->t1->val.shape));
    }

    template<typename T>
    Tensor_t<T> SumOperation<T>::forward()
    {
        T s = this->t1->val.sum();
        return std::make_shared<Tensor<T>>(Matrix<T>(s), this->shared_from_this());
    }

    template <typename T>
    void SumOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void SumOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
    }


#endif