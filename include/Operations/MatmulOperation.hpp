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
        // If 3D batched tensor, swap ONLY the last two dims (seq_len and features)
        Matrix<T> b_T = (this->t2->val.shape.size() == 3) 
                            ? this->t2->val.transpose({0, 2, 1}) 
                            : this->t2->val.transpose();

        Matrix<T> a_T = (this->t1->val.shape.size() == 3) 
                            ? this->t1->val.transpose({0, 2, 1}) 
                            : this->t1->val.transpose();

        // Compute raw gradients
        Matrix<T> raw_grad1 = grad.matmul(b_T);
        Matrix<T> raw_grad2 = a_T.matmul(grad);

        // Sum over batch/broadcasted dimensions so 2D weights get 2D gradients!
        Matrix<T> grad1 = sumGradForBroadcast(raw_grad1, this->t1->val.shape);
        Matrix<T> grad2 = sumGradForBroadcast(raw_grad2, this->t2->val.shape);

        this->t1->backward(grad1);
        this->t2->backward(grad2);
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