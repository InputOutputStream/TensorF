#include "Types/types.hpp"
#include "Operation.hpp"
#include "MatmulOperation.hpp"

#ifndef __DOT_OPP_INCLUDED__
#define __DOT_OPP_INCLUDED__


template <typename T>
class DotOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    DotOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();

    void zero_grad();
    void reset_graph();

    void to_string(){
        std::cout << "Dot Operation \n";
    }
      
};


/**
 * Dot Operation Implementation
*/

    template <typename T>
    void DotOperation<T>::backward(Matrix<T> grad)
    {
        const Matrix<T>& A = this->t1->val;
        const Matrix<T>& B = this->t2->val;
        // Matrix::dot: 1D.1D or equal-shape 2D.2D -> flattened inner product (scalar result)
        if ((A.shape.size() == 1 && B.shape.size() == 1) ||
            (A.shape.size() == 2 && B.shape.size() == 2 && A.shape == B.shape)) {
            T g = grad.data.at(0);
            this->t1->backward(B * g);
            this->t2->backward(A * g);
            return;
        }
        auto gr = MatmulOperation<T>::matmul_grads(A, B, grad);
        this->t1->backward(gr.first);
        this->t2->backward(gr.second);
    }

    template<typename T>
    Tensor_t<T> DotOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.dot(this->t2->val), this->shared_from_this());
    }

    template <typename T>
    void DotOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad(); 
    }

    template <typename T>
    void DotOperation<T>::reset_graph(){
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