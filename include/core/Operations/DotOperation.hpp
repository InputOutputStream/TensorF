#include "Types/types.hpp"
#include "Operation.hpp"
#include "MatmulOperation.hpp"

#ifndef __DOT_OPP_INCLUDED__
#define __DOT_OPP_INCLUDED__


template <typename T>
class DotOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1, this->t2}; }
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    DotOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad) override;

    Tensor_t<T> forward() override;

    void zero_grad() override;
    void reset_graph() override;

    void to_string() override {
        std::cout << "Dot Operation \n";
    }
      
};


/**
 * Dot Operation Implementation
*/

    template <typename T>
    void DotOperation<T>::backward(Matrix<T> grad)
    {
        if (!this->t1->requires_grad && !this->t2->requires_grad) return;

        const Matrix<T>& A = this->t1->val;
        const Matrix<T>& B = this->t2->val;
        const size_t ra = A.shape.size(), rb = B.shape.size();

        if (ra == 1 && rb == 1) {
            T g = grad.data.at(0);
            if (this->t1->requires_grad) this->t1->accumulate(B * g);
            if (this->t2->requires_grad) this->t2->accumulate(A * g);
            return;
        }
        if (rb == 1 || (ra == 2 && rb == 2)) {
            auto gr = MatmulOperation<T>::matmul_grads(A, B, grad);
            if (this->t1->requires_grad) this->t1->accumulate(gr.first);
            if (this->t2->requires_grad) this->t2->accumulate(gr.second);
            return;
        }
        // general numpy dot
        const size_t K = A.shape.back();
        const size_t M = K ? A.data.size() / K : 0, N = K ? B.data.size() / K : 0;
        const shape_t perm = Matrix<T>::dot_rhs_perm(rb);
        shape_t inv(rb);
        for (size_t i = 0; i < rb; i++) inv[perm[i]] = i;
        Matrix<T> G2 = grad.reshape(shape_t{M, N});
        Matrix<T> A2 = A.reshape(shape_t{M, K});
        Matrix<T> Bp = B.transpose(perm);
        Matrix<T> B2 = Bp.reshape(shape_t{K, N});

        if (this->t1->requires_grad) {
            Matrix<T> gA = G2.matmul(B2.transpose()).reshape(A.shape);
            this->t1->accumulate(std::move(gA));
        }
        if (this->t2->requires_grad) {
            Matrix<T> gB = A2.transpose().matmul(G2).reshape(Bp.shape).transpose(inv);
            this->t2->accumulate(std::move(gB));
        }
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