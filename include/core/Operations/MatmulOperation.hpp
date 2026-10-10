#include "Types/types.hpp"
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

    // Gradients of C = A.matmul(B) for any rank (1D promotion, batch broadcasting, last-two-axes transpose).
    static std::pair<Matrix<T>, Matrix<T>> matmul_grads(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G)
    {
        Matrix<T> A2 = A, B2 = B;
        if (A.shape.size() == 1) A2 = A.reshape({1, A.shape[0]});
        if (B.shape.size() == 1) B2 = B.reshape({B.shape[0], 1});
        // restore the (possibly squeezed) output axes to rank-2+ form
        shape_t gs = G.shape;
        if (A.shape.size() == 1) gs.insert(gs.end() - (B.shape.size() == 1 ? 0 : 1), 1);
        if (B.shape.size() == 1) gs.push_back(1);
        Matrix<T> G2 = G.reshape(gs);
        Matrix<T> gA = sumGradForBroadcast(G2.matmul(mxd::swap_last2(B2)), A2.shape);
        Matrix<T> gB = sumGradForBroadcast(mxd::swap_last2(A2).matmul(G2), B2.shape);
        return { gA.reshape(A.shape), gB.reshape(B.shape) };
    }

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
        auto g = matmul_grads(this->t1->val, this->t2->val, grad);
        this->t1->backward(g.first);
        this->t2->backward(g.second);
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