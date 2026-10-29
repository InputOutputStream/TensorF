#include "Types/types.hpp"
#include "Operation.hpp"

#ifndef __MATMUL_OPP_INCLUDED__
#define __MATMUL_OPP_INCLUDED__


template <typename T>
class MatmulOperation : public Operation<T>
{
    public:
    std::vector<Tensor_t<T>> inputs() const override { return {this->t1, this->t2}; }
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    MatmulOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad) override;

    // Gradients of C = A.matmul(B) for any rank (1D promotion, batch broadcasting, last-two-axes transpose).

     static Matrix<T> matmul_grad_a(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G);
    // Gradient w.r.t. B only: dB = Aᵀ @ G
    static Matrix<T> matmul_grad_b(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G);

    static std::pair<Matrix<T>, Matrix<T>> matmul_grads(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G) {
        return { matmul_grad_a(A, B, G), matmul_grad_b(A, B, G) };
    }

    Tensor_t<T> forward() override;

    void zero_grad() override;
    void reset_graph() override;

    void to_string() override {
        std::cout << "Matmul Operation \n";
    }
      
};



/**
 * Matmul Operation Implementation
*/

    template <typename T>
    Matrix<T> MatmulOperation<T>::matmul_grad_a(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G)
    {
        // promote 1-D inputs and restore the (possibly squeezed) output rank
        const Matrix<T>* A2 = &A;
        Matrix<T> A2_buf;
        if (A.shape.size() == 1) { A2_buf = A.reshape({1, A.shape[0]}); A2 = &A2_buf; }

        const Matrix<T>* B2 = &B;
        Matrix<T> B2_buf;
        if (B.shape.size() == 1) { B2_buf = B.reshape({B.shape[0], 1}); B2 = &B2_buf; }

        const Matrix<T>* G2 = &G;
        Matrix<T> G2_buf;
        if (A.shape.size() == 1 || B.shape.size() == 1) {
            shape_t gs = G.shape;
            if (A.shape.size() == 1) gs.insert(gs.end() - (B.shape.size() == 1 ? 0 : 1), 1);
            if (B.shape.size() == 1) gs.push_back(1);
            G2_buf = G.reshape(gs);
            G2 = &G2_buf;
        }

        Matrix<T> gA = sumGradForBroadcast(G2->matmul(mxd::swap_last2(*B2)), A2->shape);
        return gA.reshape(A.shape);
    }


    template <typename T>
    Matrix<T> MatmulOperation<T>::matmul_grad_b(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& G)
    {
        const Matrix<T>* A2 = &A;
        Matrix<T> A2_buf;
        if (A.shape.size() == 1) { A2_buf = A.reshape({1, A.shape[0]}); A2 = &A2_buf; }

        const Matrix<T>* B2 = &B;
        Matrix<T> B2_buf;
        if (B.shape.size() == 1) { B2_buf = B.reshape({B.shape[0], 1}); B2 = &B2_buf; }

        const Matrix<T>* G2 = &G;
        Matrix<T> G2_buf;
        if (A.shape.size() == 1 || B.shape.size() == 1) {
            shape_t gs = G.shape;
            if (A.shape.size() == 1) gs.insert(gs.end() - (B.shape.size() == 1 ? 0 : 1), 1);
            if (B.shape.size() == 1) gs.push_back(1);
            G2_buf = G.reshape(gs);
            G2 = &G2_buf;
        }

        Matrix<T> gB = sumGradForBroadcast(mxd::swap_last2(*A2).matmul(*G2), B2->shape);
        return gB.reshape(B.shape);
    }

    template <typename T>
    void MatmulOperation<T>::backward(Matrix<T> grad)
    {
        if (!this->t1->requires_grad && !this->t2->requires_grad) return;
        if (this->t1->requires_grad)
            this->t1->accumulate(matmul_grad_a(this->t1->val, this->t2->val, grad));
        if (this->t2->requires_grad)
            this->t2->accumulate(matmul_grad_b(this->t1->val, this->t2->val, grad));
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