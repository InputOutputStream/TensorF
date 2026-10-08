#ifndef __TENSORF_SPMM_OPP_INCLUDED__
#define __TENSORF_SPMM_OPP_INCLUDED__

// Sparse x dense multiply as an autograd operation:   Y = A * X
//   A : N x N constant CSR matrix (no gradient)      X : N x F tensor
// Backward: dX = A^T * dY.
//   * A symmetric (Graph::build() output): A^T == A, so it is the same row-gather as forward
//     and parallelises over rows with no synchronisation.
//   * A general: scatter form (serial), correct for any CSR.

#include "Types/types.hpp"
#include "Operation.hpp"
#include <algorithm>
#include <thread>
#include <vector>

template <typename T>
class SpMMOperation : public Operation<T>
{
public:
    Tensor_t<T> t1;                              // X
    std::shared_ptr<const CSR<T>> A;
    size_t num_threads;

    SpMMOperation(std::shared_ptr<const CSR<T>> A, Tensor_t<T> x, size_t num_threads = 1)
        : t1(x), A(std::move(A)), num_threads(std::max<size_t>(1, num_threads))
    {
        if (!this->A) throw std::invalid_argument("SpMM: null adjacency");
        if (t1->val.shape.size() != 2 || t1->val.shape[0] != this->A->n)
            throw std::invalid_argument("SpMM: X must be 2-D with N rows (N = adjacency size)");
    }

    // Y[rows r0..r1) = A[rows r0..r1) * X       (row gather)
    static void gather_rows(const CSR<T>& a, const T* X, T* Y, size_t F, size_t r0, size_t r1);

    void run_gather(const T* X, T* Y, size_t F) const;

    Tensor_t<T> forward() override;

    void backward(Matrix<T> grad) override;

    void zero_grad() override { this->t1->zero_grad(); }

    void reset_graph() override {
        if (this->t1) { this->t1->reset_graph(); this->t1 = nullptr; }
        A.reset();
    }

    void to_string() override { std::cout << "SpMM Operation \n"; }
};


    // Y[rows r0..r1) = A[rows r0..r1) * X       (row gather)
    template<typename T>
    void SpMMOperation<T>::gather_rows(const CSR<T>& a, const T* X, T* Y, size_t F, size_t r0, size_t r1) {
        for (size_t i = r0; i < r1; i++) {
            T* y = Y + i * F;
            std::fill(y, y + F, T(0));
            for (size_t k = a.row_ptr[i]; k < a.row_ptr[i + 1]; k++) {
                const T v = a.values[k];
                const T* x = X + size_t(a.col_idx[k]) * F;
                for (size_t f = 0; f < F; f++) y[f] += v * x[f];
            }
        }
    }

    template<typename T>
    void SpMMOperation<T>::run_gather(const T* X, T* Y, size_t F) const {
        const size_t N = A->n;
        const size_t work = A->nnz() * F;
        const size_t nt = (work < (1u << 16)) ? 1 : std::min(num_threads, N);   // threads not worth it for tiny jobs
        if (nt <= 1) { gather_rows(*A, X, Y, F, 0, N); return; }
        std::vector<std::thread> pool;
        const size_t chunk = (N + nt - 1) / nt;
        for (size_t t = 0; t < nt; t++) {
            size_t r0 = t * chunk, r1 = std::min(N, r0 + chunk);
            if (r0 >= r1) break;
            pool.emplace_back([&, r0, r1] { gather_rows(*A, X, Y, F, r0, r1); });
        }
        for (auto& th : pool) th.join();
    }

    template<typename T>
    Tensor_t<T> SpMMOperation<T>::forward()
    {
        const size_t N = A->n, F = t1->val.shape[1];
        std::vector<T> y(N * F);
        run_gather(t1->val.data.data(), y.data(), F);
        return std::make_shared<Tensor<T>>(Matrix<T>(std::move(y), shape_t{N, F}), this->shared_from_this());
    }

    template<typename T>
    void SpMMOperation<T>::backward(Matrix<T> grad)
    {
        const size_t N = A->n, F = grad.shape[1];
        std::vector<T> g(N * F, T(0));
        if (A->symmetric) {
            run_gather(grad.data.data(), g.data(), F);
        } else {
            for (size_t i = 0; i < N; i++) {                         // dX[col,:] += v * dY[i,:]
                const T* dy = grad.data.data() + i * F;
                for (size_t k = A->row_ptr[i]; k < A->row_ptr[i + 1]; k++) {
                    const T v = A->values[k];
                    T* dx = g.data() + size_t(A->col_idx[k]) * F;
                    for (size_t f = 0; f < F; f++) dx[f] += v * dy[f];
                }
            }
        }
        this->t1->backward(Matrix<T>(std::move(g), shape_t{N, F}));
    }
  
#endif