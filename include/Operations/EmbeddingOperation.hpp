#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __Embedding_OPP_INCLUDED__
#define __Embedding_OPP_INCLUDED__


template <typename T>
class EmbeddingOperation : public Operation<T>
{

    public:
        Tensor_t<T> t1;
        Tensor_t<T> idx_saved; 

    EmbeddingOperation(Tensor_t<T> t1, Tensor_t<T> indices)
    {
        this->t1 = t1;
        idx_saved = indices;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    void zero_grad();
    void reset_graph();
    void to_string(){
        std::cout << "Embedding Operation \n";
    }
};


/**
 * Natural Embedding function Implementation
*/

    template <typename T>
    void EmbeddingOperation<T>::backward(Matrix<T> grad)
    {
        // grad has shape {len(idx), embed_dim}
        // scatter each grad row back to the corresponding row in t1

        Matrix<T> t1_grad = Matrix<T>::zeros(this->t1->val.shape);
        size_t embed_dim = this->t1->val.shape[1];

        for (size_t i = 0; i < this->idx_saved->val.get_size(); ++i)
        {
            size_t row = (size_t)this->idx_saved->val.data[i];
            for (size_t j = 0; j < embed_dim; ++j)
                t1_grad.data[row * embed_dim + j] += grad.data[i * embed_dim + j];
        }
        
        this->t1->backward(t1_grad);
    }

    template <typename T>
    Tensor_t<T> EmbeddingOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val.elemsAt(idx_saved->val), this->shared_from_this());
    }

    template <typename T>
    void EmbeddingOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->idx_saved->zero_grad();
    }

    template <typename T>
    void EmbeddingOperation<T>::reset_graph(){
        if (this->t1) {
            this->t1->reset_graph();
            this->t1 = nullptr; // Drop strong reference
        }
        if (this->idx_saved) {
            this->idx_saved->reset_graph();
            this->idx_saved = nullptr; // Drop strong reference
        }
    }
#endif