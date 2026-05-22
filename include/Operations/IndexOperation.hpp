#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __Index_OPP_INCLUDED__
#define __Index_OPP_INCLUDED__


template <typename T>
class IndexOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        Matrix<bool> mask;
        shape_t orig_shape;

    IndexOperation(Tensor_t<T> t1, Tensor_t<bool> inmask): t1(t1), mask(inmask->val)
    {
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 

    void zero_grad();
    void reset_graph();
    void to_string(){
        std::cout << "Bool Index Operation \n";
    }
};


/**
 * Natural Index function Implementation
*/

    template <typename T>
    Tensor_t<T> IndexOperation<T>::forward()
    {
        this->orig_shape = this->t1->val.shape;
        return std::make_shared<Tensor<T>>(
            this->t1->val.at(mask), this->shared_from_this());
    }

    template <typename T>
    void IndexOperation<T>::backward(Matrix<T> grad)
    {
        Matrix<T> t1_grad = Matrix<T>::zeros(this->orig_shape);
        size_t total = this->mask.get_size();
        size_t gi = 0;
        for (size_t i = 0; i < total; ++i)
            if (this->mask.data[i])
                t1_grad.data[i] = grad.data[gi++];
        this->t1->backward(t1_grad);
    }

    template <typename T>
    void IndexOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void IndexOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }


#endif