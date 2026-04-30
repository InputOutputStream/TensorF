#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __TRANSPOSE_OPP_INCLUDED__
#define __TRANSPOSE_OPP_INCLUDED__


template <typename T>
class TransposeOperation : public Operation<T>
{
    protected: 
        Tensor_t<T> tmp;
        shape_t inperm;

    public:
        Tensor_t<T> t1;
        
    TransposeOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    Tensor_t<T> forward(std::initializer_list<size_t> inperm);
    Tensor_t<T> forward(shape_t inperm);
   
    void zero_grad();


    void reset_graph();
    void to_string(){
        std::cout << "Transpose Operation \n";
    }
};


/**
 * Transpose Implementation
*/

    template <typename T>
    void TransposeOperation<T>::backward(Matrix<T> grad)
    {
        if(this->inperm.size() > 0)
            this->t1->backward(grad.transpose(this->inperm)); 
        else 
            this->t1->backward(grad.transpose()); 
    }

    template <typename T>
    Tensor_t<T> TransposeOperation<T>::forward()
    {
        this->tmp = std::make_shared<Tensor<T>>(this->t1->val.transpose());
        return std::make_shared<Tensor<T>>(this->tmp->val, this->shared_from_this());
    }

    template <typename T>
    Tensor_t<T> TransposeOperation<T>::forward(std::initializer_list<size_t> inperm)
    {
        this->inperm = Matrix<T>::getShape(inperm);
        this->tmp = std::make_shared<Tensor<T>>(this->t1->val.transpose(this->inperm));
        return std::make_shared<Tensor<T>>(this->tmp->val, this->shared_from_this());
    }

    template <typename T>
    Tensor_t<T> TransposeOperation<T>::forward(shape_t inperm)
    {
        this->inperm = inperm;
        this->tmp = std::make_shared<Tensor<T>>(this->t1->val.transpose(inperm));
        return std::make_shared<Tensor<T>>(this->tmp->val, this->shared_from_this());
    }

    template <typename T>
    void TransposeOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->tmp->zero_grad();
    }

    template <typename T>
    void TransposeOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->tmp->reset_graph(); 
    }


#endif