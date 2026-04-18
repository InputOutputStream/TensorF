#ifndef __TENSOR_CLASS_INCLUDED__
#define __TENSOR_CLASS_INCLUDED__

#include <memory>
#include <vector>
#include "Matrix.hpp"

#include "../Operations/AddOperation.hpp"
#include "../Operations/MultiplyOperation.hpp"
#include "../Operations/DivisionOperation.hpp"
#include "../Operations/ExponentOperation.hpp"
#include "../Operations/SubtractOperation.hpp"

#include "../Types/types.hpp"
#include "../Overloads/tensor_overloads.hpp"
#include "../Overloads/Overload.hpp"


template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        Matrix<T> data; //Tensor value
        Matrix<T> grad; //Tensor gradian
        Operation_t<T> frontOp = nullptr, backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->data = 0;
    }

    Tensor(Matrix<T> data) // ov
    {
        this->data.copy_from(data);
    }


    Tensor(Matrix<T> *data) // ov
    {
        this->data.copy_from(data);
    }

    Tensor(Matrix<T> data, Operation_t<T> op)
    {
        this->data.copy_from(data);
        this->backOp = op;
    }

    Tensor(const Tensor_t<T> two) //Const Copy Constructor
    {
        this->data.copy_from(two->data);
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad.copy_from(two->grad);
    }


//.....................................................................................
    void backward(Matrix<T> grad)
    { // x = x - f`(x)*x

        this->grad.copy_from(grad);

        if(this->backOp != nullptr)
        { 
            this->backOp->backward(grad); 
            this->backOp = nullptr;
            this->frontOp = nullptr;
        }
    }

    void zero_grad()
    {
        this->grad.clear();

        if(this->backOp != nullptr)
        { 
            this->backOp->zero_grad(); 
        }
    }

    void reset_graph()
    {
        if(this->frontOp!= nullptr)
        { 
            this->frontOp->reset_graph(); 
            this->frontOp = nullptr;
        }
    }

    // Overloads..........................................................................

    Tensor_t<T> operator =(const Tensor_t<T> &rhs)
    {
        return make_tensor<T>(rhs);
    }


        
    // Functions...........................................................................
    Tensor_t<T> exp()
    {
        this->frontOp = std::make_shared<ExponentOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> relu(){
        return make_tensor<T>(this->data.maximum(0));
    }

    Tensor_t<T> derivative_RelU()
    {
        return make_tensor<T>(this->data > 0);
    }

    Tensor_t<T> sigmoid(Tensor_t<T> x)
    {
        return 1 / (1 + (-1*x)->exp());
    }

    Tensor_t<T> softmax()
    {
        return this->exp() / (this->exp())->sum();
    }

    Tensor_t<T> mse(Tensor_t<T> yp, Tensor_t<T> yt)
    {
        return (yt^2 - yp^2)^(1/2);
    }

    Tensor_t<T> matmul(Tensor_t<T> x)
    {
        return make_tensor<T>(this->data.matmul(x));
    }


    Tensor_t<T> dot(Tensor_t<T> x)
    {
        return make_tensor<T>(this->data.dot(x));
    }

    Tensor_t<T> sum(size_t axis)
    {
        return make_tensor<T>(this->data.sum(axis));
    }

     Tensor_t<T> sum()
    {
        return make_tensor<T>(this->data.sum());
    }

    Tensor_t<T> transpose(std::initializer_list<long> inshape)
    {
        return make_tensor<T>(this->data.transpose(inshape));
    }

    Tensor_t<T> transpose(shape_t inshape)
    {
        return make_tensor<T>(this->data.transpose(inshape));
    }

     Tensor_t<T> transpose()
    {
        return make_tensor<T>(this->data.transpose());
    }

    /**
     *  create a friend of a static
     *    
     *  template <typename E>
        friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);

     */

};


#endif