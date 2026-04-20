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
#include "../Operations/ReluOperation.hpp"
#include "../Operations/DotOperation.hpp"
#include "../Operations/MatmulOperation.hpp"
#include "../Operations/SigmoidOperation.hpp"

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

    Tensor(Matrix<T> *data) // ov
    {
        this->data.copy_from(data);
    }


    Tensor(const Matrix<T> &data) // ov
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

    void backward()
    { // x = x - f`(x)*x

        this->grad.copy_from(this->data);

        if(this->backOp != nullptr)
        { 
            this->backOp->backward(this->data); 
            this->backOp = nullptr;
            this->frontOp = nullptr;
        }
    }

    void backward(Tensor_t<T> grad)
    { // x = x - f`(x)*x

        this->grad->data.copy_from(grad->data);

        if(this->backOp != nullptr)
        { 
            this->backOp->backward(grad->data); 
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

   Tensor<T>& operator=(const Tensor<T>& rhs)
    {
        this->data.copy_from(rhs.data);
        this->grad.copy_from(rhs.grad);
        this->backOp = rhs.backOp;
        this->frontOp = rhs.frontOp;
        return *this;
    }


        
    // Functions In graph...........................................................................
    Tensor_t<T> exp()
    {
        this->frontOp = std::make_shared<ExponentOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> relu()
    {
        this->frontOp = std::make_shared<ReluOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    Tensor_t<T> matmul(Tensor_t<T> x)
    {
        this->frontOp = std::make_shared<MatmulOperation<T>>((this->shared_from_this()), x);
        return this->frontOp->forward(); 
    }

    Tensor_t<T> dot(Tensor_t<T> x)
    {
        this->frontOp = std::make_shared<DotOperation<T>>((this->shared_from_this()), x);
        return this->frontOp->forward(); 
    }

    Tensor_t<T> sigmoid()
    {
        this->frontOp = std::make_shared<SigmoidOperation<T>>(this->shared_from_this());
        return this->frontOp->forward();
    }
    
    // Functions Off graph...........................................................................

    Tensor_t<T> softmax()
    {
        return this->exp() / (this->exp())->sum();
    }

    Tensor_t<T> mse(Tensor_t<T> yt)
    {
        T n = static_cast<T>(yt->data.shape[0]);
        return (((yt - this->shared_from_this()) ^ (T)2)->sum()) / n;
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
     *  friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);
     */

};


#endif