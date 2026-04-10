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
        }

        // this->reset_graph();
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
        if(this->backOp != nullptr)
        { 

            this->backOp->reset_graph(); 
            this->backOp = nullptr;
        }

        if(this->frontOp!= nullptr)
        { 
            this->frontOp->reset_graph(); 
            this->frontOp = nullptr;
        }
    }


        
    // Functions...........................................................................
    Tensor_t<T> exp()
    {
        this->frontOp = std::make_shared<ExponentOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

    /**
     *  create a friend of a static
     *    
     *  template <typename E>
        friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);

     */

};


#endif