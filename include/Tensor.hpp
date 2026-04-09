#ifndef __TENSOR_CLASS_INCLUDED__
#define __TENSOR_CLASS_INCLUDED__

#include "types.hpp"
#include "header.hpp"
#include "tensor_extern.hpp"

//#include "Matrix.hpp"
template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        std::vector<T> data; //Tensor value
        std::vector<T> grad; //Tensor gradian
        Operation_t<T> frontOp = nullptr, backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->data = 0;
    }

    Tensor(std::vector<T> data) // ov
    {
        this->data = data;
    }

    Tensor(std::vector<T> data, Operation_t<T> op)
    {
        this->data = data;
        this->backOp = op;
    }


    Tensor(const Tensor_t<T> two) //Const Copy Constructor
    {
        this->data = two->data;
        this->backOp = two->backOp;
        this->frontOp = two->frontOp;
        this->grad = two->grad;
    }

    void backward(std::vector<T> grad)
    { // x = x - f`(x)*x

        this->grad = grad;

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