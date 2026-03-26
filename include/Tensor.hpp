#ifndef __TENSOR_CLASS_INCLUDED__
#define __TENSOR_CLASS_INCLUDED__

#include "types.hpp"
#include "header.hpp"

template <typename T>
class Tensor : public std::enable_shared_from_this<Tensor<T>>
{
    public:
        std::vector<T> val; //Tensor Value
        std::vector<T> grad; //Tensor gradian
        Operation_t<T> frontOp = nullptr, backOp = nullptr;

    //....................................................................................................
    Tensor() //ov
    {
        this->val = 0;
    }

    Tensor(std::vector<T> val) // ov
    {
        this->val = val;
    }

    Tensor(std::vector<T> val, Operation_t<T> op)
    {
        this->val = val;
        this->backOp = op;
    }


    Tensor(const Tensor_t<T> two) //Const Copy Constructor
    {
        this->val = two->val;
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
    }


        
    // Functions...........................................................................
    Tensor_t<T> exp()
    {
        this->frontOp = std::make_shared<ExponentOperation<T>>((this->shared_from_this()));
        return this->frontOp->forward(); 
    }

};

template <typename T>
Tensor_t<T> make_tensor()
{
    return std::make_shared<Tensor<T>>();        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> val)
{
    return std::make_shared<Tensor<T>>(val);        
}

template <typename T>
Tensor_t<T> make_tensor(Tensor<T> ten)
{
    return std::make_shared<Tensor<T>>(ten);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> val, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(val, op);        
}

template <typename T>
Tensor_t<T> make_tensor(Tensor_t<T> two)
{
    return std::make_shared<Tensor_t<T>>(two);        
}



// Overloads to get actual tensors during operations............................................................
template <typename T>
Tensor_t<T> operator *(Tensor_t<T> left, Tensor_t<T> right)
{
    left->frontOp = std::make_shared<MultiplyOperation<T>>(left, right);
    right->frontOp = left->frontOp;
    return left->frontOp->forward(); 
}
    
template <typename T>
Tensor_t<T> operator +(Tensor_t<T> left, Tensor_t<T> right)
{
    left->frontOp = std::make_shared<AddOperation<T>>(left, right);
    right->frontOp = left->frontOp;
    return left->frontOp->forward(); 
}

template <typename T>
Tensor_t<T> operator -(Tensor_t<T> left, Tensor_t<T> right)
{
    left->frontOp = std::make_shared<SubtractOperation<T>>(left, right);
    right->frontOp = left->frontOp;
    return left->frontOp->forward(); 
}

template <typename T>
Tensor_t<T> operator /(Tensor_t<T> left, Tensor_t<T> right)
{
    left->frontOp = std::make_shared<DivisionOperation<T>>(left, right);
    right->frontOp = left->frontOp;
    return left->frontOp->forward(); 
}


//Scalar Operations..................................................................
template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator *(Tensor_t<S> left, S a)
{
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res->val =  a * res->val;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator *(E a, Tensor_t<E> right)
{
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res->val =  a * res->val;
    return res; 
}

#endif