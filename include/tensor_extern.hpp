#ifndef TENSOR_EXTERN_HPP
#define TENSOR_EXTERN_HPP

#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <cassert>
#include <memory>

#include "Operation.hpp"
#include "AddOperation.hpp"
#include "MultiplyOperation.hpp"
#include "DivisionOperation.hpp"
#include "ExponentOperation.hpp"
#include "SubtractOperation.hpp"
#include "Overload.hpp"
#include "operation_impl.hpp"


template <typename T>
class Tensor;

template <typename T>
Tensor_t<T> make_tensor()
{
    return std::make_shared<Tensor<T>>();        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> data)
{
    return std::make_shared<Tensor<T>>(data);        
}

template <typename T>
Tensor_t<T> make_tensor(Tensor<T> ten)
{
    return std::make_shared<Tensor<T>>(ten);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> data, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(data, op);        
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
Tensor_t<S> operator *(Tensor_t<S> left, const S a)
{
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res->data =  a * res->data;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator *(const E a, Tensor_t<E> right)
{
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res->data =  a * res->data;
    return res; 
}

template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator /(Tensor_t<S> left, const S a)
{
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res->data =  a / res->data;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator /(const E a, Tensor_t<E> right)
{
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res->data =  a / res->data;
    return res; 
}



#endif // !TENSOR_EXTERN__HPP