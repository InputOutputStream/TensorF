#ifndef TENSOR_EXTERN_HPP
#define TENSOR_EXTERN_HPP

#include <iostream>
#include <vector>
#include <memory>
#include "Overload.hpp"

#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"

#include "../Operations/Operation.hpp"
#include "../Operations/AddOperation.hpp"
#include "../Operations/MultiplyOperation.hpp"
#include "../Operations/PowerOperation.hpp"
#include "../Operations/DivisionOperation.hpp"
#include "../Operations/ExponentOperation.hpp"
#include "../Operations/SubtractOperation.hpp"
#include "../Operations/ReluOperation.hpp"
#include "../Operations/DotOperation.hpp"
#include "../Operations/MatmulOperation.hpp"
#include "../Operations/SigmoidOperation.hpp"

template <typename T>
class Tensor;

// empty
template <typename T>
Tensor_t<T> make_tensor()
{
    return std::make_shared<Tensor<T>>();        
}

// from constant
template <typename T>
Tensor_t<T> make_tensor(const T a)
{
    return std::make_shared<Tensor<T>>(Matrix<T>({(T)a}));        
}

template <typename T>
Tensor_t<T> make_tensor(const T a, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>({a}), op);        
}

// from vector
template <typename T>
Tensor_t<T> make_tensor(std::vector<T> data)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(data));        
}


template <typename T>
Tensor_t<T> make_tensor(std::vector<T> data, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(data), op);        
}

// from other tensor
template <typename T>
Tensor_t<T> make_tensor(Tensor_t<T> two)
{
    return std::make_shared<Tensor<T>>(two);        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata, std::initializer_list<long> inshape)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape));        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata, std::initializer_list<long> inshape, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape), op);        
}

template <typename T>
Tensor_t<T> make_tensor(Matrix<T>* two)
{
    return std::make_shared<Tensor<T>>(two);        
}

template <typename T>
Tensor_t<T> make_tensor(Matrix<T>* two, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(two, op);        
}

template <typename T>
Tensor_t<T> make_tensor(Matrix<T>& two)
{
    return std::make_shared<Tensor<T>>(two);        

}

template <typename T>
Tensor_t<T> make_tensor(const Matrix<T>& two)
{
    return std::make_shared<Tensor<T>>(two);        

}

template <typename T>
Tensor_t<T> make_tensor(Matrix<T>& two, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(two, op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, shape_t inshape)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape));        

}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, shape_t inshape, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape), op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<std::vector<T>> indata)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata));        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<std::vector<T>> indata, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata), op);        

}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata));        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata), op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, std::initializer_list<long> inshape)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape));        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, std::initializer_list<long> inshape, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape), op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<std::initializer_list<T>> indata)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata));        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<std::initializer_list<T>> indata, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata), op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> indata)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata));        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> indata, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata), op);        
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

template <typename T>
Tensor_t<T> operator ^(Tensor_t<T> left, Tensor_t<T> right)
{
    left->frontOp = std::make_shared<PowerOperation<T>>(left, right);
    right->frontOp = left->frontOp;
    return left->frontOp->forward(); 
}

template <typename E>
Tensor_t<E> operator ^(Tensor_t<E> right, const E a)
{
    Tensor_t<E> cte = make_tensor<E>(a);
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res =  res^cte;
    return res; 
}

//Scalar Operations..................................................................
template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator +(Tensor_t<S> left, const S a)
{
    Tensor_t<S> cte = make_tensor<S>(a);
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res =  res + cte;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator +(const E a, Tensor_t<E> right)
{    
    Tensor_t<E> cte = make_tensor<E>(a);
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res =  cte + res;
    return res; 
}

template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator -(Tensor_t<S> left, const S a)
{
    Tensor_t<S> cte = make_tensor<S>(a);
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res =  res - cte;
    return res; 
}


template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator -(const S a, Tensor_t<S> left)
{
    Tensor_t<S> cte = make_tensor<S>(a);
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res =  cte - res;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator *(const E a, Tensor_t<E> right)
{    
    Tensor_t<E> cte = make_tensor<E>(a);
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res =  cte * res;
    return res; 
}

template <typename S>
//requires std::is_arithmetic_v<S>
Tensor_t<S> operator /(Tensor_t<S> left, const S a)
{
    Tensor_t<S> cte = make_tensor<S>(a);
    Tensor_t<S> res = std::make_shared<Tensor<S>>(left);
    res =  res/cte;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator /(const E a, Tensor_t<E> right)
{
    Tensor_t<E> cte = make_tensor<E>(a);
    Tensor_t<E> res = std::make_shared<Tensor<E>>(right);
    res = cte / res;
    return res; 
}


template <typename E>
//requires std::is_arithmetic_v<S>
Tensor_t<E> operator -(Tensor_t<E> ten)
{
    Tensor_t<E> n = make_tensor<E>(-1);
    Tensor_t<E> res = std::make_shared<Tensor<E>>(ten);
    res =  res * n;
    return res; 
}

template <typename E>
Tensor_t<E> operator <(const E a, Tensor_t<E> right)
{    
    Matrix<E> cte(a);
    Matrix<E> res(right);
    res =  cte < res;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator <(Tensor_t<E> right, const E a)
{    
    Matrix<E> cte(a);
    Matrix<E> res(right);
    res =  res < cte;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator >(const E a, Tensor_t<E> right)
{    
    Matrix<E> cte(a);
    Matrix<E> res(right);
    res =  cte < res;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator >(Tensor_t<E> right, const E a)
{    
    Matrix<E> cte(a);
    Matrix<E> res(right);
    res =  res > cte;
    return make_tensor<E>(res); 
}

#endif // !TENSOR_EXTERN__HPP