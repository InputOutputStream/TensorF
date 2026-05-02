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
#include "../Operations/SumOperation.hpp"
#include "../Operations/SumAxisOperation.hpp"
#include "../Operations/SoftmaxOperation.hpp"
 
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
    return std::make_shared<Tensor<T>>(Matrix<T>({a}));        
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
Tensor_t<T> make_tensor(std::initializer_list<T> indata, std::initializer_list<size_t> inshape)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape));        
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata, std::initializer_list<size_t> inshape, Operation_t<T> op)
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
    std::vector<T> e;
    
    Matrix<T>::flattenReccursive(indata, e);
    return std::make_shared<Tensor<T>>(Matrix<T>(e));
}

template <typename T>
Tensor_t<T> make_tensor(std::initializer_list<T> indata, Operation_t<T> op)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata), op);        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, std::initializer_list<size_t> inshape)
{
    return std::make_shared<Tensor<T>>(Matrix<T>(indata, inshape));        
}

template <typename T>
Tensor_t<T> make_tensor(std::vector<T> indata, std::initializer_list<size_t> inshape, Operation_t<T> op)
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

// In tensor_overloads.hpp, remove the right->frontOp assignment:
template <typename T>
Tensor_t<T> operator +(Tensor_t<T> left, Tensor_t<T> right)
{
    auto op = std::make_shared<AddOperation<T>>(left, right);
    left->frontOp = op;
    return op->forward(); 
}

template <typename T>
Tensor_t<T> operator *(Tensor_t<T> left, Tensor_t<T> right)
{
    auto op = std::make_shared<MultiplyOperation<T>>(left, right);
    left->frontOp = op;
    return op->forward(); 
}

  
template <typename T>
Tensor_t<T> operator -(Tensor_t<T> left, Tensor_t<T> right)
{
    auto op = std::make_shared<SubtractOperation<T>>(left, right);
    left->frontOp = op;
    return op->forward(); 
}


template <typename T>
Tensor_t<T> operator /(Tensor_t<T> left, Tensor_t<T> right)
{
    auto op = std::make_shared<DivisionOperation<T>>(left, right);
    left->frontOp = op;
    return op->forward(); 
}


template <typename T>
Tensor_t<T> operator ^(Tensor_t<T> left, Tensor_t<T> right)
{
    auto op = std::make_shared<PowerOperation<T>>(left, right);
    left->frontOp = op;
    return op->forward(); 
}




template <typename T>
bool operator ==(Tensor_t<T> left, Tensor_t<T> right)
{
    if (!left || !right) return left == right; 
    return left->val == right->val;
}


template <typename T>
Tensor_t<T> operator +=(Tensor_t<T> left, Tensor_t<T> right)
{
    left->val += right->val;
    return left;
}

template <typename T>
Tensor_t<T> operator -=(Tensor_t<T> left, Tensor_t<T> right)
{
    left->val -= right->val;
    return left;
}

template <typename T>
Tensor_t<T> operator *=(Tensor_t<T> left, Tensor_t<T> right)
{
    left->val *= right->val;
    return left;
}

template <typename T>
Tensor_t<T> operator /=(Tensor_t<T> left, Tensor_t<T> right)
{
    left->val /= right->val;
    return left;
}


template <typename T>
Tensor_t<T> operator +=(Tensor_t<T> left, const T cte)
{
    left->val += cte;
    return left;
}

template <typename T>
Tensor_t<T> operator -=(Tensor_t<T> left, const T cte)
{
    left->val -= cte;
    return left;
}

template <typename T>
Tensor_t<T> operator *=(Tensor_t<T> left, const T cte)
{
    left->val *= cte;
    return left;
}

template <typename T>
Tensor_t<T> operator /=(Tensor_t<T> left, const T cte)
{
    left->val /= cte;    
    return left;
}

//Scalar Operations..................................................................
template <typename E>
Tensor_t<E> operator ^(Tensor_t<E> left, const E a) {
    auto cte = make_tensor<E>(a);
    auto op = std::make_shared<PowerOperation<E>>(left, cte);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename S>
Tensor_t<S> operator +(Tensor_t<S> left, const S a)
{
    auto cte = make_tensor<S>(a);
    auto op = std::make_shared<AddOperation<S>>(left, cte);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename E>
Tensor_t<E> operator +(const E a, Tensor_t<E> right)
{    
    auto cte = make_tensor<E>(a);
    auto op = std::make_shared<AddOperation<E>>(cte, right);
    right->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}

template <typename S>
Tensor_t<S> operator -(Tensor_t<S> left, const S a)
{
    auto cte = make_tensor<S>(a);
    auto op = std::make_shared<SubtractOperation<S>>(left, cte);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename S>
Tensor_t<S> operator -(const S a, Tensor_t<S> left)
{
    auto cte = make_tensor<S>(a);
    auto op = std::make_shared<SubtractOperation<S>>(cte, left);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename E>
Tensor_t<E> operator *(Tensor_t<E> left, const E a)
{    
    auto cte = make_tensor<E>(a);
    auto op = std::make_shared<MultiplyOperation<E>>(left, cte);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}

template <typename E>
Tensor_t<E> operator *(const E a, Tensor_t<E> right)
{    
    auto cte = make_tensor<E>(a);
    auto op = std::make_shared<MultiplyOperation<E>>(cte, right);
    right->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename S>
Tensor_t<S> operator /(Tensor_t<S> left, const S a)
{
    auto cte = make_tensor<S>(a);
    auto op = std::make_shared<DivisionOperation<S>>(left, cte);
    left->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename E>
Tensor_t<E> operator /(const E a, Tensor_t<E> right)
{
    auto cte = make_tensor<E>(a);
    auto op = std::make_shared<DivisionOperation<E>>(cte, right);
    right->frontOp = op;
    cte->frontOp = op;
    return op->forward();
}


template <typename E>
Tensor_t<E> operator -(Tensor_t<E> ten)
{
    Tensor_t<E> n = make_tensor<E>(-1);
    auto op = std::make_shared<MultiplyOperation<E>>(ten, n);
    ten->frontOp = op;
    n->frontOp = op;
    return op->forward();
}

// Non graph related ops........................................................
template <typename E>
Tensor_t<E> operator <(const E a, Tensor_t<E> right)
{    
    Matrix<E> res(right->val);
    res =  a < res;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator <(Tensor_t<E> right, const E a)
{    
    Matrix<E> res(right->val);
    res =  res < a;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator >(const E a, Tensor_t<E> right)
{    
    Matrix<E> res(right->val);
    res =  a > res;
    return make_tensor<E>(res); 
}

template <typename E>
Tensor_t<E> operator >(Tensor_t<E> right, const E a)
{    
    Matrix<E> res(right->val);
    res =  res > a;
    return make_tensor<E>(res); 
}


#endif // !TENSOR_EXTERN__HPP