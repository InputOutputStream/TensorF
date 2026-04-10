
#ifndef __OPERATION_IMPL_INCLUDED__
#define __OPERATION_IMPL_INCLUDED__

#include "../Types/types.hpp"
#include <memory>
#include <vector>
#include <iostream>

#include "../Operations/Operation.hpp"
#include "../Operations/AddOperation.hpp"
#include "../Operations/MultiplyOperation.hpp"
#include "../Operations/DivisionOperation.hpp"
#include "../Operations/ExponentOperation.hpp"
#include "../Operations/SubtractOperation.hpp"

/**
 * Multiply Operation Implementation
*/

    template <typename T>
    void MultiplyOperation<T>::backward(Matrix<T> grad)
    {
        // Switching Gradients when carrying out product
        this->t1->backward(grad * this->t2->data); 
        this->t2->backward(grad * this->t1->data);
    }

    template <typename T>
    Tensor_t<T> MultiplyOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data*this->t2->data, this->shared_from_this());
    }

    template <typename T>
    void MultiplyOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void MultiplyOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }



/**
 * Add Operation Implementation
*/

    template <typename T>
    void AddOperation<T>::backward(Matrix<T> grad)
    {
        // Distributing Gradients when carrying out addition
        this->t1->backward(grad);
        this->t2->backward(grad);
    }

    template<typename T>
    Tensor_t<T> AddOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data+this->t2->data, this->shared_from_this());
    }

    template <typename T>
    void AddOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void AddOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }


/**
 * Subtract Operation Implementation
*/

    template <typename T>
    void SubtractOperation<T>::backward(Matrix<T> grad)
    {
        // Distributing Gradients when carrying out subtraction
        this->t1->backward(grad);
        this->t2->backward(-grad);
    }

    
    template <typename T>
    void SubtractOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void SubtractOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }

    template<typename T>
    Tensor_t<T> SubtractOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data-this->t2->data, this->shared_from_this());
    }


/**
 * Divide Operation Implementation
*/
 // forward is :x/y
 // backward will be the derivative of both wrt to x and wrt y
 // wrt x: grad/y
 // wrt y: (grad *x * (-1)) / y^2

    template <typename T>
    void DivisionOperation<T>::backward(Matrix<T> grad)
    {
        auto temp = ((T)-1) * grad * this->t1->data ;
        this->t1->backward(grad / this->t2->data);  // ok
        this->t2->backward( temp / ((this->t2->data) * (this->t2->data)));
    }        

    template <typename T>
    void DivisionOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
        this->t2->zero_grad();
    }

    template <typename T>
    void DivisionOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
        this->t2->reset_graph();
    }

    template <typename T>
    Tensor_t<T> DivisionOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>((this->t1->data/this->t2->data), this->shared_from_this());
    }


/**
 * Exponential function Implementation
*/

    template <typename T>
    void ExponentOperation<T>::backward(Matrix<T> grad)
    {
        this->t1->backward(grad * this->t1->data.exponent()); 
    }

    template <typename T>
    Tensor_t<T> ExponentOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->data.exponent(), this->shared_from_this());
    }

    template <typename T>
    void ExponentOperation<T>::zero_grad(){
        this->t1->zero_grad(); 
    }

    template <typename T>
    void ExponentOperation<T>::reset_graph(){
        this->t1->reset_graph(); 
    }

/**
 * Tensor function definitions
*/

#endif