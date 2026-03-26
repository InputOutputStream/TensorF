#include "types.hpp"
#include "header.hpp"

#ifndef __OPERATION_IMPL_INCLUDED__
#define __OPERATION_IMPL_INCLUDED__


/**
 * Multiply Operation Implementation
*/

    template <typename T>
    void MultiplyOperation<T>::backward(std::vector<T> grad)
    {
        // Switching Gradients when carrying out product
        this->t1->backward(grad * this->t2->val); 
        this->t2->backward(grad * this->t1->val);
    }

    template <typename T>
    Tensor_t<T> MultiplyOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val*this->t2->val, this->shared_from_this());
    }


/**
 * Add Operation Implementation
*/

    template <typename T>
    void AddOperation<T>::backward(std::vector<T> grad)
    {
        // Distributing Gradients when carrying out addition
        this->t1->backward(grad);
        this->t2->backward(grad);
    }

    template<typename T>
    Tensor_t<T> AddOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val+this->t2->val, this->shared_from_this());
    }

/**
 * Subtract Operation Implementation
*/

    template <typename T>
    void SubtractOperation<T>::backward(std::vector<T> grad)
    {
        // Distributing Gradients when carrying out subtraction
        this->t1->backward(grad);
        this->t2->backward(grad);
    }

    template<typename T>
    Tensor_t<T> SubtractOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val-this->t2->val, this->shared_from_this());
    }


/**
 * Divide Operation Implementation
*/
 // forward is :x/y
 // backward will be the derivative of both wrt to x and wrt y
 // wrt x: grad/y
 // wrt y: (grad *x * (-1)) / y^2
    
 
 /**
        std::cout << "this->t1->val = " << this->t1->val << " this->t2->val " << this->t2->val<<"\n";
        std::cout << "grad = " << grad <<"\n";
        std::cout << "temp ::  (this->t2->val) = " << temp << " :: " << this->t2->val<<"\n";
        std::cout << "temp / this->t2->val = " <<temp / this->t2->val<<"\n";
        std::cout << "temp / (this->t2->val)^(T)2 = " <<(temp /((this->t2->val)^(T)2))<<"\n";
*/

    template <typename T>
    void DivisionOperation<T>::backward(std::vector<T> grad)
    {
        auto temp = ((T)-1) * grad * this->t1->val ;
        this->t1->backward(grad / this->t2->val);  // ok
        this->t2->backward( temp / ((this->t2->val)^(T)2) );
    }        


    template <typename T>
    Tensor_t<T> DivisionOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(this->t1->val/this->t2->val, this->shared_from_this());
    }




    void sleep(int n){
        int stime = 1000000000;
        for(int i = 0; i<(stime * n); i++)
        {

        }
}


/**
 * Exponential function Implementation
*/

    template <typename T>
    void ExponentOperation<T>::backward(std::vector<T> grad)
    {
        this->t1->backward(grad * exponent(this->t1->val)); 
    }

    template <typename T>
    Tensor_t<T> ExponentOperation<T>::forward()
    {
        return std::make_shared<Tensor<T>>(exponent(this->t1->val), this->shared_from_this());
    }

/**
 * Tensor function definitions
*/

#endif