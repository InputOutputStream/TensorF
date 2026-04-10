
#ifndef __OVERLOAD_INCLUDED__
#define __OVERLOAD_INCLUDED__

#include "../Types/types.hpp"
#include <vector>
#include <iostream>
#include <cassert>

/**
 * Arithmetic Operations
 */

template <typename T>
std::vector<T> operator *(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Multiplication
{
    assert("Tensors are not of the same size!!!" && a.size() == b.size());
    std::vector<T> arr;
    for(size_t i=0; i<a.size(); i++)
    {
        T prod = (T)(a.at(i) * b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator +(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Addition
{
    assert("Tensors are not of the same size!!!" && a.size() == b.size());
    std::vector<T> arr;
    for(size_t i=0; i< a.size(); i++)
    {
        T prod = (T)(a.at(i) + b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator -(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Subtraction
{
    assert("Tensors are not of the same size!!!" && a.size() == b.size());
    std::vector<T> arr;
    for(size_t i=0; i< a.size(); i++)
    {
        T prod = (T)(a.at(i) - b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator /(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Division
{
    assert("Tensors are not of the same size!!!" && a.size() == b.size());
    std::vector<T> arr;
    for(size_t i=0; i<a.size(); i++)
    { 
        T quot = (T)(a.at(i) / b.at(i));
        arr.push_back(quot);
    }

    return arr;
} 


/**
 * scalar Operations..........................................................................................................
 */
 
template <typename T>
std::vector<T> operator * (const T a, const std::vector<T> &b) // Scalar Product l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a * b.at(i));
        arr.push_back(prod);
    }

    return arr;
}

template <typename T>
std::vector<T> operator * (const std::vector<T> &b, const T a) // scalar product r
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a * b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 



template <typename T>
std::vector<T> operator / (const T a, const std::vector<T> &b) // Scalar Division l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a / b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator / (const std::vector<T> &b, const T a) // scalar Division r
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i)/a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const std::vector<T> &b, const T a) // scalar Division r
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i)%a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const T a, const std::vector<T> &b) // Scalar Division l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a % b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

/**
 * mathematical functions......................................................................................
 */

template <typename T>
std::vector<T> exponent(const std::vector<T> &a) // Exponential of a std::vector
{
    std::vector<T> arr;
    for(size_t i=0; i< a.size(); i++)
    { 
        T prod = (T)exp(a.at(i));
        arr.push_back(prod);
    }
    return arr;
} 

template <typename T>
std::vector<T> operator ^(const std::vector<T> &a, const T n) // Power of a std::vector
{
    std::vector<T> arr;
    for(size_t i=0; i<a.size(); i++)
    { 
        T prod = (T)pow(a.at(i), n);
        arr.push_back(prod);
    }

    return arr;
} 

/**
 * Matrix Overloads..........................................................................................
*/

template <typename T>
std::ostream& operator << (std::ostream &out , const std::vector<T> &a) // Print
{
    char C[] = "[]";
    if (std::is_same<T, long>::value) {
        C[0] = '(';
        C[1] = ')';
        
    } else {
        C[0] = '[';
        C[1] = ']';
    }

    out<<C[0];
    for(auto i : a)
    {
        out<<i<<",";
    }
    out<<C[1];
    
    return out;
} 




#endif