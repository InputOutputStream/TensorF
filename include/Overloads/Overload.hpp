
#ifndef __OVERLOAD_INCLUDED__
#define __OVERLOAD_INCLUDED__

#include "../Types/types.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <stdexcept>

/**
 * Arithmetic Operations
 */

template <typename T>
std::vector<T> operator *(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Multiplication
{
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

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
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

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
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

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
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

    std::vector<T> arr;
    for(size_t i=0; i<a.size(); i++)
    { 
        if (b.at(i) == T(0))
            throw std::runtime_error("Division by zero in vector division\n");
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
    if (std::find(b.begin(), b.end(), (double)0) != b.end())
        throw std::runtime_error("Division by zero in vector division");
    
    for(size_t i=0; i<b.size(); i++)
    { 
        if (b.at(i) == T(0))
            throw std::runtime_error("Division by zero in vector division\n");
        T prod = (T)(a / b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator/(const std::vector<T> &b, const T a)
{
    if (a == T(0))
        throw std::runtime_error("Division by zero in vector/scalar division\n");

    std::vector<T> arr;
    arr.reserve(b.size());
    for (size_t i = 0; i < b.size(); i++)
        arr.push_back(b.at(i) / a);
    return arr;
}


template <typename T>
std::vector<T> operator + (const T a, const std::vector<T> &b) // Scalar Division l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a + b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator + (const std::vector<T> &b, const T a) // scalar  r
{
    std::vector<T> arr;

    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i)+a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator -(const std::vector<T> &a) // scalar  r
{
    std::vector<T> arr;
    for(size_t i=0; i<a.size(); i++)
    { 
        T prod = (T)(a.at(i)*-1);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator - (const T a, const std::vector<T> &b) // Scalar  l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a - b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator - (const std::vector<T> &b, const T a) // scalar  r
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i)-a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const std::vector<T> &b, const T a) // scalar mod r
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
std::vector<T> operator % (const T a, const std::vector<T> &b) // Scalar mod l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a % b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator > (const T a, const std::vector<T> &b) // Scalar  l
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a > b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator > (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i) > a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator < (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a < b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator < (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i) < a);
        arr.push_back(prod);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator <= (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a <= b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator <= (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i) <= a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator >= (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(a >= b.at(i));
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator >= (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr;
    for(size_t i=0; i<b.size(); i++)
    { 
        T prod = (T)(b.at(i) >= a);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
bool operator ==(const std::vector<T> &a, const std::vector<T> &b)
{
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (a[i] != b[i]) return false;
        
    return true;
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
    if (n == T(2))
    {    // a[i] * a[i] instead of pow(a[i], 2)
        for(size_t i=0; i<a.size(); i++)
        { 
            T prod = a.at(i) * a.at(i);
            arr.push_back(prod);
        }
        return arr;
    }    

    if (n == T(3))
    {    // a[i] * a[i] instead of pow(a[i], 3)
        for(size_t i=0; i<a.size(); i++)
        { 
            T prod = a.at(i) * a.at(i) * a.at(i);
            arr.push_back(prod);
        }
        return arr;
    }    

    if (n == T(4))
    {    
        for(size_t i=0; i<a.size(); i++)
        { 
            T prod = a.at(i) * a.at(i) * a.at(i)* a.at(i);
            arr.push_back(prod);
        }
        return arr;
    }
    
    for(size_t i=0; i<a.size(); i++)
    { 
        T prod = (T)std::pow(a.at(i), n);
        arr.push_back(prod);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator ^(const std::vector<T> &a, const std::vector<T> &b) // Power of a std::vector
{
    std::vector<T> arr;
    T cte = b[0];
    for(size_t i=0; i<a.size(); i++)
    { 
        T prod = (T)pow(a.at(i), cte);
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
    if (std::is_same<T, size_t>::value) {
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