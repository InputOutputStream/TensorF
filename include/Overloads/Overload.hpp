
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
    for(size_t i=0; i<b.size(); i++)
    { 
        if (b.at(i) == T(0))   
            throw std::runtime_error("Division by zero");
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(a.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
    for(size_t i=0; i<b.size(); i++)
    { 
        if constexpr (std::is_integral_v<T>)
            arr.push_back(b[i] % a);
        else
            arr.push_back(std::fmod(b[i], a));
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const T a, const std::vector<T> &b) // Scalar mod l
{
    std::vector<T> arr;
    arr.reserve(b.size());
    for(size_t i=0; i<b.size(); i++)
    { 
        if constexpr (std::is_integral_v<T>)
            arr.push_back(b[i] % a);
        else
            arr.push_back(std::fmod(b[i], a));
    }

    return arr;
} 

template <typename T>
std::vector<T> operator > (const T a, const std::vector<T> &b) // Scalar  l
{
    std::vector<T> arr;
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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
    arr.reserve(b.size());
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


template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
        throw std::runtime_error("Size mismatch in +=");
    for(size_t i = 0; i < a.size(); i++)
        a[i] += b[i];
    return a;
}

template <typename T>
std::vector<T>& operator -=(std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
        throw std::runtime_error("Size mismatch in -=");
    for(size_t i = 0; i < a.size(); i++)
        a[i] -= b[i];
    return a;
}

template <typename T>
std::vector<T>& operator *=(std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
        throw std::runtime_error("Size mismatch in *=");
    for(size_t i = 0; i < a.size(); i++)
        a[i] *= b[i];
    return a;
}

template <typename T>
std::vector<T>& operator /=(std::vector<T>& a, const std::vector<T>& b)
{
    if(a.size() != b.size())
        throw std::runtime_error("Size mismatch in /=");
    for(size_t i = 0; i < a.size(); i++)
        a[i] /= b[i];
    return a;
}

template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const T b)
{
    for(size_t i = 0; i < a.size(); i++) a[i] += b;
    return a;
}

template <typename T>
std::vector<T>& operator-=(std::vector<T>& a, const T b)
{
    for(size_t i = 0; i < a.size(); i++) a[i] -= b;
    return a;
}

template <typename T>
std::vector<T>& operator*=(std::vector<T>& a, const T b)
{
    for(size_t i = 0; i < a.size(); i++) a[i] *= b;
    return a;
}

template <typename T>
std::vector<T>& operator/=(std::vector<T>& a, const T b)
{
    if(b == T(0)) throw std::runtime_error("Division by zero in /=");
    for(size_t i = 0; i < a.size(); i++) a[i] /= b;
    return a;
}
/**
 * mathematical functions......................................................................................
 */

template <typename T>
std::vector<T> exponent(const std::vector<T> &a) // Exponential of a std::vector
{
    std::vector<T> arr;
    arr.reserve(a.size());
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
    arr.reserve(a.size());
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
    if(b->size() == 1)
        return a^b[0];
    else if(b.size() == a.size())
    {
        arr.reserve(a.size());
        for(size_t i = 0; i < a.size(); i++)
        arr.push_back((T)std::pow(a[i], b[i]));
   }
   else{
        throw std::runtime_error("Invalid vector sizes for power op\n");
   }
    return arr;
} 

/**
 * ........................................................................................
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