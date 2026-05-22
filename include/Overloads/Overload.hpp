
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

    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<a.size(); i++)
    {
        arr[i] = (T)(a[i] * b[i]);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator +(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Addition
{
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i< a.size(); i++)
    {
        arr[i] = (T)(a[i] + b[i]);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator -(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Subtraction
{
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i< a.size(); i++)
    {
        arr[i] = (T)(a[i] - b[i]);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator /(const std::vector<T> &a, const std::vector<T> &b) //std::Vector Division
{
    if(a.size() != b.size())
        throw std::runtime_error("Tensors are not of the same size!!!\n");

    std::vector<T> arr(a.size(), (T)0);
    for(size_t i=0; i<a.size(); i++)
    { 
        if (b[i] == T(0))
            throw std::runtime_error("Division by zero in vector division\n");
        arr[i] = (T)(a[i] / b[i]);
    }

    return arr;
} 


/**
 * scalar Operations..........................................................................................................
 */
 
template <typename T>
std::vector<T> operator * (const T a, const std::vector<T> &b) // Scalar Product l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a * b[i]);
    }

    return arr;
}

template <typename T>
std::vector<T> operator * (const std::vector<T> &b, const T a) // scalar product r
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a * b[i]);
    }

    return arr;
} 



template <typename T>
std::vector<T> operator / (const T a, const std::vector<T> &b) // Scalar Division l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        if (b[i] == T(0))
            throw std::runtime_error("Division by zero");
        arr[i] = (T)(a / b[i]);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator/(const std::vector<T> &b, const T a)
{
    if (a == T(0))
        throw std::runtime_error("Division by zero in vector/scalar division\n");

    std::vector<T> arr(b.size(), (T)0);
    for (size_t i = 0; i < b.size(); i++)
        arr[i] = (T)(b[i] / a);
    return arr;
}


template <typename T>
std::vector<T> operator + (const T a, const std::vector<T> &b) // Scalar Division l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a + b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator + (const std::vector<T> &b, const T a) // scalar  r
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i]+a);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator -(const std::vector<T> &a) // scalar  r
{
    std::vector<T> arr(a.size(), (T)0);
    for(size_t i=0; i<a.size(); i++)
    { 
        arr[i] = (T)(a[i]*-1);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator - (const T a, const std::vector<T> &b) // Scalar  l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a - b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator - (const std::vector<T> &b, const T a) // scalar  r
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i]-a);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const std::vector<T> &b, const T a) // scalar mod r
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        if constexpr (std::is_integral_v<T>)
            arr[i] = (b[i] % a);
        else
            arr[i] =(std::fmod(b[i], a));
    }

    return arr;
} 

template <typename T>
std::vector<T> operator % (const T a, const std::vector<T> &b) // Scalar mod l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        if constexpr (std::is_integral_v<T>)
            arr[i]=(a % b[i]);
        else
            arr[i]=(std::fmod(a, b[i]));
    }

    return arr;
} 


template <typename T>
std::vector<T> operator > (const T a, const std::vector<T> &b) // Scalar  l
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a > b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator > (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i] > a);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator < (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a < b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator < (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i] < a);
    }

    return arr;
} 


template <typename T>
std::vector<T> operator <= (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a <= b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator <= (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i] <= a);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator >= (const T a, const std::vector<T> &b) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(a >= b[i]);
    }

    return arr;
} 

template <typename T>
std::vector<T> operator >= (const std::vector<T> &b, const T a) 
{
    std::vector<T> arr(b.size(), (T)0);
    for(size_t i=0; i<b.size(); i++)
    { 
        arr[i] = (T)(b[i] >= a);
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


template <typename T, typename U>
requires (std::is_arithmetic_v<U>)
std::vector<uint8_t> operator==(const std::vector<T>& a, const U& b) {
    std::vector<uint8_t> arr(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++)
        arr[i]=(a[i] == static_cast<T>(b));
    return arr;
}

template <typename T, typename U>
requires (std::is_arithmetic_v<U>)
std::vector<uint8_t> operator==(const U& b, const std::vector<T>& a) {
    return a == b;
}

template <typename T, typename U>
requires (std::is_arithmetic_v<U>)
std::vector<uint8_t> operator!=(const std::vector<T>& a, const U& b) {
    std::vector<uint8_t> arr(a.size(), 0);
    for (size_t i = 0; i < a.size(); i++)
        arr[i]=(a[i] != static_cast<T>(b));
    return arr;
}

template <typename T, typename U>
requires (std::is_arithmetic_v<U>)
std::vector<uint8_t> operator!=(const U& b, const std::vector<T>& a) {
    return a != b;
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
    {
        if (b[i] == T(0))
            throw std::runtime_error("Division by zero in /=");
        a[i] /= b[i];
    }
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
    std::vector<T> arr(a.size(), 0);
    for(size_t i=0; i< a.size(); i++)
    { 
        arr[i] = (T)exp(a[i]);
    }
    return arr;
} 

template <typename T>
std::vector<T> pow(const std::vector<T> &a, const T n) // Power of a std::vector
{
    std::vector<T> arr(a.size(), 0);
    if (n == T(2))
    {    // a[i] * a[i] instead of pow(a[i], 2)
        for(size_t i=0; i<a.size(); i++)
        { 
            arr[i] = a[i] * a[i];
        }
        return arr;
    }    

    if (n == T(3))
    {    // a[i] * a[i] instead of pow(a[i], 3)
        for(size_t i=0; i<a.size(); i++)
        { 
            arr[i] = a[i] * a[i] * a[i];

        }
        return arr;
    }    

    if (n == T(4))
    {    
        for(size_t i=0; i<a.size(); i++)
        { 
            arr[i] = a[i] * a[i] * a[i]* a[i];

        }
        return arr;
    }
    
    for(size_t i=0; i<a.size(); i++)
    { 
        arr[i] = (T)std::pow(a[i], n);
    }

    return arr;
} 

template <typename T>
std::vector<T> pow(const std::vector<T> &a, const std::vector<T> &b) // Power of a std::vector
{
    std::vector<T> arr(a.size(), 0);
    if(b.size() == 1)
        return pow(a,b[0]);
    else if(b.size() == a.size())
    {
        for(size_t i = 0; i < a.size(); i++)
        arr[i]=(T)std::pow(a[i], b[i]);
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


template<typename T>
void check_nan(const std::vector<T>& v)
{
    for(size_t i=0;i<v.size();i++)
    {
        if(std::isnan(v[i]) || std::isinf(v[i]))
        {
            std::cerr << "Invalid value at " << i << "\n";
            std::abort();
        }
    }
}

#endif