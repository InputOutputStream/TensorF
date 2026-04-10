#include "../Types/types.hpp"

#ifndef __OPERATION_INCLUDED__
#define __OPERATION_INCLUDED__


template <typename T>
class Tensor;

template <typename T>
class Matrix;

template <typename T>

class Operation : public std::enable_shared_from_this<Operation<T>>// Abstract Operation class
{
    public:
        Tensor_t<T> t1, t2;

    
    virtual void backward(Matrix<T> grad) = 0;
    virtual void to_string() = 0;
    virtual void zero_grad() = 0;
    virtual void reset_graph() = 0;
    virtual Tensor_t<T> forward() = 0;
      
};

#endif