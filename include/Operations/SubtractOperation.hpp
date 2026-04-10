#include "../Types/types.hpp"
#include "Operation.hpp"

#ifndef __SUBTRACT_OPP_INCLUDED__
#define __SUBTRACT_OPP_INCLUDED__


template <typename T>
class SubtractOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    SubtractOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(Matrix<T> grad);

    Tensor_t<T> forward();
    void zero_grad();

    void reset_graph();

    void to_string(){
        std::cout << "Subtract Operation \n";
    }
      
};

#endif