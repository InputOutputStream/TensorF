#include "types.hpp"
#include "Operation.hpp"

#ifndef __ADD_OPP_INCLUDED__
#define __ADD_OPP_INCLUDED__


template <typename T>
class AddOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1, t2;

    //..............................................................................................................

    AddOperation(Tensor_t<T> t1, Tensor_t<T> t2)
    {
        this->t1 = t1;
        this->t2 = t2;
    }

    void backward(std::vector<T> grad);

    Tensor_t<T> forward();

    void to_string(){
        std::cout << "Add Operation \n";
    }
      
};




#endif