#include "../Types/types.hpp"
#include "Operation.hpp"


#ifndef __EXP_OPP_INCLUDED__
#define __EXP_OPP_INCLUDED__


template <typename T>
class ExponentOperation : public Operation<T>
{
    public:
        Tensor_t<T> t1;
        
    ExponentOperation(Tensor_t<T> t1)
    {
        this->t1 = t1;
    }  

    void backward(Matrix<T> grad);

    Tensor_t<T> forward(); 
    void zero_grad();


    void reset_graph();
    void to_string(){
        std::cout << "Exp Operation \n";
    }
};

#endif