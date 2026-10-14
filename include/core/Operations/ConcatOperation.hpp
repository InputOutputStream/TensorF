#include "Types/types.hpp"
#include "Operation.hpp"


#ifndef __CONCAT_OPP_INCLUDED__
#define __CONCAT_OPP_INCLUDED__


template <typename T>
class ConcatOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> input_elms;
    size_t axis;
    std::vector<size_t> split_sizes;  // size of each input_elms along axis, for backward

    ConcatOperation(std::vector<Tensor_t<T>> input_elms, size_t axis)
        : input_elms(input_elms), axis(axis) {}
   
    Tensor_t<T> forward();
    void backward(Matrix<T> grad);

    void zero_grad() override { for (auto& t : input_elms) t->zero_grad(); }
    void reset_graph() override { 
        for (auto& t : input_elms){
            if(t)
                t->reset_graph(); 
        }
    }
    void to_string() override { std::cout << "Concat Operation\n"; }
    
    std::vector<Tensor_t<T>> inputs() const override {
        return {this->input_elms};
    }

};

template <typename T>
Tensor_t<T> ConcatOperation<T>::forward() 
    {
        std::vector<Matrix<T>> mats;
        for (auto& t : input_elms) {
            mats.push_back(t->val);
            split_sizes.push_back(t->val.shape[axis]);
        }
        return std::make_shared<Tensor<T>>(
            Matrix<T>::concat(mats, axis), this->shared_from_this());
    }

template <typename T>
void ConcatOperation<T>::backward(Matrix<T> grad) 
    {
        // split grad along axis at the recorded boundaries
        size_t offset = 0;
        for (size_t i = 0; i < input_elms.size(); ++i) {
            // slice grad from offset to offset+split_sizes[i] along axis
            Matrix<T> g = grad.slice_axis(offset, offset + split_sizes[i], axis);
            input_elms[i]->accumulate(g);
            offset += split_sizes[i];
        }
    }



#endif