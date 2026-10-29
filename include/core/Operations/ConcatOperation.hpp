#include "Types/types.hpp"
#include "Operation.hpp"


#ifndef __CONCAT_OPP_INCLUDED__
#define __CONCAT_OPP_INCLUDED__


template <typename T>
class ConcatOperation : public Operation<T>
{
public:
    std::vector<Tensor_t<T>> inputs() const override { return in_tensors; }
    std::vector<Tensor_t<T>> in_tensors;
    size_t axis;
    std::vector<size_t> split_sizes;  // size of each input along axis, for backward

    ConcatOperation(std::vector<Tensor_t<T>> in, size_t axis)
        : in_tensors(in), axis(axis) {}
   
    Tensor_t<T> forward() override;
    void backward(Matrix<T> grad) override;

    void zero_grad() override { for (auto& t : in_tensors) t->zero_grad(); }
    void reset_graph() override { 
        for (auto& t : in_tensors){
            if(t)
                t->reset_graph(); 
        }
    }
    void to_string() override { std::cout << "Concat Operation\n"; }
};

template <typename T>
Tensor_t<T> ConcatOperation<T>::forward() 
    {
        std::vector<Matrix<T>> mats;
        for (auto& t : in_tensors) {
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
        for (size_t i = 0; i < in_tensors.size(); ++i) {
            if (in_tensors[i]->requires_grad) {
                Matrix<T> g = grad.slice_axis(offset, offset + split_sizes[i], axis);
                in_tensors[i]->accumulate(std::move(g));
            }
            offset += split_sizes[i];
        }
    }



#endif