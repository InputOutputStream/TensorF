#ifndef __FEED_FORWARD_H
#define __FEED_FORWARD_H

#include "../Types/types.hpp"
#include "Linear.hpp"
#include "Optimizer.hpp"
#include "Module.hpp"

template <typename T>
class FeedForward: public Module<T>{
            
    private:
        Linear<T> l1;
        Linear<T> l2;

    public:

    FeedForward(size_t in_features, size_t hidden, size_t out_features)
    : l1(in_features, hidden, true),
      l2(hidden, out_features, true)
        {
            this->register_module(&l1);
            this->register_module(&l2);
        }

    FeedForward(FeedForward&& other)
        : Module<T>(),          // fresh, empty submodules list
        l1(std::move(other.l1)),
        l2(std::move(other.l2))
    {
        // re-register at NEW addresses
        this->register_module(&l1);
        this->register_module(&l2);
    }

    FeedForward(const FeedForward&) = delete; 

    Tensor_t<T> forward(Tensor_t<T> x) {   
        Tensor_t<T> a = l1.forward(x);
        Tensor_t<T> b = a->relu();

        return l2.forward(b);
    }
};


#endif // !__FEED_FORWARD_H