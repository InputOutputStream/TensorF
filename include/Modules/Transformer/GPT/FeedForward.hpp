#ifndef __FEED_FORWARD_H
#define __FEED_FORWARD_H


#include "../../../Types/types.hpp"
#include "../../Linear.hpp"
#include "../../Optimizer.hpp"
#include "../../Module.hpp"

template <typename T>
class FeedForward: public Module<T>{
            
    private:
        Linear<T> up;
        Linear<T> down;

    public:

    FeedForward(size_t in_features, size_t hidden, size_t out_features)
    : up(in_features, hidden, true),
      down(hidden, out_features, true)
        {
            this->register_module(&up);
            this->register_module(&down);
        }

    FeedForward(FeedForward&& other)
        : Module<T>(),          // fresh, empty submodules list
        up(std::move(other.up)),
        down(std::move(other.down))
    {
        // re-register at NEW addresses
        this->register_module(&up);
        this->register_module(&down);
    }

    FeedForward(const FeedForward&) = delete; 

    Tensor_t<T> forward(Tensor_t<T> x) {   
        Tensor_t<T> a = up.forward(x);
        Tensor_t<T> b = a->relu();
        auto k = down.forward(b);
        // std::cerr << " feef forward out val : "<< k->val<< "\n";
        return k;
    }

    friend class GGUFLoader<T>;
    friend class GPTGGUFLoader<T>; 

};


#endif // !__FEED_FORWARD_H