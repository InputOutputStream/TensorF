#ifndef __FEED_FORWARD_H
#define __FEED_FORWARD_H


#include "../../../Types/types.hpp"
#include "../../Optimizer.hpp"
#include "../../Module.hpp"

template <typename T, template<typename> class LinearT>
class FeedForward: public Module<T>{
            
    private:
        LinearT<T> up;
        LinearT<T> down;

    public:

    template <typename... Args>
    FeedForward(size_t in_features, size_t hidden, size_t out_features, Args&&... args)
    : up(hidden, in_features, args...),
      down(out_features, hidden, args...)
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
        Tensor_t<T> b = Tensor<T>::GeLU(a);
        auto k = down.forward(b);
        // std::cerr << " feef forward out val : "<< k->val<< "\n";
        return k;
    }

    friend class GGUFLoader<T>;
    friend class GPTGGUFLoader<T>; 

};


#endif // !__FEED_FORWARD_H