#ifndef __FEED_FORWARD_H
#define __FEED_FORWARD_H

#include "Types/types.hpp"
#include "Linear.hpp"
#include "Optimizer.hpp"
#include "Module.hpp"

template <typename T, template<typename> class LinearT>
class FeedForward: public Module<T>{
            
    private:
        Linear<T> up;
        Linear<T> down;

    public:

    FeedForward(size_t in_features, size_t hidden, size_t out_features)
    : up(hidden, in_features, true),
      down(out_features, hidden, true)
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

    LinearT<T>& get_up()   { return up; }
    LinearT<T>& get_down() { return down; }

    void load_pretrained(FeedForward<T, Linear>& src) {
        up.load_pretrained(src.get_up());
        down.load_pretrained(src.get_down());
    }

    Tensor_t<T> forward(Tensor_t<T> x) {   
        Tensor_t<T> a = up.forward(x);
        Tensor_t<T> b = a->relu();
        auto k = down.forward(b);
        // std::cerr << " feef forward out val : "<< k->val<< "\n";
        return k;
    }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;
};


#endif // !__FEED_FORWARD_H