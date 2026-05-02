#ifndef __FEED_FORWARD_H
#define __FEED_FORWARD_H

#include "../Types/types.hpp"
#include "Linear.hpp"
#include "Optimizer.hpp"
#include "Module.hpp"

template <typename T>
class FeedForward: public Module<T>{
    public:
        Tensor_t<T> ypred;
        
    private:
        Linear<T> l1;
        Linear<T> l2;
        Linear<T> l3;

    public:

    FeedForward(size_t in_features, size_t hidden, size_t out_features)
    : l1(in_features, hidden, true),
      l2(hidden, hidden, true),
      l3(hidden, out_features, true)
        {
            this->register_module(&l1);
            this->register_module(&l2);
            this->register_module(&l3);
        }

        Tensor_t<T> forward(Tensor_t<T> x) {   
            Tensor_t<T> a = l1.forward(x);
            Tensor_t<T> b = a->relu();

            Tensor_t<T> c = l2.forward(b);
            Tensor_t<T> d = c->relu();

            Tensor_t<T> e = l3.forward(d);

            this->ypred = e->softmax();
            return this->ypred;
        }
};


#endif // !__FEED_FORWARD_H