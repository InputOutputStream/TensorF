#ifndef __NEURAL_LAYER__
#define __NEURAL_LAYER__

#include "../Types/types.hpp"
#include "Module.hpp"

    #include <vector>

    template <typename T>
    class NeuralLayer: protected Module<T>{

        private:
    
        void getMods(const std::initializer_list<Module<T>> inmods)
        {
            if (inmods.size() == 0) return;

            for (const auto& item : inmods)
            {   
                this->register_module(item);
            }
        }
        public:

        NeuralLayer(std::initializer_list<Module<T>> inmods)
        {
            getMods(inmods);
        }

        Tensor_t<T> forward(Tensor_t<T> x){
            Tensor_t<T> out = x;
            for(Module<T> mod: this->submodules)
            {
                out = mod.forward(out);
            }

            return out;
        }

    };

#endif