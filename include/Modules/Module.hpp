#ifndef __MODULE__
#define __MODULE__

#include "../Types/types.hpp"
#include "../DataLoader/Serializer.hpp"

    #include <iostream>
    #include <vector>

template <typename T>
    class Parameter {
        public:
            Tensor_t<T> tensor;   
            bool requires_grad;  

            Parameter(Tensor_t<T> t) : tensor(t) {}
    };

    template <typename T>
    class Module {
        protected:
            std::vector<Tensor_t<T>> params;      
            std::vector<Module<T>*> submodules;   
            Serializer<T> Ser;

        public:
            void register_parameter(Tensor_t<T> p)
            {
                params.push_back(p);
            }

            void register_module(Module<T>* m)
            {
                submodules.push_back(m);
            }

            std::vector<Tensor_t<T>> parameters()
            {
                std::vector<Tensor_t<T>> all = params;
                for(auto m : this->submodules)
                {
                    auto child_params = m->parameters();
                    all.insert(all.end(), child_params.begin(), child_params.end());
                }
                return all;
            }

            void zero_grad()
            {
                for(auto p : params)
                    p->zero_grad();
                for(auto m : submodules)
                    m->zero_grad();
            }

            void save(std::string path) {
                Ser.save(this->parameters(), path);  // recursive collection
            }

            void load(std::string path) {
                auto loaded = Ser.load(path);
                auto all = this->parameters();  // get all params recursively
                
                if(loaded.size() != all.size())
                    throw std::runtime_error("Loaded tensor count mismatch: expected " 
                        + std::to_string(all.size()) + " got " + std::to_string(loaded.size()));
                
                for(size_t i = 0; i < all.size(); i++)
                    all[i]->val.copy_from(loaded[i]->val);
            }

            void to_string();
            void reset_graph();
            Tensor_t<T> forward();
    };

#endif

