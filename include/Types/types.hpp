#ifndef __TYPES__
#define __TYPES__

    #include <memory>
    #include <vector>
    
    using shape_t = std::vector<long>;

    template <typename T>
    class Tensor;

    template <typename T>
    class Operation;

    template <typename T>
    using Operation_t=std::shared_ptr<Operation<T>>;

    template <typename T>
    using Tensor_t=std::shared_ptr<Tensor<T>>;

#endif // !__TYPES__
