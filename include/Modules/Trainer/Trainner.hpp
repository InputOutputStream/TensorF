#ifndef __TRAINER__HPP_
#define __TRAINER__HPP_

#include "../../Types/types.hpp"
#include "../../DataStructures/Matrix.hpp"

enum TrainMode { FINETUNE, DISTILL, FEDAVG, FEDMETA };


template<typename T>
class Trainer {
    Model<T>& student;
    Model<T>& teacher;
    Optimizer<T> op;
    T temperature;
    T alpha;

public:
    Trainer(Model<T>& student, Model<T>& teacher,
            T lr, T temperature, T alpha)
        : student(student), teacher(teacher),
          op(student.params, lr, SGD),
          temperature(temperature), alpha(alpha)
    {}

    // expose logits separately from softmax output — needed for temperature scaling
    Tensor_t<T> forward_logits(Tensor_t<T> x)
    {
        auto a = l1.forward(x)->relu();
        auto b = l2.forward(a)->relu();
        return l3.forward(b);          // raw logits, no softmax
    }

    // existing forward just calls forward_logits then softmax
    Tensor_t<T> forward(Tensor_t<T> x)
    {
        this->ypred = forward_logits(x)->softmax();
        return this->ypred;
    }

    // freeze all parameters — used on teacher
    void freeze()
    {
        for (auto& p : params)
            p->requires_grad = false;  // you'll need this flag in Tensor
    }

    void distill(Tensor_t<T> X, Tensor_t<T> y, int iters);
    void finetune(Tensor_t<T> X, Tensor_t<T> y, int iters);
    void evaluate(Tensor_t<T> X, Tensor_t<T> y);
};

#endif