#ifndef __TRAINER__HPP_
#define __TRAINER__HPP_

#include "../../Types/types.hpp"
#include "../../DataStructures/Matrix.hpp"
#include "../../Modules/Transformer/GPT.hpp"

#include <functional>
#include <iostream>

enum TrainMode { FINETUNE, DISTILL, FEDAVG, FEDMETA };

// ---------------------------------------------------------------------------
// Trainer<T>
//
// Wraps a student GPT model and an optional teacher GPT model.
// Supports:
//   finetune()  — standard supervised fine-tuning on labelled data
//   distill()   — knowledge distillation (teacher soft labels + hard targets)
//   evaluate()  — validation-set loss reporting
//
// ---------------------------------------------------------------------------

template<typename T>
class Trainer {
    GPT<T>& student;
    GPT<T>* teacher;   // nullable — only needed for distillation
    Optimizer<T> op;

    T temperature;   // temperature for soft-label generation
    T alpha;         // distillation blend:  alpha*hard + (1-alpha)*soft

public:
    // teacher is optional (pass nullptr when only fine-tuning)
    Trainer(GPT<T>& student,
            GPT<T>* teacher,
            T lr,
            T temperature = T(3.0),
            T alpha       = T(0.5))
        : student(student),
          teacher(teacher),
          op(student.parameters(), lr, ADAMw),
          temperature(temperature),
          alpha(alpha)
    {}

    // ── Internal helpers ─────────────────────────────────────────────────────

    // Run student forward, return raw logits (no softmax).
    //      Delegate to GPT::forward with targets=nullptr → returns logit tensor.
    Tensor_t<T> forward_logits(Tensor_t<T> inputs) {
        return student.forward(inputs, nullptr, /*apply_mask=*/true);
    }

    // Convenience wrapper that adds softmax on top.
    Tensor_t<T> forward(Tensor_t<T> inputs) {
        return forward_logits(inputs)->softmax();
    }

    // Freeze all student parameters (used to lock teacher during distillation).
    void freeze(GPT<T>& m) {
        for (auto& p : m.parameters())
            p->requires_grad = false;
    }
    void unfreeze(GPT<T>& m) {
        for (auto& p : m.parameters())
            p->requires_grad = true;
    }

    // ── Distillation loss ────────────────────────────────────────────────────
    //
    //  L = alpha * CE(hard_targets, student_probs)
    //    + (1-alpha) * T^2 * CE(teacher_soft, student_soft)
    //
    Tensor_t<T> distill_loss(Tensor_t<T> student_logits,
                              Tensor_t<T> teacher_logits,
                              Tensor_t<T> one_hot_targets)
    {
        // Hard loss — standard cross-entropy vs ground-truth labels
        auto student_probs = student_logits->softmax();
        auto hard_loss = Tensor<T>::cross_entropy(one_hot_targets, student_probs);

        // Soft loss — KL(teacher_soft ‖ student_soft) approximated by cross-entropy
        auto T_ten         = make_tensor<T>(temperature);
        auto teacher_soft  = (teacher_logits / T_ten)->softmax();
        auto student_soft  = (student_logits / T_ten)->softmax();
        // Scale by T² so gradients have the same magnitude regardless of temperature
        auto soft_loss = Tensor<T>::cross_entropy(teacher_soft, student_soft)
                         * make_tensor<T>(temperature * temperature);

        auto a     = make_tensor<T>(alpha);
        auto one_a = make_tensor<T>(T(1.0) - alpha);
        return a * hard_loss + one_a * soft_loss;
    }

    // ── Public training API ──────────────────────────────────────────────────

    // get_batch_fn("train") → {inputs, targets}  where both are {B, seq_len}
    using BatchFn = std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)>;

    void finetune(Tensor_t<T> X, Tensor_t<T> y, int iters) {
    for (int i = 0; i < iters; i++) {
        auto logits = student.forward(X);          // forward pass
        auto loss   = cross_entropy(logits, y);    // loss
        op.zero_grad();
        loss->backward({T(1)});                    // backward
        op.step();                                 // AdamW step
        
        if (i % 10 == 0)
            std::cout << "step " << i << " loss: " << loss->val.data[0] << "\n";
    }
}

    // Knowledge distillation: student learns from teacher soft labels + hard targets.
    void distill(BatchFn get_batch, int iters) {
        if (!teacher)
            throw std::runtime_error("Trainer::distill: no teacher model provided");

        // Freeze teacher so its weights don't move during distillation
        freeze(*teacher);

        for (int i = 0; i < iters; ++i) {
            op.zero_grad();
            auto [inputs, targets] = get_batch("train");

            size_t B   = inputs->shape[0];
            size_t S   = inputs->shape[1];
            size_t V   = student.get_vocab_size();  

            // Raw logits from both models — no softmax yet
            auto student_logits = student.forward(inputs, nullptr, true)
                                         ->reshape({B * S, V});
            auto teacher_logits = teacher->forward(inputs, nullptr, true)
                                         ->reshape({B * S, V});

            // One-hot encode hard targets for the CE term
            auto targets_flat   = targets->reshape({B * S});
            auto one_hot        = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, V));

            auto loss = distill_loss(student_logits, teacher_logits, one_hot);
            loss->backward(make_tensor<T>(T(1.0)));
            op.step();
            loss->reset_graph();

            if (i % 100 == 0)
                std::cout << "[distill]  iter " << i << "  loss: " << loss->val << "\n";
        }

        unfreeze(*teacher);  // restore in case caller reuses teacher elsewhere
    }

    // Compute and print validation loss without updating weights.
    void evaluate(BatchFn get_batch) {
        auto [inputs, targets] = get_batch("val");
        auto val_loss = student.forward(inputs, targets, /*apply_mask=*/true);
        std::cout << "[evaluate] val loss: " << val_loss->val << "\n";
        val_loss->reset_graph();
    }
};

#endif