#ifndef __TRAINER__HPP_
#define __TRAINER__HPP_

#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataLoader/Quantizer.hpp"

#include <functional>
#include <iostream>
#include <string>
#include <type_traits>

enum TrainMode { FINETUNE, DISTILL, FEDAVG, FEDMETA, FEDDISTILL };

// ── Selective-training detection ────────────────────────────────────────────
//
// If Model exposes `lora_parameters()` (e.g. GPTLoRA<T> — see GPTLoRA.hpp),
// the Trainer's optimizer is built from THAT instead of the full
// `parameters()`. This is what makes "client owns a small LoRA adapter on a
// frozen backbone" actually train as a small adapter rather than a full
// model: nothing here changes for Model types that don't define
// lora_parameters() (e.g. plain GPT<T, LinearT>) — they fall through to the original
// `parameters()`-based behaviour untouched.
namespace trainer_detail {
    template<typename M, typename = void>
    struct has_lora_parameters : std::false_type {};
    template<typename M>
    struct has_lora_parameters<M, std::void_t<decltype(std::declval<M&>().lora_parameters())>>
        : std::true_type {};
}

// ---------------------------------------------------------------------------
// Trainer<Model, T>
//
// Single, central place that owns the optimizer + ALL training/aggregation
// operations for any model that exposes:
//
//   Tensor_t<T> forward(Tensor_t<T> inputs, Tensor_t<T> targets, bool apply_mask)
//   std::vector<Parameter_t<T>*>  parameters()
//   size_t                        get_vocab_size()
//
// ── Ownership contract ───────────────────────────────────────────────────────
//
//  CLIENT side  (mode = FEDAVG | FEDDISTILL)
//    • Owns the STUDENT model only — no teacher model.
//    • finetune()        — local training steps (FedAvg local round).
//    • compute_logits()  — forward-only pass for FedDistill phase 1.
//    • distill_logits()  — trains against the server-supplied consensus tensor;
//                          the "teacher" is a Tensor_t received over the wire,
//                          not a live model — client never loads a teacher.
//
//  SERVER side  (mode = FEDAVG | FEDDISTILL, lr = 0)
//    • Owns the GLOBAL model (acts as teacher signal source when needed).
//    • aggregate()        — FedAvg: average client deltas → overwrite global model.
//    • aggregate_logits() — FedDistill: average client logit vectors → consensus.
//    • get_flat_weights() — serialize global model for broadcast.
//    • set_flat_weights() — deserialize received flat buffer into global model.
//    • Server never calls finetune() / distill() — optimizer is present but
//      unused (lr = 0 ensures zero-effect if step() is called accidentally).
//
//  LOCAL distillation (mode = DISTILL, non-federated)
//    • distill()          — requires a non-null teacher pointer in the same
//                           process. Not used in federated clients.
//
// ---------------------------------------------------------------------------

template<typename Model, typename T>
class Trainer {
    Model& student;
    Model* teacher;   // nullable — only for local (non-federated) distillation
    Optimizer<T> op;

    T temperature;
    T alpha;
    TrainMode mode;

    static const char* mode_name(TrainMode m) {
        switch (m) {
            case FINETUNE:   return "finetune";
            case DISTILL:    return "distill";
            case FEDAVG:     return "fedavg";
            case FEDMETA:    return "fedmeta";
            case FEDDISTILL: return "feddistill";
        }
        return "trainer";
    }

    // Picks Model::lora_parameters() when available, else the full
    // Model::parameters(). See trainer_detail::has_lora_parameters above.
    static std::vector<Tensor_t<T>> select_optimizer_params(Model& m) {
        if constexpr (trainer_detail::has_lora_parameters<Model>::value)
            return m.lora_parameters();
        else
            return m.parameters();
    }

public:
    using BatchFn = std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)>;

    // Full constructor — local distillation capable (teacher != nullptr).
    //
    // The optimizer trains select_optimizer_params(student): the full model
    // for a plain GPT<T, LinearT> student, or just {A,B} for a GPTLoRA<T> student.
    // Either way, get_flat_weights()/set_flat_weights()/aggregate() below
    // still operate over student.parameters() (the COMPLETE model) — so
    // checkpoints and FedAvg weight averaging stay correct regardless of
    // which parameters the optimizer itself is touching.
    Trainer(Model& student,
            Model* teacher,
            T lr,
            TrainMode mode  = FINETUNE,
            T temperature   = T(3.0),
            T alpha         = T(0.5))
        : student(student),
          teacher(teacher),
          op(select_optimizer_params(student), lr, ADAMw, /*requires_grad=*/true),
          temperature(temperature),
          alpha(alpha),
          mode(mode)
    {}

    // Convenience constructor: no teacher (federated clients, server aggregation,
    // plain fine-tuning). Pass lr = T(0) on the server side — the optimizer is
    // present but never driven backward, so it has zero effect.
    Trainer(Model& student, T lr, TrainMode mode = FINETUNE)
        : Trainer(student, nullptr, lr, mode)
    {}

    // ── Internal helpers ─────────────────────────────────────────────────────

    Tensor_t<T> forward_logits(Tensor_t<T> inputs) {
        return student.forward(inputs, nullptr, /*apply_mask=*/true);
    }

    Tensor_t<T> forward(Tensor_t<T> inputs) {
        return forward_logits(inputs)->softmax();
    }

    void freeze(Model& m) {
        for (auto& p : m.parameters())
            p->requires_grad = false;
    }
    void unfreeze(Model& m) {
        for (auto& p : m.parameters())
            p->requires_grad = true;
    }

    void set_mode(TrainMode m) { mode = m; }
    TrainMode get_mode() const { return mode; }

    // ── Distillation loss ────────────────────────────────────────────────────
    //
    //  L = alpha * CE(hard_targets, student_probs)
    //    + (1-alpha) * T^2 * CE(teacher_soft, student_soft)
    //
    Tensor_t<T> distill_loss(Tensor_t<T> student_logits,
                              Tensor_t<T> teacher_logits,
                              Tensor_t<T> one_hot_targets)
    {
        auto student_probs = student_logits->softmax();
        auto hard_loss = Tensor<T>::cross_entropy(one_hot_targets, student_probs);

        auto T_ten         = make_tensor<T>(temperature);
        auto teacher_soft  = (teacher_logits / T_ten)->softmax();
        auto student_soft  = (student_logits / T_ten)->softmax();
        auto soft_loss = Tensor<T>::cross_entropy(teacher_soft, student_soft)
                         * make_tensor<T>(temperature * temperature);

        auto a     = make_tensor<T>(alpha);
        auto one_a = make_tensor<T>(T(1.0) - alpha);
        return a * hard_loss + one_a * soft_loss;
    }

    // ── Public training API (client-side) ─────────────────────────────────────

    // Central training loop — pulls a fresh batch every iteration via
    // get_batch(split). Returns the final-step loss.
    T finetune(BatchFn get_batch, int iters,
               int eval_interval = 10, bool verbose = true,
               const std::string& split = "train")
    {
        T last_loss = T(0);
        for (int i = 0; i < iters; i++) {
            auto [inputs, targets] = get_batch(split);

            op.zero_grad();
            auto loss = student.forward(inputs, targets, /*apply_mask=*/true);
            loss->backward(Matrix<T>(T(1.0)));
            op.step();

            last_loss = loss->val.data[0];
            loss->reset_graph();

            if (verbose && eval_interval > 0 && (i % eval_interval == 0))
                std::cout << "[trainer:" << mode_name(mode) << "]  iter " << i
                          << "  loss: " << last_loss << "\n";
        }
        return last_loss;
    }

    // Back-compat overload — single fixed (X, y) batch repeated `iters` times.
    T finetune(Tensor_t<T> X, Tensor_t<T> y, int iters) {
        return finetune([&](const std::string&) { return std::make_pair(X, y); },
                         iters);
    }

    // Local distillation: student learns from a live teacher model in the same
    // process. NOT used by federated clients (use distill_logits() instead).
    void distill(BatchFn get_batch, int iters) {
        if (!teacher)
            throw std::runtime_error(
                "Trainer::distill: no teacher model — federated clients must "
                "use distill_logits() with the server-supplied consensus tensor.");

        freeze(*teacher);

        for (int i = 0; i < iters; ++i) {
            op.zero_grad();
            auto [inputs, targets] = get_batch("train");

            size_t B = inputs->shape[0];
            size_t S = inputs->shape[1];
            size_t V = student.get_vocab_size();

            auto student_logits = student.forward(inputs, nullptr, true)
                                         ->reshape({B * S, V});
            auto teacher_logits = teacher->forward(inputs, nullptr, true)
                                         ->reshape({B * S, V});

            auto targets_flat = targets->reshape({B * S});
            auto one_hot       = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, V));

            auto loss = distill_loss(student_logits, teacher_logits, one_hot);
            loss->backward(make_tensor<T>(T(1.0)));
            op.step();
            loss->reset_graph();

            if (i % 100 == 0)
                std::cout << "[trainer:distill]  iter " << i << "  loss: " << loss->val << "\n";
        }

        unfreeze(*teacher);
    }

    // ── Federated distillation — client half ─────────────────────────────────
    //
    // FedDistill splits into two Trainer calls per round:
    //
    //  Phase 1 — compute_logits(proxy_inputs)
    //    Forward-only, no grad. Client sends the returned flat matrix to the
    //    server via Client::sendLogits(). No teacher model needed.
    //
    //  Phase 2 — distill_logits(proxy_inputs, local_targets, consensus)
    //    Server broadcasts averaged consensus; caller reshapes it to
    //    [B*S, vocab_size] and passes it here as `teacher_logits`. Trains
    //    student against that external signal — no live teacher required.

    /// Inference-only forward pass over a proxy batch.
    /// Returns raw logits shaped [B*S, vocab_size] for sendLogits().
    Matrix<T> compute_logits(Tensor_t<T> inputs) {
        size_t B = inputs->shape[0];
        size_t S = inputs->shape[1];
        size_t V = student.get_vocab_size();
        // Reshape to [B*S, V] so n_examples and vocab_size are unambiguous
        // for the wire protocol (Client::sendLogits).
        auto logits = forward_logits(inputs)->reshape({B * S, V});
        Matrix<T> out = logits->val;
        logits->reset_graph();
        return out;  // shape: {B*S, V}
    }

    /// One federated-distillation training step.
    ///
    ///   inputs / targets   — client's own local labelled batch (same proxy
    ///                        batch used in compute_logits() this round).
    ///   teacher_logits     — server's averaged consensus shaped [B*S, vocab_size],
    ///                        reconstructed from Client::receiveLogits().
    ///
    /// The teacher signal is supplied externally (from the wire), so this client
    /// never loads or references a teacher model.
    T distill_logits(Tensor_t<T> inputs, Tensor_t<T> targets,
                      Tensor_t<T> teacher_logits)
    {
        op.zero_grad();

        size_t B = inputs->shape[0];
        size_t S = inputs->shape[1];
        size_t V = student.get_vocab_size();

        auto student_logits = student.forward(inputs, nullptr, true)
                                     ->reshape({B * S, V});

        auto targets_flat = targets->reshape({B * S});
        auto one_hot       = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, V));

        auto loss = distill_loss(student_logits, teacher_logits, one_hot);
        loss->backward(make_tensor<T>(T(1.0)));
        op.step();

        T last_loss = loss->val.data[0];
        loss->reset_graph();

        std::cout << "[trainer:" << mode_name(mode) << "]  loss: " << last_loss << "\n";
        return last_loss;
    }

    // Compute and print validation loss without updating weights.
    void evaluate(BatchFn get_batch) {
        auto [inputs, targets] = get_batch("val");
        auto val_loss = student.forward(inputs, targets, /*apply_mask=*/true);
        std::cout << "[trainer:" << mode_name(mode) << "]  val loss: " << val_loss->val << "\n";
        val_loss->reset_graph();
    }

    // ── Federated aggregation — server-side ──────────────────────────────────
    //
    // These methods are called exclusively by the server's Trainer instance.
    // Clients never call them.

    /// FedAvg: element-wise mean of all client delta tensors, then overwrite
    /// the student (global model) parameters in place.
    ///
    /// Called once per round after all client updates arrive. The `updates`
    /// vector is built by Server::handleClient() and cleared after this call.
    void aggregate(const std::vector<std::vector<Tensor_t<T>>>& updates) {
        if (updates.empty())
            throw std::runtime_error("Trainer::aggregate: no updates received");

        const size_t n_clients = updates.size();
        const size_t n_layers  = updates[0].size();

        // Element-wise average across clients.
        std::vector<Tensor_t<T>> avg(n_layers);
        for (size_t l = 0; l < n_layers; ++l) {
            avg[l] = make_tensor<T>(Matrix<T>::zeros(updates[0][l]->shape));
            for (size_t c = 0; c < n_clients; ++c)
                avg[l] = avg[l] + updates[c][l];
            avg[l] = avg[l] / static_cast<T>(n_clients);
        }

        // Write averaged tensors back into student (global model) parameters.
        auto params = student.parameters();
        if (avg.size() != params.size())
            throw std::runtime_error("Trainer::aggregate: param/avg size mismatch");
        for (size_t i = 0; i < params.size(); ++i)
            params[i]->val = avg[i]->val;
    }

    /// FedDistill consensus: element-wise mean of per-client flat logit vectors.
    ///
    /// Architecture-agnostic — only the proxy-batch × vocab size needs to
    /// match across clients, not the parameter layout. Returns the consensus
    /// vector for Server::broadcastLogits().
    std::vector<T> aggregate_logits(const std::vector<std::vector<T>>& logit_updates) {
        if (logit_updates.empty())
            throw std::runtime_error("Trainer::aggregate_logits: no logits received");

        const size_t n_clients = logit_updates.size();
        const size_t n         = logit_updates[0].size();

        std::vector<T> avg(n, T(0));
        for (const auto& client_logits : logit_updates) {
            if (client_logits.size() != n)
                throw std::runtime_error(
                    "Trainer::aggregate_logits: size mismatch across clients "
                    "(proxy batch × vocab must match)");
            for (size_t i = 0; i < n; ++i)
                avg[i] += client_logits[i];
        }
        for (auto& v : avg) v /= static_cast<T>(n_clients);
        return avg;
    }

    // ── Weight serialisation helpers ─────────────────────────────────────────
    //
    // Used on both sides:
    //   Server: get_flat_weights() → broadcast to clients.
    //   Client: set_flat_weights() → apply received global weights to student.
    //           get_flat_weights() → pack deltas before send_deltas().

    /// Pack all student parameters into a contiguous flat vector.
    /// Wire format: [T × total_params] preceded by [uint64 total_bytes]
    /// added by Client::send() / Server::broadcast().
    std::vector<T> get_flat_weights() {
        std::vector<T> flat;
        for (auto& p : student.parameters())
            flat.insert(flat.end(), p->val.data.begin(), p->val.data.end());
        return flat;
    }

    /// Unpack a flat vector (received from the wire) back into student params.
    /// Throws if the buffer is too small for the current model layout.
    void set_flat_weights(const std::vector<T>& flat) {
        auto params = student.parameters();
        size_t offset = 0;
        for (auto& p : params) {
            size_t n = p->val.get_size();
            if (offset + n > flat.size())
                throw std::runtime_error(
                    "Trainer::set_flat_weights: buffer underflow — received "
                    "fewer bytes than the model's parameter count");
            std::copy(flat.begin() + offset, flat.begin() + offset + n,
                      p->val.data.begin());
            offset += n;
        }
    }

    // ── Checkpointing ─────────────────────────────────────────────────────────
    //
    // "Who calls save()" lives in client.cpp / server.cpp (ownership is a
    // round-coordination decision, not a training-math one) — but the actual
    // I/O is centralised here, same as get_flat_weights()/set_flat_weights(),
    // so there's one place that knows how to serialise `student`.
    //
    // Plain float — full fidelity, resumable. Always use this if you intend
    // to keep training/distilling after loading.
    void save_checkpoint(const std::string& path) { student.save(path); }
    void load_checkpoint(const std::string& path) { student.load(path); }

    // Quantized — ~4× (FP8) / ~8× (FP4, nibble-packed) smaller on disk. This
    // is the "ensure the distilled model is indeed a smaller version" path.
    // Training still happens in float T; only the bytes at rest shrink.
    // Loading a quantized checkpoint dequantizes straight back into `student`
    // — distillation can continue immediately afterwards, same as load_checkpoint.
    template<typename QT>
    void save_quantized_checkpoint(const std::string& path, size_t block_size = 32) {
        quant::save_module<QT>(student, path, block_size);
    }
    template<typename QT>
    void load_quantized_checkpoint(const std::string& path) {
        quant::load_module_into<QT>(student, path);
    }
};

#endif // __TRAINER__HPP_