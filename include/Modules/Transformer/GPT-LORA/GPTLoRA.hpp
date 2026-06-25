#ifndef __GPT_LORA_HPP__
#define __GPT_LORA_HPP__

#include "../../../Types/types.hpp"
#include "../../Module.hpp"

#include "../../PositionalEncoding.hpp"
#include "../../Embedding.hpp"
#include "../../LayerNorm.hpp"
#include "../../LoRALinear.hpp"

#include "Block.hpp"
#include "GPT.hpp"     // reused only as the *source* of pretrained weights — see load_backbone_from()

#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>

// ─────────────────────────────────────────────────────────────────────────────
// GPTLoRA<T> — distillation student with a LoRA-adapted LM head.
//
// Why the head specifically: in a GPT-2-small-shaped model the LM head
// (vocab_size × d_model, e.g. 50257×768 ≈ 38.6M params) is the single
// largest weight matrix outside the (often tied) token embedding — and it's
// the one tensor GPT.hpp already exposes cleanly enough to swap.
//
// Why not also the attention/MLP projections inside each Block: those
// Linear<T> instances are private to Block.hpp, which wasn't available
// when this file was written. The pattern is identical — templatize Block
// on its projection type (`Block<T, LinearT = Linear<T>>`) and pass
// LoRALinear<T> the same way this file does for the head — apply that
// inside Block.hpp once you're ready to LoRA-fy attention/MLP too.
//
// ── Ownership / training story ────────────────────────────────────────────
//
//   backbone (position/token embedding, attention blocks, final LayerNorm)
//     → loaded from a pretrained GPT<T> via load_backbone_from(), then left
//       untouched. Still gets gradients during backward() (this codebase has
//       no engine-level requires_grad cutoff), it's just never written to,
//       because...
//
//   lm_head (LoRALinear<T>: frozen `weight` + trainable `A`,`B`)
//     → load_head_from() seeds `weight` from the pretrained head, A/B start
//       at their LoRALinear-default init (A random, B zero, so the adapter
//       starts as a no-op exactly like vanilla LoRA).
//     → lora_parameters() returns ONLY {A,B}. Trainer<GPTLoRA<T>,T> detects
//       this method (see the SFINAE dispatch added to Trainner.hpp) and
//       builds its optimizer from lora_parameters() instead of the full
//       parameters() list — so only the adapter trains.
//
//   parameters() (inherited from Module<T>, untouched/un-hidden here)
//     → still returns EVERYTHING (backbone + A + B), so save()/load()
//       round-trip a complete, self-contained checkpoint. This is what you
//       want for "resume distillation later without the original GGUF."
//
//   backbone_parameters() (added below)
//     → everything EXCEPT the adapter. Useful if you want a split
//       checkpoint: one (large, FP8/FP4-quantized, save-once, shared by
//       every client) backbone file + one (tiny, float, per-client) adapter
//       file — see usage notes at the bottom of this file.
// ─────────────────────────────────────────────────────────────────────────────

template <typename T>
class GPTLoRA : public Module<T> {

    size_t vocab_size = 0;
    size_t max_sequence_length = 0;

    PositionalEncoding<T> position_embedding_table;
    Embedding<T>          embedding_table;
    std::vector<std::unique_ptr<Block<T>>> decoder_blocks;
    LayerNorm<T>          ln_f;

public:
    LoRALinear<T> lm_head;

    GPTLoRA(size_t vocab_size, size_t input_dim, size_t block_size,
            size_t n_heads, size_t n_layer,
            size_t lora_rank = 8, T lora_alpha = T(16))
        : vocab_size(vocab_size),
          max_sequence_length(block_size),
          position_embedding_table(input_dim, block_size),
          embedding_table(vocab_size, input_dim),
          ln_f({input_dim}),
          lm_head(vocab_size, input_dim, lora_rank, lora_alpha)
    {
        this->register_module(&position_embedding_table);
        this->register_module(&embedding_table);
        this->register_module(&ln_f);
        this->register_module(&lm_head);   // registers {A,B} only — weight stays untracked by design (LoRALinear's own ctor)

        decoder_blocks.reserve(n_layer);
        for (size_t i = 0; i < n_layer; i++) {
            decoder_blocks.push_back(std::make_unique<Block<T>>(input_dim, block_size, n_heads));
            this->register_module(decoder_blocks.back().get());
        }
    }

    size_t get_vocab_size() const { return vocab_size; }

    // ── Pretrained-weight bridge ──────────────────────────────────────────
    //
    // GPTGGUFLoader<T> only knows how to populate a GPT<T> (its friend
    // declarations are in GPT.hpp, not here), so we don't load GGUF
    // directly into GPTLoRA. Instead: load a normal GPT<T> the usual way,
    // then copy tensor-for-tensor into the matching GPTLoRA submodules.
    //
    //   GPTGGUFLoader<float> loader;
    //   GPT<float> pretrained = loader.load_model(model_path, hp);
    //   GPTLoRA<float> student(hp.vocab_size, hp.d_model, hp.block_size,
    //                          hp.n_head, hp.n_layer, /*rank=*/8, /*alpha=*/16.0f);
    //   student.load_backbone_from(pretrained);
    //   student.load_head_from(pretrained);
    //   // `pretrained` can be destroyed after this — its weights are copied, not referenced.

    void load_backbone_from(GPT<T>& pretrained) {
        copy_params(pretrained.get_position_embedding().parameters(),
                    position_embedding_table.parameters());
        copy_params(pretrained.get_embedding_table().parameters(),
                    embedding_table.parameters());
        copy_params(pretrained.get_ln_f().parameters(),
                    ln_f.parameters());

        auto& pblocks = pretrained.get_blocks();
        if (pblocks.size() != decoder_blocks.size())
            throw std::runtime_error(
                "GPTLoRA::load_backbone_from: layer count mismatch (this=" +
                std::to_string(decoder_blocks.size()) + ", pretrained=" +
                std::to_string(pblocks.size()) +
                ") — construct GPTLoRA with the same n_layer as the pretrained model.");
        for (size_t i = 0; i < decoder_blocks.size(); i++)
            copy_params(pblocks[i]->parameters(), decoder_blocks[i]->parameters());
    }

    // ASSUMPTION: Linear<T> exposes a public `Tensor_t<T> weight` member,
    // mirroring LoRALinear<T>'s own field of the same name (see
    // LoRALinear.hpp). If Linear<T>'s field is named differently, this is
    // the one line to fix — everything else in this file is independent of
    // Linear<T>'s internals.
    void load_head_from(GPT<T>& pretrained) {
        lm_head.weight->val.copy_from(pretrained.get_lm_head().weight->val);
    }

    // ── Forward / generate — identical to GPT<T>, lm_head swapped for the
    //    LoRA-adapted version. Kept as a full copy rather than inheriting
    //    GPT<T> so there's no redundant plain-Linear head wasting memory. ──

    Tensor_t<T> forward(Tensor_t<T> index, Tensor_t<T> targets, bool apply_mask = true) {
        size_t batch_size = index->shape[0], seq_len = index->shape[1];

        Tensor_t<T> tok_embed = embedding_table.forward(index);
        Tensor_t<T> pos_indices = make_tensor<T>(Matrix<T>::arrange(seq_len).reshape({1, seq_len}));
        Tensor_t<T> pos_embed = position_embedding_table.forward(pos_indices);
        Tensor_t<T> x_emdb = tok_embed + pos_embed;

        for (auto& block : decoder_blocks)
            x_emdb = block->forward(x_emdb, apply_mask);

        Tensor_t<T> x_after_ln = ln_f.forward(x_emdb);
        Tensor_t<T> output = lm_head.forward(x_after_ln);   // ← LoRALinear, not Linear

        if (targets != nullptr) {
            auto logits_flat = output->reshape({batch_size * seq_len, vocab_size});
            auto targets_flat = targets->reshape({batch_size * seq_len});
            auto targets_onehot = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, vocab_size));
            auto probs = logits_flat->softmax();
            return Tensor<T>::cross_entropy(targets_onehot, probs);
        }
        return output;
    }

    std::vector<T> topk_softmax(const std::vector<T>& logits, size_t V,
                                int k = 40, float temp = 0.8f) {
        std::vector<T> scaled(logits.begin(), logits.begin() + V);
        for (auto& v : scaled) v /= temp;

        std::vector<size_t> indices(V);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
            [&scaled](size_t a, size_t b){ return scaled[a] > scaled[b]; });

        T mx = scaled[indices[0]];
        std::vector<T> probs(V, T(0));
        T sum = T(0);
        for (int i = 0; i < k; i++) {
            probs[indices[i]] = std::exp(scaled[indices[i]] - mx);
            sum += probs[indices[i]];
        }
        for (int i = 0; i < k; i++) probs[indices[i]] /= sum;
        return probs;
    }

    Tensor_t<T> generate(Tensor_t<T> index, size_t max_new_tokens = 50,
                         float temperature = 0.7f, size_t k = 40) {
        Tensor_t<T> current_index = index;

        for (size_t i = 0; i < max_new_tokens; i++) {
            size_t seq   = current_index->shape[1];
            size_t start = (seq > max_sequence_length) ? (seq - max_sequence_length) : 0;
            Matrix<T> index_cond = current_index->val.slice_axis(start, seq, 1);

            Tensor_t<T> output = forward(make_tensor<T>(index_cond), nullptr, true);

            size_t B = output->shape[0], S = output->shape[1], V = output->shape[2];
            Matrix<T> last_step = output->val.slice_axis(S - 1, S, 1);
            Tensor_t<T> logits  = make_tensor<T>(last_step.reshape({B, V}));

            std::vector<Matrix<T>> index_next;
            for (size_t j = 0; j < B; j++) {
                std::vector<T> row(logits->val.data.begin() + j * V,
                                   logits->val.data.begin() + j * V + V);
                auto probs_vec = topk_softmax(row, V, k, temperature);
                Matrix<T> prob_mat(probs_vec, {V});
                Matrix<T> next_tok = Matrix<T>::choice(V, prob_mat);
                index_next.push_back(next_tok.reshape({1, 1}));
            }

            Matrix<T> new_tokens = Matrix<T>::stack(index_next, 0);
            current_index = make_tensor<T>(Matrix<T>::concat({current_index->val, new_tokens}, 1));
        }
        return current_index;
    }

    // ── The selective-training hook Trainner.hpp looks for ───────────────
    //
    // LoRALinear<T> only ever registered {A,B} as parameters (its frozen
    // `weight` is a plain member, never passed to register_parameter) — so
    // this is already exactly "the adapter, nothing else."
    std::vector<Tensor_t<T>> lora_parameters() const {
        return lm_head.parameters();
    }

    // Everything EXCEPT the adapter — for the split-checkpoint pattern
    // described at the top of this file (quantize+save this once, share it;
    // save lora_parameters() per-client, small and in plain float).
    std::vector<Tensor_t<T>> backbone_parameters() const {
        std::vector<Tensor_t<T>> out;
        auto add = [&](const std::vector<Tensor_t<T>>& v) { out.insert(out.end(), v.begin(), v.end()); };
        add(position_embedding_table.parameters());
        add(embedding_table.parameters());
        add(ln_f.parameters());
        for (auto& b : decoder_blocks) add(b->parameters());
        return out;
    }

private:
    static void copy_params(const std::vector<Tensor_t<T>>& src,
                            const std::vector<Tensor_t<T>>& dst) {
        if (src.size() != dst.size())
            throw std::runtime_error(
                "GPTLoRA::copy_params: size mismatch while copying pretrained weights "
                "(src=" + std::to_string(src.size()) + ", dst=" + std::to_string(dst.size()) + ")");
        for (size_t i = 0; i < src.size(); i++)
            dst[i]->val.copy_from(src[i]->val);
    }
};

#endif // __GPT_LORA_HPP__

// ─────────────────────────────────────────────────────────────────────────────
// Usage — QLoRA-style distillation student, frozen backbone quantized for
// storage/wire only (compute always stays in float T; see Quantizer.hpp):
//
//   #include "Quantizer.hpp"
//
//   GPTGGUFLoader<float> loader;
//   GPT<float> pretrained = loader.load_model(model_path, hp);
//
//   GPTLoRA<float> student(hp.vocab_size, hp.d_model, hp.block_size,
//                          hp.n_head, hp.n_layer, /*rank=*/8, /*alpha=*/16.0f);
//   student.load_backbone_from(pretrained);
//   student.load_head_from(pretrained);
//
//   Trainer<GPTLoRA<float>, float> trainer(student, lr, FEDAVG);
//   // Trainer's SFINAE check finds lora_parameters() and trains ONLY {A,B}.
//   // backward() still walks through the frozen backbone (no requires_grad
//   // cutoff in this engine), so it costs the same forward/backward time as
//   // a full model — what shrinks is the optimizer state and the federated
//   // wire payload (trainer.get_flat_weights() with this Model still flattens
//   // student.parameters(), i.e. backbone+adapter; if you want the wire
//   // payload itself to shrink too, send lora_parameters() over the wire
//   // instead — that's a one-line change in client.cpp's send_deltas()).
//
//   // ... train rounds ...
//
//   student.save(adapter_and_backbone_path);                 // full fp32 checkpoint, self-contained
//   quant::save_module<fp8_e4m3>(student, backbone_fp8_path); // OR: ~4x smaller, shared across clients
// ─────────────────────────────────────────────────────────────────────────────