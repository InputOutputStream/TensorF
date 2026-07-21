#ifndef __GPT_HPP__
#define __GPT_HPP__

#include "../../../Types/types.hpp"
#include "../../Module.hpp"

#include "../../PositionalEncoding.hpp"
#include "../../Embedding.hpp"
#include "../../LayerNorm.hpp"
#include "../../Linear.hpp"
#include "../../LoRALinear.hpp"

#include "Block.hpp"

#include <vector>
#include <functional>
#include <numeric>
#include <algorithm>
   

template <typename T, template<typename> class LinearT>
    class GPT: public Module<T>{

        private:
            size_t vocab_size=0;
            size_t max_sequence_length=0;   
            PositionalEncoding<T> position_embedding_table;
            Embedding<T> embedding_table;
            std::vector<std::unique_ptr<Block<T, LinearT>>> decoder_blocks;
            LayerNorm<T> ln_f; 

        public:
            LinearT<T> lm_head; 

        // Extra ctor args (Args&&...) are forwarded to every LinearT<T> instance
        // constructed inside (lm_head, and each Block's internal projections):
        // for Linear that's (bias), for LoRALinear that's (rank, alpha).
        template <typename... Args>
        GPT(size_t vocab_size, size_t input_dim, size_t block_size, 
            size_t n_heads, size_t n_layer, Args&&... args):
            vocab_size(vocab_size),
            max_sequence_length(block_size),
            position_embedding_table(input_dim, block_size),
            embedding_table(vocab_size, input_dim),

            ln_f({input_dim}),
            lm_head(vocab_size, input_dim, args...)
        {
            this->register_module(&position_embedding_table);
            this->register_module(&embedding_table);
            this->register_module(&ln_f);
            this->register_module(&lm_head);

            decoder_blocks.reserve(n_layer);
            for(size_t i = 0; i < n_layer; i++){
                decoder_blocks.push_back(std::make_unique<Block<T, LinearT>>(input_dim, block_size, n_heads, args...));
                this->register_module(decoder_blocks.back().get());
            }
        }

        size_t get_vocab_size() const { return vocab_size; }

        // ── Accessors used by load_backbone_from()/load_head_from() below, and by
        //    anything else that wants to reuse a pretrained GPT<T,...>'s backbone
        //    without re-loading from GGUF
        size_t get_block_size() const { return max_sequence_length; }
        PositionalEncoding<T>& get_position_embedding() { return position_embedding_table; }
        Embedding<T>&          get_embedding_table()    { return embedding_table; }
        LayerNorm<T>&          get_ln_f()                { return ln_f; }
        LinearT<T>&            get_lm_head()             { return lm_head; }
        const std::vector<std::unique_ptr<Block<T, LinearT>>>& get_blocks() const { return decoder_blocks; }

        // Copies backbone weights tensor-for-tensor from a pretrained model that
        // may use a *different* LinearT (e.g. a dense GPT<T,Linear> checkpoint
        // loaded from GGUF, feeding a GPT<T,LoRALinear> student). Only the
        // backbone (embeddings/positional/blocks/ln_f) is copied here — the head
        // is handled separately by load_head_from(), since LoRA heads carry
        // {A,B} adapters on top of the copied base weight.
        template <template<typename> class OtherLinearT>
        void load_backbone_from(GPT<T, OtherLinearT>& pretrained) {
            copy_params(pretrained.get_position_embedding().parameters(),
                        position_embedding_table.parameters());
            copy_params(pretrained.get_embedding_table().parameters(),
                        embedding_table.parameters());
            copy_params(pretrained.get_ln_f().parameters(),
                        ln_f.parameters());

            auto& pblocks = pretrained.get_blocks();
            if (pblocks.size() != decoder_blocks.size())
                throw std::runtime_error(
                    "GPT::load_backbone_from: layer count mismatch (this=" +
                    std::to_string(decoder_blocks.size()) + ", pretrained=" +
                    std::to_string(pblocks.size()) +
                    ") — construct GPT with the same n_layer as the pretrained model.");

            // NOTE: blocks go through load_pretrained(), NOT copy_params() over
            // parameters() 
            for (size_t i = 0; i < decoder_blocks.size(); i++)
                decoder_blocks[i]->load_pretrained(*pblocks[i]);
        }

        template <template<typename> class OtherLinearT>
        void load_head_from(GPT<T, OtherLinearT>& pretrained) {
            lm_head.weight->val.copy_from(pretrained.get_lm_head().weight->val);
        }

        Tensor_t<T> forward(Tensor_t<T> index, Tensor_t<T> targets, bool apply_mask=true)
        {
            size_t batch_size = index->shape[0], seq_len = index->shape[1]; 
                    
            // Token embeddings
            Tensor_t<T> tok_embed = this->embedding_table.forward(index);  // (batch_size, seq_len, input_dim)

            // Positional embeddings
            // Give it a {1, seq_len} shape so embed output is {1, seq_len, D}
            // which broadcasts correctly against tok_embed {B, seq_len, D}
            Tensor_t<T> pos_indices = make_tensor<T>(Matrix<T>::arrange(seq_len).reshape({1, seq_len}));// {1, seq_len}
            Tensor_t<T> pos_embed = this->position_embedding_table.forward(pos_indices);  // {1, seq_len, D}

            // Add embeddings
            // Now tok_embed {B, seq_len, D} + pos_embed {1, seq_len, D}
            Tensor_t<T> x_emdb = tok_embed + pos_embed;
            
            // Through transformer blocks
            for(auto& block : this->decoder_blocks)
                x_emdb = block->forward(x_emdb, apply_mask);

            // std::cerr << " x val : "<< x_emdb->val<< "\n";

            // Final layer norm
            Tensor_t<T> x_after_ln = this->ln_f.forward(x_emdb);

            // std::cerr << " x_after_ln val : "<< x_after_ln->val<< "\n";

            // Language modeling head
            Tensor_t<T> output = this->lm_head.forward(x_after_ln);  // (batch_size, seq_len, vocab_size)

            // std::cerr << " output val : "<< output->val<< "\n";

            // Compute loss if targets provided
            if (targets != nullptr){
                // Reshape for loss calculation
                auto logits_flat = output->reshape({batch_size * seq_len, this->vocab_size});  // (batch_size * seq_len, vocab_size)
                // std::cerr << " logits_flat val : "<< logits_flat->val<< "\n";
                auto targets_flat = targets->reshape({batch_size* seq_len});  // (batch_size * seq_len,)
                // std::cerr << " targets_flat val : "<< targets_flat->val<< "\n";
                auto targets_onehot = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, this->vocab_size));
                // std::cerr << " targets_onehot val : "<< targets_onehot->val<< "\n";
                auto probs = logits_flat->softmax();
                // std::cerr << " probs val : "<< probs->val<< "\n";

                return Tensor<T>::cross_entropy(targets_onehot, probs);
            }

            return output;
        }

        std::vector<T> topk_softmax(const std::vector<T>& logits, size_t V, 
                                    int k = 40, float temp = 0.8f) {
            // Apply temperature
            std::vector<T> scaled(logits.begin(), logits.begin() + V);
            for (auto& v : scaled) v /= temp;

            // Find top-k indices by partial sort
            std::vector<size_t> indices(V);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                [&scaled](size_t a, size_t b){ return scaled[a] > scaled[b]; });

            // Softmax over top-k only, zero out the rest
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

        Tensor_t<T> generate(Tensor_t<T> index, size_t max_new_tokens = 50, float temperature = 0.7f, size_t k = 40)
        {
            Tensor_t<T> current_index = index;

            for (size_t i = 0; i < max_new_tokens; i++)
            {
                size_t seq   = current_index->shape[1];
                size_t start = (seq > this->max_sequence_length)
                            ? (seq - this->max_sequence_length) : 0;
                Matrix<T> index_cond = current_index->val.slice_axis(start, seq, 1);

                Tensor_t<T> output = this->forward(make_tensor<T>(index_cond), nullptr, true);

                size_t B = output->shape[0];
                size_t S = output->shape[1];
                size_t V = output->shape[2];

                // Extract last timestep → {B, V}
                Matrix<T> last_step = output->val.slice_axis(S - 1, S, 1);
                Tensor_t<T> logits  = make_tensor<T>(last_step.reshape({B, V}));

                // Sampling loop 
                std::vector<Matrix<T>> index_next;
                for (size_t j = 0; j < B; j++)
                {
                    // Extract row j from last_step {B, V}
                    std::vector<T> row(logits->val.data.begin() + j * V,
                                    logits->val.data.begin() + j * V + V);

                    // Top-k filtered softmax
                    auto probs_vec = topk_softmax(row, V, k, temperature);
                    Matrix<T> prob_mat(probs_vec, {V});

                    Matrix<T> next_tok = Matrix<T>::choice(V, prob_mat);
                    index_next.push_back(next_tok.reshape({1, 1}));
                }

                Matrix<T> new_tokens = Matrix<T>::stack(index_next, 0);
                current_index = make_tensor<T>(
                    Matrix<T>::concat({current_index->val, new_tokens}, 1));
            }

            return current_index;
        }

        Tensor_t<T> train_step(Optimizer<T> *Op, Tensor_t<T> inputs, Tensor_t<T> targets)
        {
            // Zero gradients
            Op->zero_grad();

            // Forward pass
            Tensor_t<T> loss = this->forward(inputs, targets, true);
            
            // Backward pass
            loss->backward(make_tensor<T>((T)1.0));

            // Update parameters
            Op->step();
            
            return loss;
        }


        void train(std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn,
        size_t iters, size_t eval_interval)
        {
            Optimizer<T> Op(this->parameters(), 1e-4, ADAMw);

            for(size_t iter=0; iter < iters; iter++)
            {
                // Get training batch
                auto [inputs, targets] = get_batch_fn("train");
                
                // Training step
                Tensor_t<T> loss = this->train_step(&Op, inputs, targets);
                
                // Print progress
                if (iter % eval_interval == 0){
                    std::cout << "Iters :" << iter << " Loss: " << loss->val << "...............................................................\n";
                }

                // Evaluation
                if((iter % eval_interval == 0) && iters > 0){
                    Op.zero_grad();
                    this->eval_step(get_batch_fn);
                }
                loss->reset_graph();
            }
        }

        void eval_step(std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn) {
            auto [inputs, targets] = get_batch_fn("val");
            auto val_loss = this->forward(inputs, targets, true);
            std::cout << "Validation Loss: " << val_loss->val << "............................................................................\n";
        }

        // friend std::ostream & operator <<(std::ostream &out, Matrix<E> &m);

        // lora_parameters()/backbone_parameters(): kept for Trainer's SFINAE check
        // (it looks for lora_parameters() to decide whether to train only the
        // adapter). For LinearT=Linear these just return the same thing as
        // parameters() would for the head — harmless but meaningless; if that's
        // undesirable, gate these with `if constexpr` on LinearT before relying
        // on them for a dense model.
        std::vector<Tensor_t<T>> lora_parameters() const {
            return lm_head.parameters();
        }

        std::vector<Tensor_t<T>> backbone_parameters() const {
            std::vector<Tensor_t<T>> out;
            auto add = [&](const std::vector<Tensor_t<T>>& v) { out.insert(out.end(), v.begin(), v.end()); };
            add(position_embedding_table.parameters());
            add(embedding_table.parameters());
            add(ln_f.parameters());
            for (auto& b : decoder_blocks) add(b->parameters());
            return out;
        }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class GPTGGUFLoader;
    private:
        static void copy_params(const std::vector<Tensor_t<T>>& src,
                                const std::vector<Tensor_t<T>>& dst) {
            if (src.size() != dst.size())
                throw std::runtime_error(
                    "GPT::copy_params: size mismatch while copying pretrained weights "
                    "(src=" + std::to_string(src.size()) + ", dst=" + std::to_string(dst.size()) + ")");
            for (size_t i = 0; i < src.size(); i++)
                dst[i]->val.copy_from(src[i]->val);
        }
};

// Convenience aliases so call sites can keep using familiar names instead of
// spelling out GPT<T, Linear> / GPT<T, LoRALinear> everywhere.
template <typename T> using GPTDense = GPT<T, Linear>;
template <typename T> using GPTLoRA  = GPT<T, LoRALinear>;

#endif

// ─────────────────────────────────────────────────────────────────────────────
// Usage — QLoRA-style distillation student, frozen backbone quantized for
// storage/wire only (compute always stays in float T; see Quantizer.hpp):
//
//   #include "Quantizer.hpp"
//
//   GPTGGUFLoader<float> loader;
//   GPT<float, Linear> pretrained = loader.load_model(model_path, hp);
//
//   GPT<float, LoRALinear> student(hp.vocab_size, hp.d_model, hp.block_size,
//                          hp.n_head, hp.n_layer, /*rank=*/8, /*alpha=*/16.0f);
//   student.load_backbone_from(pretrained);
//   student.load_head_from(pretrained);
//
//   Trainer<GPT<float, LoRALinear>, float> trainer(student, lr, FEDAVG);
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