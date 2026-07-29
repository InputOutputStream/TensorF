#ifndef __LLAMA_HPP__
#define __LLAMA_HPP__

#include "Types/types.hpp"
#include "../../Module.hpp"

#include "../../Embedding.hpp"
#include "../../RMSNorm.hpp"
#include "../../Linear.hpp"
#include "../../LoRALinear.hpp"

#include "Block.hpp"

#include <vector>
#include <functional>

// Templated on T (compute dtype) and LinearT (linear-layer implementation,
// Linear or LoRALinear) — same pattern as GPT<T, LinearT>. Instantiate
// Llama<T, Linear> for the dense/pretrained model, Llama<T, LoRALinear> for
// an adapter-only fine-tuning student built via load_backbone_from().
template <typename T, template<typename> class LinearT>
class Llama: public Module<T>{

    private:
        size_t vocab_size = 0;
        size_t max_sequence_length = 0;
        size_t ffn_hidden = 0;
        Embedding<T> embedding_table;
        std::vector<std::unique_ptr<Block<T, LinearT>>> decoder_blocks;
        RMSNorm<T> rms;

    public:
        LinearT<T> lm_head;

        // Extra ctor args (Args&&...) are forwarded to every LinearT<T> built
        // inside (lm_head, and each Block's gate/up/down/Q/K/V/proj projections):
        // for Linear that's (bias), for LoRALinear that's (rank, alpha).
        // NOTE: the original dense Llama hardcoded bias=false on the FFN's
        // internal projections but left lm_head/mha default (bias=true) — since
        // args are now shared across all of them, passing `false` here means
        // no bias anywhere, including lm_head and attention Q/K/V/proj (all of
        // which were bias=true by default before). If you need mixed bias
        // behaviour restored exactly, that requires giving FeedForward its own
        // separate Args pack rather than reusing the Block-wide one — flagging
        // this rather than silently changing it.
        template <typename... Args>
        Llama(size_t vocab_size, size_t input_dim, size_t block_size,
              size_t n_heads, size_t n_layer, size_t ffn_hidden, Args&&... args)
            : vocab_size(vocab_size),
              max_sequence_length(block_size),
              ffn_hidden(ffn_hidden),
              embedding_table(vocab_size, input_dim),
              rms({input_dim}),
              lm_head(vocab_size, input_dim, args...)
        {
            this->register_module(&embedding_table);
            this->register_module(&lm_head);
            this->register_module(&rms);

            decoder_blocks.reserve(n_layer);
            for (size_t i = 0; i < n_layer; i++) {
                decoder_blocks.push_back(
                    std::make_unique<Block<T, LinearT>>(input_dim, block_size, n_heads, ffn_hidden, args...));
                this->register_module(decoder_blocks.back().get());
            }
        }

        size_t get_vocab_size() const { return vocab_size; }
        size_t get_block_size() const { return max_sequence_length; }
        Embedding<T>& get_embedding_table() { return embedding_table; }
        RMSNorm<T>&   get_rms()             { return rms; }
        LinearT<T>&   get_lm_head()         { return lm_head; }
        const std::vector<std::unique_ptr<Block<T, LinearT>>>& get_blocks() const { return decoder_blocks; }

        
        template <template<typename> class OtherLinearT>
        void load_backbone_from(Llama<T, OtherLinearT>& pretrained) {
            copy_params(pretrained.get_embedding_table().parameters(),
                        embedding_table.parameters());
            copy_params(pretrained.get_rms().parameters(),
                        rms.parameters());

            auto& pblocks = pretrained.get_blocks();
            if (pblocks.size() != decoder_blocks.size())
                throw std::runtime_error(
                    "Llama::load_backbone_from: layer count mismatch (this=" +
                    std::to_string(decoder_blocks.size()) + ", pretrained=" +
                    std::to_string(pblocks.size()) +
                    ") — construct Llama with the same n_layer as the pretrained model.");

            for (size_t i = 0; i < decoder_blocks.size(); i++)
                decoder_blocks[i]->load_pretrained(*pblocks[i]);
        }

        template <template<typename> class OtherLinearT>
        void load_head_from(Llama<T, OtherLinearT>& pretrained) {
            lm_head.weight->val.copy_from(pretrained.get_lm_head().weight->val);
        }

        Tensor_t<T> forward(Tensor_t<T> index, Tensor_t<T> targets, bool apply_mask = true)
        {
            size_t batch_size = index->shape[0];
            size_t seq_len    = index->shape[1];

            // Token embeddings — RoPE is applied inside each Head, not here
            Tensor_t<T> x = this->embedding_table.forward(index);  // (B, T, D)

            // Transformer blocks
            for (auto& block : this->decoder_blocks)
                x = block->forward(x, apply_mask);

            // Final RMS norm
            Tensor_t<T> x_normed = this->rms.forward(x);

            // Language-model head
            Tensor_t<T> output = this->lm_head.forward(x_normed);  // (B, T, vocab_size)

            // Loss
            if (targets != nullptr) {
                auto logits_flat  = output->reshape({batch_size * seq_len, this->vocab_size});
                auto targets_flat = targets->reshape({batch_size * seq_len});
                auto targets_onehot = make_tensor<T>(
                    Matrix<T>::one_hot(targets_flat->val, this->vocab_size));
                auto probs = logits_flat->softmax();
                return Tensor<T>::cross_entropy(targets_onehot, probs);
            }

            return output;
        }

        std::vector<T> topk_softmax(const std::vector<T>& logits, size_t V,
                                    int k = 40, float temp = 0.8f)
        {
            std::vector<T> scaled(logits.begin(), logits.begin() + V);
            for (auto& v : scaled) v /= temp;

            std::vector<size_t> indices(V);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + k, indices.end(),
                [&scaled](size_t a, size_t b){ return scaled[a] > scaled[b]; });

            T mx  = scaled[indices[0]];
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
                             float temperature = 0.7f, size_t k = 40)
        {
            Tensor_t<T> current_index = index;
            const int eos_token_id = 2;   // <|im_end|>

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

                Matrix<T>   last_step = output->val.slice_axis(S - 1, S, 1);
                Tensor_t<T> logits    = make_tensor<T>(last_step.reshape({B, V}));

                std::vector<Matrix<T>> index_next;
                for (size_t j = 0; j < B; j++)
                {
                    std::vector<T> row(logits->val.data.begin() + j * V,
                                       logits->val.data.begin() + j * V + V);
                    auto probs_vec = topk_softmax(row, V, k, temperature);
                    Matrix<T> prob_mat(probs_vec, {V});
                    Matrix<T> next_tok = Matrix<T>::choice(V, prob_mat);
                    index_next.push_back(next_tok.reshape({1, 1}));
                }

                Matrix<T> new_tokens = Matrix<T>::stack(index_next, 0);
                current_index = make_tensor<T>(
                    Matrix<T>::concat({current_index->val, new_tokens}, 1));
                
                // Check EOS for the first batch element (assuming batch size 1)
                auto last_id = current_index->val.at({0, current_index->shape[1] - 1});
                if (last_id.data[0] == eos_token_id)
                    break;
            }

            return current_index;
        }


        Tensor_t<T> train_step(Optimizer<T>* Op, Tensor_t<T> inputs, Tensor_t<T> targets)
        {
            Op->zero_grad();
            Tensor_t<T> loss = this->forward(inputs, targets, true);
            loss->backward(make_tensor<T>((T)1.0));
            Op->step();
            return loss;
        }

        void train(std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn,
                   size_t iters, size_t eval_interval, bool requires_grad)
        {
            Optimizer<T> Op(this->parameters(), 1e-4, ADAMw, requires_grad);

            for (size_t iter = 0; iter < iters; iter++)
            {
                auto [inputs, targets] = get_batch_fn("train");
                Tensor_t<T> loss = this->train_step(&Op, inputs, targets);

                if (iter % eval_interval == 0)
                    std::cout << "Iter: " << iter << "  Loss: " << loss->val << "\n";
                    
                if ((iter % eval_interval == 0) && iter > 0) {
                    Op.zero_grad();
                    this->eval_step(get_batch_fn);
                }

                loss->reset_graph();
            }
        }

        void eval_step(
            std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn)
        {
            auto [inputs, targets] = get_batch_fn("val");
            auto val_loss = this->forward(inputs, targets, true);
            std::cout << "Validation Loss: " << val_loss->val << "\n";
        }

        std::vector<Tensor_t<T>> lora_parameters() const {
            return lm_head.parameters();
        }

        std::vector<Tensor_t<T>> backbone_parameters() const {
            std::vector<Tensor_t<T>> out;
            auto add = [&](const std::vector<Tensor_t<T>>& v) { out.insert(out.end(), v.begin(), v.end()); };
            add(embedding_table.parameters());
            add(rms.parameters());
            for (auto& b : decoder_blocks) add(b->parameters());
            return out;
        }

    friend class GGUFLoader<T>;
    template <typename, template<typename> class> friend class LlamaGGUFLoader;

    private:
        static void copy_params(const std::vector<Tensor_t<T>>& src,
                                const std::vector<Tensor_t<T>>& dst) {
            if (src.size() != dst.size())
                throw std::runtime_error(
                    "Llama::copy_params: size mismatch while copying pretrained weights "
                    "(src=" + std::to_string(src.size()) + ", dst=" + std::to_string(dst.size()) + ")");
            for (size_t i = 0; i < src.size(); i++)
                dst[i]->val.copy_from(src[i]->val);
        }
};

template <typename T> using LlamaDense = Llama<T, Linear>;
template <typename T> using LlamaLoRA  = Llama<T, LoRALinear>;

#endif // !__LLAMA_HPP__