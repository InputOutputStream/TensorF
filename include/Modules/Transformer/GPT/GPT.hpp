#ifndef __GPT_HPP__
#define __GPT_HPP__

#include "../../../Types/types.hpp"
#include "../../Module.hpp"

#include "../../PositionalEncoding.hpp"
#include "../../Embedding.hpp"
#include "../../LayerNorm.hpp"
#include "../../Linear.hpp"

#include "Block.hpp"

#include <vector>
#include <functional>
   
    template <typename T>
    class GPT: public Module<T>{

        private:
            size_t vocab_size=0;
            size_t max_sequence_length=0;   
            PositionalEncoding<T> position_embedding_table;
            Embedding<T> embedding_table;
            std::vector<std::unique_ptr<Block<T>>> decoder_blocks;
            LayerNorm<T> ln_f; 
            Linear<T> lm_head; 
    
        public:

        GPT(size_t vocab_size, size_t input_dim, size_t block_size, 
            size_t n_heads, size_t n_layer):
            vocab_size(vocab_size),
            max_sequence_length(block_size),
            position_embedding_table(input_dim, block_size),
            embedding_table(vocab_size, input_dim),

            ln_f({input_dim}),
            lm_head(input_dim, vocab_size)
        {
            this->register_module(&position_embedding_table);
            this->register_module(&embedding_table);
            this->register_module(&ln_f);
            this->register_module(&lm_head);

            decoder_blocks.reserve(n_layer);
            for(size_t i = 0; i < n_layer; i++){
                decoder_blocks.push_back(std::make_unique<Block<T>>(input_dim, block_size, n_heads));
                this->register_module(decoder_blocks.back().get());
            }
        }

        size_t get_vocab_size() const { return vocab_size; }

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


    friend class GGUFLoader<T>;
    friend class GPTGGUFLoader<T>; 

};

#endif
