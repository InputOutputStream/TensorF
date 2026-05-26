#ifndef __GPT_HPP__
#define __GPT_HPP__

#include "../../Types/types.hpp"
#include "../Module.hpp"

#include "../PositionalEncoding.hpp"
#include "../Embedding.hpp"
#include "../LayerNorm.hpp"
#include "../Linear.hpp"

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

        Tensor_t<T> forward(Tensor_t<T> index, Tensor_t<T> targets, bool apply_mask=true)
        {
            size_t batch_size = index->shape[0], seq_len = index->shape[1]; 
                    
            // Token embeddings
            Tensor_t<T> tok_embed = this->embedding_table.forward(index);  // (batch_size, seq_len, input_dim)
            
            // Positional embeddings
            Tensor_t<T> pos_indices = make_tensor<T>(Matrix<T>::arrange(seq_len));
            Tensor_t<T> pos_embed = this->position_embedding_table.forward(pos_indices);  // (seq_len, input_dim)
            
            // Add embeddings
            Tensor_t<T> x_emdb = tok_embed + pos_embed;
            
            // Through transformer blocks
            for(auto& block : this->decoder_blocks)
                x_emdb = block->forward(x_emdb, apply_mask);

            std::cerr << " x shape : "<< x_emdb->shape<< "\n";

            // Final layer norm
            Tensor_t<T> x_after_ln = this->ln_f.forward(x_emdb);
            
            std::cerr << " x_after_ln shape : "<< x_after_ln->shape<< "\n";

            // Language modeling head
            Tensor_t<T> output = this->lm_head.forward(x_after_ln);  // (batch_size, seq_len, vocab_size)
            
            std::cerr << " output shape : "<< output->shape<< "\n";

            // Compute loss if targets provided
            if (targets != nullptr){
                // Reshape for loss calculation
                auto logits_flat = output->reshape({batch_size * seq_len, this->vocab_size});  // (batch_size * seq_len, vocab_size)
                std::cerr << " logits_flat shape : "<< logits_flat->shape<< "\n";
                auto targets_flat = targets->reshape({batch_size* seq_len});  // (batch_size * seq_len,)
                std::cerr << " targets_flat shape : "<< targets_flat->shape<< "\n";
                auto targets_onehot = make_tensor<T>(Matrix<T>::one_hot(targets_flat->val, this->vocab_size));
                std::cerr << " targets_onehot shape : "<< targets_onehot->shape<< "\n";

                auto probs = logits_flat->softmax();
                std::cerr << " probs shape : "<< probs->shape<< "\n";

                return Tensor<T>::cross_entropy(targets_onehot, probs);
            }

            return output;
        }

        Tensor_t<T> generate(Tensor_t<T> index, size_t max_new_tokens=50)
        {
            // Make a copy to avoid modifying the input
            Tensor_t<T> current_index = index;
            
            for (auto i = 0; i < max_new_tokens; i++)
            {
                // Crop to last block_size tokens                
                size_t seq = current_index->shape[1];
                size_t start = (seq > this->max_sequence_length) ? (seq - this->max_sequence_length) : 0;
                Matrix<T> index_cond = current_index->val.slice_col(start, seq);

                // Get predictions 
                Tensor_t<T> logits = this->forward(make_tensor<T>(index_cond), nullptr, true);
                
                // Focus on last time step
                logits = make_tensor<T>(logits->val.slice_col(logits->shape[1]-1, logits->shape[1])); // (batch_size, vocab_size)

                // Apply softmax to get probabilities
                Tensor_t<T> probs = logits->softmax();
                
                // Sample from distribution
                size_t batch_size = probs->shape[0];
                std::vector<Matrix<T>> index_next;

                for(int j=0; j<batch_size; j++)
                {
                    // Handle potential numerical issues
                    Tensor_t<T> prob_dist = probs->at({j});
                    prob_dist = prob_dist / prob_dist->sum();  // Ensure probabilities sum to 1
                    Tensor_t<T> next_token = make_tensor<T>(Matrix<T>::choice(this->vocab_size, prob_dist->val));
                    index_next.push_back(next_token->val);
                }

                // index_next = index_next->reshape({-1, 1});
                Matrix<T> new_tokens = Matrix<T>::stack(index_next, 0);  // (batch, 1)
                current_index = make_tensor<T>(Matrix<T>::concat({current_index->val, new_tokens}, 1));
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

        loss->reset_graph();
        
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

            // // Evaluation
            // if((iter % eval_interval == 0) && iters > 0){
            //     Op.zero_grad();
            //     this->eval_step(get_batch_fn);
            // }
        }
    }

    void eval_step(std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn) {
        auto [inputs, targets] = get_batch_fn("val");
        auto val_loss = this->forward(inputs, targets, true);
        std::cout << "Validation Loss: " << val_loss->val << "............................................................................\n";
    }

};

#endif
