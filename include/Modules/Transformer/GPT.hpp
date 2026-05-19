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
            
            Embedding<T> embedding_table;
            PositionalEncoding<T> position_embedding_table;
            std::vector<Block<T>> decoder_blocks;
            LayerNorm<T> ln_f; 
            Linear<T> lm_head; 
    
        public:

        GPT(size_t vocab_size, size_t input_dim, size_t block_size, 
            size_t n_heads, size_t n_layer):

            embedding_table(vocab_size, input_dim),
            position_embedding_table(input_dim, block_size),

            ln_f({input_dim}),
            lm_head(input_dim, vocab_size)
        {
            this->register_module(&embedding_table);
            this->register_module(&ln_f);
            this->register_module(&lm_head);

            if (this->max_sequence_length == 0)
                max_sequence_length = block_size;

            for(size_t i = 0; i < n_layer; i++)
                decoder_blocks.emplace_back(input_dim, block_size, n_heads);
            for(auto& b : decoder_blocks)
                this->register_module(&b);
        }

        Tensor_t<T> forward(Tensor_t<T> index, Tensor_t<T> targets, bool apply_mask=true)
        {
            size_t batch_size = index->shape[0], seq_len = index->shape[1]; 
                    
            // Token embeddings
            Tensor_t<T> tok_embed = this->embedding_table.forward(index);  // (batch_size, seq_len, input_dim)
            
            // Positional embeddings
            Tensor_t<T> pos_indices = make_tensor<T>(Matrix<T>::arange(seq_len));
            Tensor_t<T> pos_embed = this->position_embedding_table.forward(pos_indices);  // (seq_len, input_dim)
            
            // Broadcast positional embeddings to match batch size
            if (pos_embed->ndims == 2)  // (seq_len, input_dim)
                pos_embed = make_tensor<T>(Matrix<T>::expand_dims(pos_embed->val, 0));  // (1, seq_len, input_dim)
            
            // Add embeddings
            Tensor_t<T> x_after_embed = tok_embed + pos_embed;
            
            // Through transformer blocks
            Tensor_t<T> x = x_after_embed;

            for(auto& block : this->decoder_blocks)
                x = block.forward(x, apply_mask);
            
            // Final layer norm
            Tensor_t<T> x_after_ln = this->ln_f.forward(x);
            
            // Language modeling head
            Tensor_t<T> output = this->lm_head.forward(x_after_ln);  // (batch_size, seq_len, vocab_size)
            
            // Compute loss if targets provided
            Tensor_t<T> loss;
            if (targets != nullptr){
                // Reshape for loss calculation
                auto logits_flat = output->reshape({batch_size * seq_len, this->vocab_size});  // (batch_size * seq_len, vocab_size)
                auto targets_flat = targets->reshape({batch_size* seq_len});  // (batch_size * seq_len,)
                
                // Filter out padding tokens if any (assuming -1 or similar)
                Tensor_t<T> valid_mask = targets_flat >= 0;
                if (Matrix<T>::any(valid_mask->val))
                {
                    Tensor_t<T> logits_valid = logits_flat->at(valid_mask);
                    Tensor_t<T> targets_valid = targets_flat->at({valid_mask});
                    loss = Tensor<T>::sparse_cross_entropy(targets_valid, logits_valid);
                    return loss;
                }
                else
                {
                    loss = Tensor<T>::sparse_cross_entropy(targets_flat, logits_flat);
                    return loss;
                }
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
                Matrix<T> index_cond = current_index->val.slice_col((current_index->shape[1] - this->max_sequence_length), current_index->shape[1]);
                
                // Get predictions (no mask for generation)
                Tensor_t<T> logits = this->forward(make_tensor<T>(index_cond), nullptr, false);
                
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
        
        return loss;
    }


        void train(std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> get_batch_fn,
                std::function<std::pair<Tensor_t<T>, Tensor_t<T>>(std::string)> eval_get_batch_fn,
                size_t iters, size_t eval_interval)
        {
            Optimizer<T> Op(this->parameters(), 1e-4, ADAMw);

            for(auto iter=0; iter < iters; iter++)
            {
                // Get training batch
                auto [inputs, targets] = get_batch_fn("train");
                
                // Training step
                Tensor_t<T> loss = this->train_step(&Op, inputs, targets);
                
                // Print progress
                if (iter % eval_interval == 0)
                    std::cout << "Iters :"<< iter << "Loss: "<< loss<<"\n";
                
                // Evaluation
                if( eval_get_batch_fn && (iter % eval_interval == 0) && iters > 0)
                    this->eval_step(eval_get_batch_fn);
            }
    }

    // int (*funcPtr)(int, int); 


    void eval_step(std::pair<Tensor_t<T>, Tensor_t<T>> (* get_batch_fn)(std::string )){
        auto [inputs, targets] = get_batch_fn("val");
        auto val_loss = this->forward(inputs, targets, true);
        std::cout <<"Validation Loss"<< val_loss << "\n";
    }

};

#endif

template <typename T>
Tensor_t<T> get_batch_fn(std::string){

}
        
template <typename T>
std::function<std::pair<Tensor_t<T>, Tensor_t<T>>> eval_get_batch_fn(std::string){
    
}   