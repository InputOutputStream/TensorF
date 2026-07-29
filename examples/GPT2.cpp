#include "core/Types/types.hpp"
#include "core/DataStructures/Matrix.hpp"
#include "core/DataStructures/Tensor.hpp"
#include "nn/Modules/Transformer/GPT/GPT.hpp"
#include "nn/Modules/Linear.hpp"
#include "nn/ModelLoader/GPTLoader.hpp"
#include "data/DataLoader/DataLoading.hpp"
#include "nn/Tokenizer/GPT2Tokenizer.hpp"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <unordered_map>


void run_gpt2_inference() {
    // Hyperparamètres GPT-2 small — DOIVENT correspondre exactement au GGUF
    GPT2HyperParams hp {
        .vocab_size = 50257,
        .d_model    = 768,
        .block_size = 1024,
        .n_layer    = 12,
        .n_head     = 12
    };


    GPTGGUFLoader<float, Linear> loader;
    GPTDense<float> model = loader.load_model("SLM/gpt2-small-f32.gguf", hp);

    GPT2Tokenizer tokenizer;
    tokenizer = loader.load_tokenizer();
    if (tokenizer.encoder.empty()) {
        std::cerr << "Failed to load tokenizer from GGUF.\n";
        return;
    }

    std::string prompt = "The data type is int";
    std::vector<int> token_ids = tokenizer.encode(prompt);

    int i= 0;
    for(auto t: token_ids)  
        std::cout <<"Token :" <<i++ <<": " << t <<"\n";

    // Construire le contexte {1, seq_len}
    std::vector<float> ctx_data(token_ids.begin(), token_ids.end());

    Tensor_t<float> context = make_tensor<float>(
        Matrix<float>(ctx_data, {1, ctx_data.size()})
    );

    // Générer 50 nouveaux tokens
    auto out = model.generate(context, 30, 0.8f, 30);

    // Décoder seulement les tokens générés (après le prompt)
    size_t prompt_len = token_ids.size();
    std::vector<int> generated;
    for (size_t i = prompt_len; i < out->val.data.size(); i++)
        generated.push_back((int)out->val.data[i]);

    std::cout << "Prompt    : " << prompt << "\n";
    std::cout << "Généré    : " << tokenizer.decode(generated) << "\n";
}


int main() {

    // Le texte généré sera en anglais / tokens ASCII.
    run_gpt2_inference();

    return 0;
}