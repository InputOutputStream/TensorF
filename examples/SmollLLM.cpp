#include "core/Types/types.hpp"
#include "core/DataStructures/Matrix.hpp"
#include "nn/Modules/Transformer/Llama/Llama.hpp"
#include "nn/ModelLoader/LlamaLoader.hpp"
#include "nn/Modules/Linear.hpp"
#include "data/DataLoader/GGUF.hpp"
#include "data/DataLoader/DataLoading.hpp"
#include "nn/Tokenizer/LlamaTokenizer.hpp"
#include <iostream>
#include <vector>

int main() {
    LlamaHyperParams hp {
        .vocab_size  = 49152,
        .input_dim   = 576,
        .block_size  = 512,      // adjust if memory allows
        .n_heads     = 9,
        .n_kv_heads  = 3,
        .n_layer     = 30,
        .ffn_hidden  = 1536,
    };

    LlamaGGUFLoader<float, Linear> loader;
    LlamaDense<float> model = loader.load_model("SLM/lSmolLM2-135M-Instruct-f16.gguf", hp);
    LlamaTokenizer tokenizer = loader.load_tokenizer();

    std::string test = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
    auto test_ids = tokenizer.encode(test);
    std::cout << "Encoded IDs: ";
    for (int id : test_ids) std::cout << id << " ";
    std::cout << "\n";
    std::cout << "Decoded back: " << tokenizer.decode(test_ids) << "\n";


    // ── Chat template for SmolLM instruct ──────────────────────────────
    std::string system_msg = "You are a helpful assistant.";
    std::string user_prompt = "the data type is int";
    std::string formatted = 
        "<|im_start|>system\n" + system_msg + "<|im_end|>\n" +
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n" +
        "<|im_start|>assistant\n";

    // Encode with special-token awareness
    std::vector<int> token_ids = tokenizer.encode(formatted);

    std::cout << "Token IDs: ";
    for (int id : token_ids) std::cout << id << " ";
    std::cout << "\n";

    // Build context tensor
    std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
    Tensor_t<float> context = make_tensor<float>(
        Matrix<float>(ctx_data, {1, ctx_data.size()})
    );

    // Generate
    auto out = model.generate(context, 30, 0.6f, 40);

    // Decode only the generated part
    size_t prompt_len = token_ids.size();
    std::vector<int> generated_ids;
    for (size_t i = prompt_len; i < out->val.data.size(); i++)
        generated_ids.push_back((int)out->val.data[i]);

    std::cout << "Generated:\n" << tokenizer.decode(generated_ids) << std::endl;

    return 0;
}