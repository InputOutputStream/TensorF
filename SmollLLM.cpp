#include "Types/types.hpp"

#include "DataStructures/Matrix.hpp"

#include "Modules/Transformer/Llama/Llama.hpp"
#include "ModelLoader/LlamaLoader.hpp"


#include "DataLoader/GGUF.hpp"
#include "DataLoader/DataLoading.hpp"
#include "Tokenizer/LlamaTokenizer.hpp"

#include <iostream>
#include <vector>
#include <cassert>

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <filesystem>
#include <random>


// ─── main ───────────────────────────────────────────────────────────────────

int main()
{
    size_t iters = 50;
    size_t evals = 10;
    const std::string folder_path = "Dataset";

    LlamaHyperParams hp {
         .vocab_size  = 49152,
         .input_dim   = 576,
         .block_size  = 512,    // crop to 512 for memory; SmolLM2 trains at 8192
         .n_heads     = 9,
         .n_kv_heads  = 3,
         .n_layer     = 30,
         .ffn_hidden  = 1536,
     };
     
    TextDataset<float> ds("Dataset", 4, 4);
    ds.load();
    
    size_t vocab_size = ds.vocab_size();

    LlamaGGUFLoader<float> loader;
    Llama<float> model = loader.load_model("SLM/SmolLM2-135M-Instruct-f16.gguf", hp);
    
    // inspect the file first
    // loader.inspect("SmolLM2-135M.gguf");

    auto get_batch     = [&](std::string split) { return ds.get_batch(split); };
    // model.train(get_batch, iters, evals);
    
    LlamaTokenizer tokenizer;

    std::string prompt = "the data type is int";
    std::vector<int> token_ids = tokenizer.encode(prompt);

    // Build context tensor
    std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
    Tensor_t<float> context = make_tensor<float>(
        Matrix<float>(ctx_data, {1, ctx_data.size()})
    );

    // Generate
    auto out = model.generate(context, 50, 0.6f, 40);

    std::vector<int> generated_ids;
    for (size_t i = token_ids.size(); i < out->val.data.size(); i++)
        generated_ids.push_back((int)out->val.data[i]);

    std::cout << "Generated: " << tokenizer.decode(generated_ids) << std::endl;

    return 0;
}