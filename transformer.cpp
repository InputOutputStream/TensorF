#include "Types/types.hpp"

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"

#include "Modules/Linear.hpp"
#include "Modules/FeedForward.hpp"
#include "Modules/Optimizer.hpp"
#include "Modules/Relu.hpp"

#include "Modules/Transformer/GPT.hpp"

#include "DataLoader/GGUF.hpp"
#include "DataLoader/DataLoading.hpp"

#include <iostream>
#include <vector>
#include <cassert>


// ─── helpers ────────────────────────────────────────────────────────────────


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

    size_t n_heads = 2;
    size_t n_layers = 1; 
    size_t input_dim = 4;
    size_t block_size = 2;
    size_t batch_size = 2;
    // size_t max_sequence_length = 100;
    size_t iters = 5000;
    // size_t epochs = 10;
    size_t evals = 1000;
    const std::string folder_path = "Dataset";

    TextDataset<float> ds("Dataset", block_size, batch_size);
    ds.load();

    size_t vocab_size = ds.vocab_size();

    GPT<float> model(vocab_size, input_dim, block_size, n_heads, n_layers);
    model.load("Models/GPT.hge");

    auto get_batch     = [&](std::string split) { return ds.get_batch(split); };
    // model.train(get_batch, iters, evals);
    // model.save("Models/GPT.hge");

    std::string prompt = "the data type is int";
    std::vector<float> enc_in;
    for(auto c: prompt)
    {
        enc_in.push_back(float(ds.encode_char(c)));
    }

    Tensor_t<float> context = make_tensor<float>(Matrix<float>(enc_in, {1, enc_in.size()}));
    Tensor_t<float> res = model.generate(context, 50);

    size_t prompt_len = enc_in.size();
    std::vector<int> dec_out;
    for (size_t i = prompt_len; i < res->val.data.size(); i++)
        dec_out.push_back(int(res->val.data[i]));

    auto output = ds.decode_sequence(dec_out);
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: " << output << "\n";

    return 0;
}