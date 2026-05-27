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
    size_t iters = 100;
    // size_t epochs = 10;
    size_t evals = 10;
    const std::string folder_path = "Dataset";

    TextDataset<float> ds("Dataset", block_size, batch_size);
    ds.load();

    size_t vocab_size = ds.vocab_size();

    GPT<float> model(vocab_size, input_dim, block_size, n_heads, n_layers);

    auto get_batch     = [&](std::string split) { return ds.get_batch(split); };
    model.train(get_batch, iters, evals);
   
    return 0;
}