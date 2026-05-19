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

using namespace std;


// ─── helpers ────────────────────────────────────────────────────────────────


#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <set>
#include <filesystem>

namespace fs = std::filesystem;

std::vector<std::string> txt_files_in_dir(const std::string& directory) {
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt")
            files.push_back(entry.path().filename().string());
    }
    return files;
}

void append_files(const std::string& folder,
                  const std::vector<std::string>& files,
                  const std::string& output_path,
                  std::set<char>& vocab)
{
    std::ofstream outfile(output_path, std::ios::out);
    if (!outfile.is_open())
        throw std::runtime_error("Cannot open output file: " + output_path);

    size_t total = files.size();
    for (size_t i = 0; i < total; ++i) {
        std::string file_path = folder + "/" + files[i];
        std::ifstream infile(file_path);
        if (!infile.is_open()) {
            std::cerr << "Warning: cannot open " << file_path << "\n";
            continue;
        }

        std::string text((std::istreambuf_iterator<char>(infile)),
                          std::istreambuf_iterator<char>());

        outfile << text;
        for (char c : text) vocab.insert(c);

        // Progress
        std::cerr << "\r[" << (i + 1) << "/" << total << "] " << files[i] << "   ";
    }
    std::cerr << "\n";
}



// ─── main ───────────────────────────────────────────────────────────────────

int main()
{
    const std::string folder_path      = "Dataset";
    const std::string output_file_train = "Dataset/train_split.txt";
    const std::string output_file_val   = "Dataset/val_split.txt";
    const std::string vocab_file        = "Dataset/vocab.txt";

    auto files = txt_files_in_dir(folder_path);
    std::cout << "Total files found: " << files.size() << "\n";

    // Mirror Python slicing: train=[:100], val=[1:10]
    std::vector<std::string> files_train(
        files.begin(),
        files.begin() + std::min<size_t>(100, files.size()));

    std::vector<std::string> files_val(
        files.begin() + std::min<size_t>(1, files.size()),
        files.begin() + std::min<size_t>(10, files.size()));

    std::set<char> vocab;

    std::cout << "Writing train split...\n";
    append_files(folder_path, files_train, output_file_train, vocab);

    std::cout << "Writing val split...\n";
    append_files(folder_path, files_val, output_file_val, vocab);

    // Write vocab — one char per line
    std::ofstream vfile(vocab_file);
    if (!vfile.is_open())
        throw std::runtime_error("Cannot open vocab file: " + vocab_file);
    for (char c : vocab)
        vfile << c << '\n';

    std::cout << "Done. Vocab size: " << vocab.size() << "\n";


    size_t vocab_size = vocab.size();
    size_t n_heads = 1;
    size_t n_layers = 2; 
    size_t input_dim = 100;
    size_t block_size = 16;
    size_t batch_size = 8;
    size_t max_sequence_length = 100;
    size_t iters = 100;
    size_t epochs = 10;
    size_t evals = 100;

    GPT<float> model(vocab_size, input_dim, block_size, n_heads, n_layers);


    return 0;
}