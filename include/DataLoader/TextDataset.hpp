#ifndef __TEXT_DATASET_HPP__
#define __TEXT_DATASET_HPP__

#include <string>
#include <vector>
#include <map>
#include <set>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <filesystem>
#include <stdexcept>

#include "../Types/types.hpp"

namespace fs = std::filesystem;

/**
 * TextDataset
 *
 * Loads text files from a directory, splits by character count (90/10),
 * builds a character-level vocabulary, encodes text to token ids,
 * and provides random batch sampling for language model training.
 *
 * Usage:
 *   TextDataset<float> ds("Dataset", 16, 8);
 *   ds.load();
 *   auto [x, y] = ds.get_batch("train");
 */

template<typename T>
class TextDataset {

private:
    std::string         folder_path;
    std::string         train_path;
    std::string         val_path;
    std::string         vocab_path;

    size_t              block_size;
    size_t              batch_size;

    std::vector<char>   vocab_list;
    std::map<char, int> stoi;
    std::map<int, char> itos;

    std::vector<int>    train_data;
    std::vector<int>    val_data;

    std::mt19937        rng;

    // ── helpers ──────────────────────────────────────────────────────────

    std::vector<std::string> scan_txt_files() {
        std::vector<std::string> files;
        for (const auto& entry : fs::directory_iterator(folder_path))
            if (entry.is_regular_file() && entry.path().extension() == ".txt")
                files.push_back(entry.path().filename().string());
        return files;
    }

    std::string read_file(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) throw std::runtime_error("Cannot open: " + path);
        return std::string((std::istreambuf_iterator<char>(f)), {});
    }

    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char c : text) {
            auto it = stoi.find(c);
            if (it != stoi.end()) ids.push_back(it->second);
        }
        return ids;
    }

public:

    TextDataset(const std::string& folder,
                size_t block_size,
                size_t batch_size,
                uint32_t seed = 42)
        : folder_path(folder),
          train_path(folder + "/train_split.txt"),
          val_path(folder + "/val_split.txt"),
          vocab_path(folder + "/vocab.txt"),
          block_size(block_size),
          batch_size(batch_size),
          rng(seed)
    {}

    // ── load ─────────────────────────────────────────────────────────────

    void load() {
        auto files = scan_txt_files();
        if (files.empty()) throw std::runtime_error("No .txt files in: " + folder_path);
        std::cout << "Files found: " << files.size() << "\n";

        // Read all content
        std::string train_text, val_text;
        std::set<char> vocab_set;
        size_t total = 0;

        std::vector<std::pair<std::string,std::string>> contents;
        for (auto& f : files) {
            std::string text = read_file(folder_path + "/" + f);
            total += text.size();
            for (char c : text) vocab_set.insert(c);
            contents.push_back({f, std::move(text)});
        }

        // 90/10 split by char count
        size_t train_limit = total * 9 / 10;
        size_t accumulated = 0;
        for (auto& [name, text] : contents) {
            if (accumulated < train_limit) train_text += text;
            else                           val_text   += text;
            accumulated += text.size();
        }

        std::cout << "Train chars: " << train_text.size()
                  << "  Val chars: " << val_text.size() << "\n";

        // Write splits
        std::ofstream(train_path) << train_text;
        std::ofstream(val_path)   << val_text;

        // Build vocab
        vocab_list.assign(vocab_set.begin(), vocab_set.end());
        std::sort(vocab_list.begin(), vocab_list.end());
        for (size_t i = 0; i < vocab_list.size(); i++) {
            stoi[vocab_list[i]] = (int)i;
            itos[(int)i]        = vocab_list[i];
        }

        // Write vocab file
        std::ofstream vf(vocab_path);
        for (char c : vocab_list) vf << c << '\n';
        std::cout << "Vocab size: " << vocab_list.size() << "\n";

        // Encode splits
        train_data = encode(train_text);
        val_data   = encode(val_text);
    }

    // ── batch sampling ───────────────────────────────────────────────────

    std::pair<Tensor_t<T>, Tensor_t<T>> get_batch(const std::string& split) {
        const auto& data = (split == "train") ? train_data : val_data;

        if (data.size() < block_size + 1)
            throw std::runtime_error("Not enough data for a batch in split: " + split);

        std::uniform_int_distribution<size_t> dist(0, data.size() - block_size - 1);

        std::vector<T> x_buf, y_buf;
        x_buf.reserve(batch_size * block_size);
        y_buf.reserve(batch_size * block_size);

        for (size_t b = 0; b < batch_size; b++) {
            size_t start = dist(rng);
            for (size_t t = 0; t < block_size; t++) {
                x_buf.push_back((T)data[start + t]);
                y_buf.push_back((T)data[start + t + 1]);
            }
        }

        return {
            make_tensor<T>(Matrix<T>(x_buf, {batch_size, block_size})),
            make_tensor<T>(Matrix<T>(y_buf, {batch_size, block_size}))
        };
    }

    // ── accessors ────────────────────────────────────────────────────────

    size_t vocab_size()      const { return vocab_list.size(); }
    size_t train_tokens()    const { return train_data.size(); }
    size_t val_tokens()      const { return val_data.size(); }
    char   decode(int id)    const { return itos.at(id); }
    int    encode_char(char c) const { return stoi.at(c); }

    std::string decode_sequence(const std::vector<int>& ids) {
        std::string out;
        for (int id : ids) out += itos.at(id);
        return out;
    }
};

#endif // __TEXT_DATASET_HPP__