#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <regex>
#include <climits>
#include <iostream>

class LlamaTokenizer {
public:
    std::unordered_map<std::string, int> encoder;   // token -> id
    std::unordered_map<int, std::string> decoder;   // id -> token
    std::map<std::pair<std::string, std::string>, int> bpe_ranks; // merge pair -> rank

    // Build from GGUF metadata
    void load_from_gguf(const std::vector<std::string>& tokens,
                        const std::vector<std::pair<std::string, std::string>>& merges) {
        // Build encoder/decoder
        for (size_t i = 0; i < tokens.size(); i++) {
            encoder[tokens[i]] = static_cast<int>(i);
            decoder[static_cast<int>(i)] = tokens[i];
        }

        // Build BPE ranks
        int rank = 0;
        for (const auto& m : merges) {
            bpe_ranks[m] = rank++;
        }
    }

    // Pre‑tokenization pattern (Llama uses similar regex as GPT‑2)
    std::vector<std::string> pre_tokenize(const std::string& text) {
        std::regex pattern(R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+)");
        std::vector<std::string> words;
        auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it)
            words.push_back(it->str());
        return words;
    }

    // Apply BPE to a single token (word)
    std::vector<std::string> bpe(const std::string& token) {
        // Split token into characters (UTF‑8 safe)
        std::vector<std::string> parts;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = token[i];
            size_t len = 1;
            if ((c & 0x80) == 0x00) len = 1;
            else if ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            parts.push_back(token.substr(i, len));
            i += len;
        }

        while (parts.size() > 1) {
            int best_rank = INT_MAX;
            int best_idx = -1;
            for (size_t j = 0; j + 1 < parts.size(); j++) {
                auto it = bpe_ranks.find({parts[j], parts[j+1]});
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx = j;
                }
            }
            if (best_idx == -1) break;
            std::string merged = parts[best_idx] + parts[best_idx+1];
            parts.erase(parts.begin() + best_idx, parts.begin() + best_idx + 2);
            parts.insert(parts.begin() + best_idx, merged);
        }
        return parts;
    }

    // Full encoding
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;
        for (const auto& word : pre_tokenize(text)) {
            auto pieces = bpe(word);
            for (const auto& p : pieces) {
                auto it = encoder.find(p);
                if (it != encoder.end())
                    ids.push_back(it->second);
                else
                    std::cerr << "[LlamaTokenizer] Unknown token: " << p << std::endl;
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string result;
        for (int id : ids) {
            auto it = decoder.find(id);
            if (it != decoder.end())
                result += it->second;
        }
        return result;
    }
};