#ifndef __GPT2_TOKENIZER__HPP
#define __GPT2_TOKENIZER__HPP

#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <regex>
#include <nlohmann/json.hpp> 

class GPT2Tokenizer {
public:
    std::unordered_map<std::string, int> encoder;   // token_str -> id
    std::unordered_map<int, std::string> decoder;   // id -> token_str
    std::map<std::pair<std::string,std::string>, int> bpe_ranks; // merge pair -> rank
    std::unordered_map<uint8_t, char32_t> byte_encoder;
    std::unordered_map<char32_t, uint8_t> byte_decoder;

    void build_byte_encoder() {
        byte_encoder.clear();
        byte_decoder.clear();

        std::vector<int> bs;
        for (int b = '!'; b <= '~'; b++) bs.push_back(b);   // 33–126
        for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b); // 161–172
        for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b); // 174–255

        // cs starts as a copy, then remapped bytes get codepoints 256+
        std::vector<int> cs(bs);
        int n = 256;
        for (int b = 0; b < 256; b++) {
            bool in_bs = std::find(bs.begin(), bs.end(), b) != bs.end();
            if (!in_bs) {
                bs.push_back(b);
                cs.push_back(n++);  // e.g. space(32) → 288, tab(9) → ..., etc.
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            byte_encoder[(uint8_t)bs[i]] = (char32_t)cs[i];
            byte_decoder[(char32_t)cs[i]] = (uint8_t)bs[i];
        }
    }
    void load(const std::string& vocab_path, const std::string& merges_path) {
        build_byte_encoder();

        // Load vocab.json
        std::ifstream vf(vocab_path);
        nlohmann::json vocab_json;
        vf >> vocab_json;
        for (auto& [k, v] : vocab_json.items()) {
            encoder[k] = v.get<int>();
            decoder[v.get<int>()] = k;
        }

        // Load merges.txt
        std::ifstream mf(merges_path);
        std::string line;
        std::getline(mf, line); // skip header "#version..."
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b;
            bpe_ranks[{a, b}] = rank++;
        }
    }

    // Apply BPE to a single word (already unicode-encoded)
    std::vector<std::string> bpe(const std::string& token) {
        std::vector<std::string> word;

        // Split into UTF-8 characters (not bytes)
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = (unsigned char)token[i];
            size_t char_len = 1;
            if      ((c & 0x80) == 0x00) char_len = 1;
            else if ((c & 0xE0) == 0xC0) char_len = 2;
            else if ((c & 0xF0) == 0xE0) char_len = 3;
            else if ((c & 0xF8) == 0xF0) char_len = 4;
            word.push_back(token.substr(i, char_len));
            i += char_len;
        }

        while (word.size() > 1) {
            int best_rank = INT_MAX;
            int best_i = -1;
            for (size_t j = 0; j + 1 < word.size(); j++) {
                auto it = bpe_ranks.find({word[j], word[j+1]});
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = (int)j;
                }
            }
            if (best_i == -1) break;

            std::string merged = word[best_i] + word[best_i+1];
            word.erase(word.begin() + best_i, word.begin() + best_i + 2);
            word.insert(word.begin() + best_i, merged);
        }
        return word;
    }

    std::vector<int> encode(const std::string& text) {
        // GPT-2 regex pattern for pre-tokenization
        std::regex pat(R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+)");
        std::vector<int> ids;

        auto begin = std::sregex_iterator(text.begin(), text.end(), pat);
        auto end   = std::sregex_iterator();

        for (auto it = begin; it != end; ++it) {
            std::string word = it->str();

            // Encode each byte through bytes_to_unicode
            std::string encoded;
            for (unsigned char c : word) {
                char32_t uc = byte_encoder[c];
                // encode uc back to UTF-8 for map lookup
                if (uc < 128) encoded += (char)uc;
                else {
                    encoded += (char)(0xC0 | (uc >> 6));
                    encoded += (char)(0x80 | (uc & 0x3F));
                }
            }

            // Apply BPE
            auto pieces = bpe(encoded);
            for (auto& piece : pieces) {
                auto it2 = encoder.find(piece);
                if (it2 != encoder.end())
                    ids.push_back(it2->second);
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) {
        std::string text;
        for (int id : ids) {
            auto it = decoder.find(id);
            if (it == decoder.end()) continue;

            const std::string& token_str = it->second;

            // Walk UTF-8 bytes of the token string, decode codepoints,
            // then map each codepoint back through byte_decoder
            size_t i = 0;
            while (i < token_str.size()) {
                unsigned char c = (unsigned char)token_str[i];
                char32_t codepoint;

                if (c < 0x80) {                        // 1-byte ASCII
                    codepoint = c; i += 1;
                } else if ((c & 0xE0) == 0xC0) {       // 2-byte sequence
                    codepoint = (c & 0x1F) << 6;
                    codepoint |= ((unsigned char)token_str[i+1] & 0x3F);
                    i += 2;
                } else if ((c & 0xF0) == 0xE0) {       // 3-byte sequence
                    codepoint = (c & 0x0F) << 12;
                    codepoint |= ((unsigned char)token_str[i+1] & 0x3F) << 6;
                    codepoint |= ((unsigned char)token_str[i+2] & 0x3F);
                    i += 3;
                } else {
                    i++; continue; // skip malformed
                }

                auto bd = byte_decoder.find(codepoint);
                if (bd != byte_decoder.end())
                    text += (char)bd->second;   // maps Ġ(U+0120) → ' ', Ċ(U+010A) → '\n'
                // else: skip unmappable codepoints
            }
        }
        return text;
    }
};


#endif