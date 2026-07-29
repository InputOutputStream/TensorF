#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>
#include <algorithm>
#include "Tokenizer.hpp"

class LlamaTokenizer : public Tokenizer {
public:
    void load_from_gguf(
        const std::vector<std::string>& tokens,
        const std::vector<std::pair<std::string, std::string>>& merges)
    {
        build_byte_encoder();
        set_vocab(tokens);
        set_merges(merges);

        // Collect all special tokens: those starting with '<' and ending with '>'
        special_tokens_.clear();
        for (const auto& [tok, id] : encoder) {
            if (tok.size() > 2 && tok.front() == '<' && tok.back() == '>') {
                special_tokens_.insert(tok);
            }
        }
        // Sort by length descending so we match the longest first
        special_tokens_list_.assign(special_tokens_.begin(), special_tokens_.end());
        std::sort(special_tokens_list_.begin(), special_tokens_list_.end(),
                  [](const std::string& a, const std::string& b) {
                      return a.size() > b.size();
                  });
    }

    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids;
        size_t pos = 0;
        const size_t n = text.size();

        while (pos < n) {
            // Try to match a special token at current position
            bool matched = false;
            for (const auto& tok : special_tokens_list_) {
                if (text.compare(pos, tok.size(), tok) == 0) {
                    auto it = encoder.find(tok);
                    if (it != encoder.end())
                        ids.push_back(it->second);
                    else
                        std::cerr << "[LlamaTokenizer] Special token not found: " << tok << "\n";
                    pos += tok.size();
                    matched = true;
                    break;
                }
            }
            if (matched) continue;

            // No special token matched: encode the plain text until the next special
            size_t start = pos;
            while (pos < n) {
                bool special_here = false;
                for (const auto& tok : special_tokens_list_) {
                    if (text.compare(pos, tok.size(), tok) == 0) {
                        special_here = true;
                        break;
                    }
                }
                if (special_here) break;
                pos++;
            }
            if (pos > start) {
                std::string plain = text.substr(start, pos - start);
                auto plain_ids = encode_plain(plain);
                ids.insert(ids.end(), plain_ids.begin(), plain_ids.end());
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::string result;
        for (int id : ids) {
            auto it = decoder.find(id);
            if (it == decoder.end()) continue;
            for_each_codepoint(it->second, [&](char32_t cp) {
                if (cp == 0x2581) { result += ' ';  return; }  // ▁ LLaMA space
                if (cp == 0x0120) { result += ' ';  return; }  // Ġ GPT‑2 space
                if (cp == 0x010A) { result += '\n'; return; }  // Ċ newline
                auto bd = byte_decoder.find(cp);
                if (bd != byte_decoder.end())
                    result += (char)bd->second;
            });
        }
        return result;
    }

private:
    std::unordered_set<std::string> special_tokens_;
    std::vector<std::string> special_tokens_list_;  // sorted by length desc

    // Standard byte‑level BPE encoding for non‑special text
    std::vector<int> encode_plain(const std::string& text) {
        std::vector<int> ids;
        for (const auto& word : pre_tokenize(text)) {
            std::string encoded;
            for (unsigned char c : word) {
                char32_t uc = byte_encoder[c];
                encoded += codepoint_to_utf8(uc);
            }
            for (const auto& p : bpe(encoded)) {
                auto it = encoder.find(p);
                if (it != encoder.end())
                    ids.push_back(it->second);
                else
                    std::cerr << "[LlamaTokenizer] Unknown token: \"" << p << "\"\n";
            }
        }
        return ids;
    }
};