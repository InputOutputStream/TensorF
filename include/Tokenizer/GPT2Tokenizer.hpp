#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// GPT2Tokenizer.hpp — byte-level BPE tokenizer for GPT-2 family models
// ─────────────────────────────────────────────────────────────────────────────
/*
 * Shared vocabulary storage, byte<->unicode mapping, the BPE merge algorithm,
 * and the pre-tokenization regex all live in Tokenizer (see Tokenizer.hpp).
 * This class only adds:
 *   - load()              — read vocab.json + merges.txt from disk.
 *   - load_from_arrays()  — build from in-memory vocab/merges (e.g. GGUF).
 *   - encode() / decode() — GPT-2's specific byte-fallback handling.
 */

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

#include "Tokenizer.hpp"

class GPT2Tokenizer : public Tokenizer {
public:

    // Build from in-memory arrays (e.g. pulled from GGUF metadata).
    void load_from_arrays(const std::vector<std::string>& vocab,
                           const std::vector<std::string>& merges) {
        build_byte_encoder();
        set_vocab(vocab);
        set_merges(merges);   // "left right" string format
    }

    // Build from on-disk Hugging Face-style vocab.json + merges.txt.
    void load(const std::string& vocab_path, const std::string& merges_path) {
        build_byte_encoder();

        // Load vocab.json
        std::ifstream vf(vocab_path);
        nlohmann::json vocab_json;
        vf >> vocab_json;
        encoder.clear();
        decoder.clear();
        for (auto& [k, v] : vocab_json.items()) {
            encoder[k] = v.get<int>();
            decoder[v.get<int>()] = k;
        }

        // Load merges.txt
        std::ifstream mf(merges_path);
        std::string line;
        std::getline(mf, line); // skip header "#version..."
        bpe_ranks.clear();
        int rank = 0;
        while (std::getline(mf, line)) {
            if (line.empty()) continue;
            std::istringstream iss(line);
            std::string a, b;
            iss >> a >> b;
            bpe_ranks[{a, b}] = rank++;
        }
    }

    std::vector<int> encode(const std::string& text) override {
        std::vector<int> ids;

        for (const auto& word : pre_tokenize(text)) {
            // Map raw bytes through byte_encoder, re-encode as UTF-8
            std::string encoded;
            for (unsigned char c : word) {
                char32_t uc = byte_encoder[c];
                encoded += codepoint_to_utf8(uc);
            }

            // Apply BPE, then look each piece up in the vocab
            for (const auto& piece : bpe(encoded)) {
                auto it = encoder.find(piece);
                if (it != encoder.end()) {
                    ids.push_back(it->second);
                } else {
                    // NOTE: previously this branch silently dropped the
                    // piece with no warning at all. If your prompt is
                    // encoding to far fewer ids than expected, this is
                    // where to look first.
                    std::cerr << "[GPT2Tokenizer] Unknown token: \"" << piece << "\"\n";
                }
            }
        }
        return ids;
    }

    std::string decode(const std::vector<int>& ids) override {
        std::string text;
        for (int id : ids) {
            auto it = decoder.find(id);
            if (it == decoder.end()) continue;

            for_each_codepoint(it->second, [&](char32_t cp) {
                auto bd = byte_decoder.find(cp);
                if (bd != byte_decoder.end())
                    text += (char)bd->second;   // Ġ(U+0120) → ' ', Ċ(U+010A) → '\n'
                // else: skip unmappable codepoints
            });
        }
        return text;
    }
};