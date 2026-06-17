#pragma once
/*
 * LlamaTokenizer.hpp — BPE tokenizer compatible with SmolLM2 / LLaMA-family models
 *   1. build_byte_encoder() — identical to GPT-2's bytes_to_unicode.
 *   2. encode() — map raw UTF-8 bytes through byte_encoder before BPE, exactly
 *      as GPT-2 does.
 *   3. decode() — walk each token string as UTF-8 codepoints, reverse through
 *      byte_decoder to recover the original raw bytes, then:
 *        • Replace ▁ (U+2581) with a space (LLaMA space marker).
 *        • Replace Ġ (U+0120) with a space (GPT-2 space marker, present in some
 *          GGUF exports).
 *   4. load_from_gguf() — unchanged public API; merges list is built from the
 *      GGUF tokenizer.merges metadata that LlamaGGUFLoader passes in.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <regex>
#include <climits>
#include <iostream>
#include <algorithm>
#include <cassert>

class LlamaTokenizer {
public:
    // ── Vocabulary ────────────────────────────────────────────────────────────
    std::unordered_map<std::string, int>  encoder;    // unicode-escaped token → id
    std::unordered_map<int, std::string>  decoder;    // id → unicode-escaped token

    // ── BPE merge table ───────────────────────────────────────────────────────
    std::map<std::pair<std::string,std::string>, int> bpe_ranks;

    // ── Byte ↔ Unicode mapping (same scheme as GPT-2) ────────────────────────
    std::unordered_map<uint8_t, char32_t> byte_encoder;   // raw byte → char32_t
    std::unordered_map<char32_t, uint8_t> byte_decoder;   // char32_t → raw byte

    // ─────────────────────────────────────────────────────────────────────────
    //  bytes_to_unicode (GPT-2 / HF standard)
    //  Visible ASCII + two Latin-1 ranges are kept as-is.
    //  Everything else gets remapped to codepoints starting at 256.
    // ─────────────────────────────────────────────────────────────────────────
    void build_byte_encoder() {
        byte_encoder.clear();
        byte_decoder.clear();

        std::vector<int> bs;
        for (int b = '!';  b <= '~';  b++) bs.push_back(b);   // 33–126
        for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);   // 161–172
        for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);   // 174–255

        std::vector<int> cs(bs);   // starts as copy, remapped bytes extend it
        int n = 256;
        for (int b = 0; b < 256; b++) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) {
                bs.push_back(b);
                cs.push_back(n++);   // e.g. space(32) → U+0120 (Ġ)
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            byte_encoder[(uint8_t)bs[i]] = (char32_t)cs[i];
            byte_decoder[(char32_t)cs[i]] = (uint8_t)bs[i];
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Encode one char32_t codepoint to a UTF-8 std::string
    // ─────────────────────────────────────────────────────────────────────────
    static std::string codepoint_to_utf8(char32_t cp) {
        std::string s;
        if (cp < 0x80) {
            s += (char)cp;
        } else if (cp < 0x800) {
            s += (char)(0xC0 | (cp >> 6));
            s += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            s += (char)(0xE0 | (cp >> 12));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        } else {
            s += (char)(0xF0 | (cp >> 18));
            s += (char)(0x80 | ((cp >> 12) & 0x3F));
            s += (char)(0x80 | ((cp >> 6) & 0x3F));
            s += (char)(0x80 | (cp & 0x3F));
        }
        return s;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Walk a UTF-8 string and call cb(codepoint) for each codepoint.
    // ─────────────────────────────────────────────────────────────────────────
    template<typename Fn>
    static void for_each_codepoint(const std::string& s, Fn&& cb) {
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            char32_t cp;
            size_t len;
            if      ((c & 0x80) == 0x00) { cp = c;                                  len = 1; }
            else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F;                           len = 2; }
            else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F;                           len = 3; }
            else if ((c & 0xF8) == 0xF0) { cp = c & 0x07;                           len = 4; }
            else { i++; continue; }  // skip invalid leading byte

            for (size_t k = 1; k < len && i+k < s.size(); k++)
                cp = (cp << 6) | ((unsigned char)s[i+k] & 0x3F);

            cb(cp);
            i += len;
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Build from GGUF metadata (called by LlamaGGUFLoader)
    // ─────────────────────────────────────────────────────────────────────────
    void load_from_gguf(
        const std::vector<std::string>& tokens,
        const std::vector<std::pair<std::string,std::string>>& merges)
    {
        build_byte_encoder();

        encoder.clear();
        decoder.clear();
        bpe_ranks.clear();

        for (size_t i = 0; i < tokens.size(); i++) {
            encoder[tokens[i]] = (int)i;
            decoder[(int)i]    = tokens[i];
        }

        int rank = 0;
        for (const auto& m : merges)
            bpe_ranks[m] = rank++;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  BPE — identical algorithm to GPT-2's implementation
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::string> bpe(const std::string& token) {
        // Split into UTF-8 characters
        std::vector<std::string> parts;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = (unsigned char)token[i];
            size_t len = 1;
            if      ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            parts.push_back(token.substr(i, len));
            i += len;
        }

        while (parts.size() > 1) {
            int  best_rank = INT_MAX;
            int  best_idx  = -1;
            for (size_t j = 0; j + 1 < parts.size(); j++) {
                auto it = bpe_ranks.find({parts[j], parts[j+1]});
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_idx  = (int)j;
                }
            }
            if (best_idx == -1) break;

            std::string merged = parts[best_idx] + parts[best_idx+1];
            parts.erase(parts.begin() + best_idx, parts.begin() + best_idx + 2);
            parts.insert(parts.begin() + best_idx, merged);
        }
        return parts;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Pre-tokenisation
    //  LLaMA uses the same GPT-2 regex pattern for word splitting.
    //  Leading spaces become ▁ (U+2581) in the GGUF vocab — we handle that
    //  in encode() by mapping raw bytes through byte_encoder first.
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::string> pre_tokenize(const std::string& text) {
        // Same pattern as GPT-2; handles contractions, words, numbers, punct, spaces
        static const std::regex pattern(
            R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+)");
        std::vector<std::string> words;
        auto begin = std::sregex_iterator(text.begin(), text.end(), pattern);
        auto end   = std::sregex_iterator();
        for (auto it = begin; it != end; ++it)
            words.push_back(it->str());
        return words;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  encode
    //  For each pre-tokenised word:
    //    1. Map every raw byte through byte_encoder (same as GPT-2).
    //    2. Encode the resulting char32_t sequence back to UTF-8.
    //    3. Run BPE.
    //    4. Look up each BPE piece in the vocab.
    //
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<int> encode(const std::string& text) {
        std::vector<int> ids;

        for (const auto& word : pre_tokenize(text)) {
            // Step 1+2: raw bytes → unicode-escaped UTF-8
            std::string encoded;
            for (unsigned char c : word) {
                char32_t uc = byte_encoder[c];   // always populated after build_byte_encoder
                encoded += codepoint_to_utf8(uc);
            }

            // Step 3: BPE
            auto pieces = bpe(encoded);

            // Step 4: vocab lookup
            for (const auto& p : pieces) {
                auto it = encoder.find(p);
                if (it != encoder.end()) {
                    ids.push_back(it->second);
                } else {
                    std::cerr << "[LlamaTokenizer] Unknown token: \"" << p << "\"\n";
                }
            }
        }
        return ids;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  decode
    //  For each token id:
    //    1. Look up the unicode-escaped string in decoder.
    //    2. Walk it codepoint by codepoint.
    //    3. If the codepoint is ▁ (U+2581, LLaMA space) or Ġ (U+0120, GPT-2
    //       space) → emit a space directly.
    //    4. Otherwise reverse through byte_decoder to get the raw byte.
    //
    // ─────────────────────────────────────────────────────────────────────────
    std::string decode(const std::vector<int>& ids) {
        std::string result;

        for (int id : ids) {
            auto it = decoder.find(id);
            if (it == decoder.end()) continue;

            const std::string& tok = it->second;

            for_each_codepoint(tok, [&](char32_t cp) {
                // LLaMA space marker (▁ U+2581)
                if (cp == 0x2581) {
                    result += ' ';
                    return;
                }
                // GPT-2 space marker (Ġ U+0120) — present in some GGUF exports
                if (cp == 0x0120) {
                    result += ' ';
                    return;
                }
                // Newline marker (Ċ U+010A) used by some models
                if (cp == 0x010A) {
                    result += '\n';
                    return;
                }

                // General case: reverse byte_decoder
                auto bd = byte_decoder.find(cp);
                if (bd != byte_decoder.end()) {
                    result += (char)bd->second;
                }
                // else: skip unmappable codepoints (should not happen with well-formed vocab)
            });
        }

        return result;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  Convenience: encode a special token like <|endoftext|> by exact string
    //  match in the vocab, bypassing BPE entirely.
    // ─────────────────────────────────────────────────────────────────────────
    int encode_special(const std::string& tok) const {
        auto it = encoder.find(tok);
        return (it != encoder.end()) ? it->second : -1;
    }
};