#pragma once
// ─────────────────────────────────────────────────────────────────────────────
// Tokenizer.hpp — abstract base class for byte-level BPE tokenizers
// ─────────────────────────────────────────────────────────────────────────────
/*
 * Every GPT-2-style byte-level BPE tokenizer (GPT-2 itself, LLaMA/SmolLM/
 * Mistral, and anything else you add later) shares the same vocabulary
 * storage, merge-rank table, byte<->unicode mapping, BPE merge algorithm,
 * and GPT-2 pre-tokenization regex. Only encode()/decode() differ — mainly
 * in how each model's space/newline markers (Ġ vs ▁ vs raw bytes) get
 * handled — so those are left as pure virtual for subclasses to implement.
 *
 * To add a new tokenizer family:
 *   1. Inherit from Tokenizer.
 *   2. Implement encode() and decode().
 *   3. Add whatever load_from_xxx() you need, calling build_byte_encoder(),
 *      set_vocab(), and set_merges() from this base class.
 */

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <regex>
#include <climits>
#include <cstdint>
#include <algorithm>

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // ── Vocabulary ───────────────────────────────────────────────────────────
    std::unordered_map<std::string, int> encoder;   // token_str -> id
    std::unordered_map<int, std::string> decoder;   // id -> token_str

    // ── BPE merge table ──────────────────────────────────────────────────────
    std::map<std::pair<std::string, std::string>, int> bpe_ranks;  // merge pair -> rank

    // ── Byte ↔ Unicode mapping (GPT-2 / HF byte-level scheme) ────────────────
    std::unordered_map<uint8_t, char32_t> byte_encoder;
    std::unordered_map<char32_t, uint8_t> byte_decoder;

    // ── Interface every concrete tokenizer must provide ──────────────────────
    virtual std::vector<int>  encode(const std::string& text) = 0;
    virtual std::string       decode(const std::vector<int>& ids) = 0;

    // ─────────────────────────────────────────────────────────────────────────
    //  bytes_to_unicode (GPT-2 / HF standard)
    //  Visible ASCII + two Latin-1 ranges are kept as-is; everything else
    //  gets remapped to codepoints starting at 256 (e.g. space -> Ġ U+0120).
    //  Identical for GPT-2 and every LLaMA-family model.
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
                cs.push_back(n++);
            }
        }

        for (size_t i = 0; i < bs.size(); i++) {
            byte_encoder[(uint8_t)bs[i]] = (char32_t)cs[i];
            byte_decoder[(char32_t)cs[i]] = (uint8_t)bs[i];
        }
    }

    // ── Encode one char32_t codepoint to a UTF-8 std::string ────────────────
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

    // ── Walk a UTF-8 string and call cb(codepoint) for each codepoint ───────
    template<typename Fn>
    static void for_each_codepoint(const std::string& s, Fn&& cb) {
        size_t i = 0;
        while (i < s.size()) {
            unsigned char c = (unsigned char)s[i];
            char32_t cp;
            size_t len;
            if      ((c & 0x80) == 0x00) { cp = c;        len = 1; }
            else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
            else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
            else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
            else { i++; continue; }  // skip invalid leading byte

            for (size_t k = 1; k < len && i + k < s.size(); k++)
                cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);

            cb(cp);
            i += len;
        }
    }

    // ── Split a token's raw bytes into UTF-8 characters ──────────────────────
    static std::vector<std::string> split_utf8_chars(const std::string& token) {
        std::vector<std::string> chars;
        size_t i = 0;
        while (i < token.size()) {
            unsigned char c = (unsigned char)token[i];
            size_t len = 1;
            if      ((c & 0xE0) == 0xC0) len = 2;
            else if ((c & 0xF0) == 0xE0) len = 3;
            else if ((c & 0xF8) == 0xF0) len = 4;
            chars.push_back(token.substr(i, len));
            i += len;
        }
        return chars;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  BPE merge loop — identical algorithm for GPT-2 and LLaMA-family models.
    //  Repeatedly merges the single lowest-rank adjacent pair until none remain.
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::string> bpe(const std::string& token) {
        std::vector<std::string> word = split_utf8_chars(token);

        while (word.size() > 1) {
            int best_rank = INT_MAX;
            int best_i = -1;
            for (size_t j = 0; j + 1 < word.size(); j++) {
                auto it = bpe_ranks.find({word[j], word[j + 1]});
                if (it != bpe_ranks.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_i = (int)j;
                }
            }
            if (best_i == -1) break;

            std::string merged = word[best_i] + word[best_i + 1];
            word.erase(word.begin() + best_i, word.begin() + best_i + 2);
            word.insert(word.begin() + best_i, merged);
        }
        return word;
    }

    // ─────────────────────────────────────────────────────────────────────────
    //  GPT-2 pre-tokenization regex — shared by GPT-2 and LLaMA-family models.
    //  Handles contractions, words, numbers, punctuation runs, and whitespace.
    // ─────────────────────────────────────────────────────────────────────────
    std::vector<std::string> pre_tokenize(const std::string& text) const {
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
    //  Vocabulary / merge loading helpers (format-agnostic — subclasses call
    //  these from whatever load_from_xxx() they expose).
    // ─────────────────────────────────────────────────────────────────────────
    void set_vocab(const std::vector<std::string>& tokens) {
        encoder.clear();
        decoder.clear();
        for (size_t i = 0; i < tokens.size(); i++) {
            encoder[tokens[i]] = (int)i;
            decoder[(int)i] = tokens[i];
        }
    }

    // Merges already split into (left, right) pairs.
    void set_merges(const std::vector<std::pair<std::string, std::string>>& merges) {
        bpe_ranks.clear();
        int rank = 0;
        for (const auto& m : merges)
            bpe_ranks[m] = rank++;
    }

    // Merges stored as raw "left right" strings (GGUF / merges.txt format).
    void set_merges(const std::vector<std::string>& merge_lines) {
        bpe_ranks.clear();
        int rank = 0;
        for (const auto& m : merge_lines) {
            size_t space = m.find(' ');
            if (space != std::string::npos) {
                std::string left  = m.substr(0, space);
                std::string right = m.substr(space + 1);
                bpe_ranks[{left, right}] = rank++;
            }
        }
    }

    // ── Common accessors ──────────────────────────────────────────────────────
    size_t vocab_size() const { return encoder.size(); }

    // Exact-string lookup for special tokens (e.g. <|endoftext|>), bypassing BPE.
    int encode_special(const std::string& tok) const {
        auto it = encoder.find(tok);
        return (it != encoder.end()) ? it->second : -1;
    }
};