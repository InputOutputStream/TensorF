#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>
#include "GGUFLoader.hpp"
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../Modules/Transformer/Llama/Llama.hpp"
#include "../DataLoader/GGUF.hpp"
#include "../Tokenizer/LlamaTokenizer.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// LlamaGGUFLoader  (Llama / SmolLM / Mistral family)
// ─────────────────────────────────────────────────────────────────────────────
/*
 * GGUF → Llama weight mapping
 * ─────────────────────────────────────────────────────────────────────────────
 * GGUF stores [out, in] PyTorch weights with fastest dim first, i.e. as [in, out].
 * load_tensor() reverses → Matrix{out, in} = PyTorch shape.
 * New Linear.weight = {out, in}  →  NO .transpose() for any Linear weight.
 *
 * GGUF name                   PyTorch shape      load_raw shape    stored as
 * ─────────────────────────────────────────────────────────────────────────────
 * token_embd.weight           [vocab, D]          {vocab, D}        Embedding ✓
 * output_norm.weight          [D]                 {D}               1-D       ✓
 * output.weight               [vocab, D]          {vocab, D}        lm_head   ✓
 * blk.N.attn_norm.weight      [D]                 {D}               1-D       ✓
 * blk.N.ffn_norm.weight       [D]                 {D}               1-D       ✓
 * blk.N.attn_q.weight         [nH*hs, D]          {nH*hs, D}        slice_row → {hs,D} ✓
 * blk.N.attn_k.weight         [nKV*hs, D]         {nKV*hs, D}       slice_row → {hs,D} ✓
 * blk.N.attn_v.weight         [nKV*hs, D]         {nKV*hs, D}       slice_row → {hs,D} ✓
 * blk.N.attn_output.weight    [D, D]              {D,  D}           proJ      ✓
 * blk.N.ffn_gate.weight       [ffn_h, D]          {ffn_h, D}        gate_proj ✓
 * blk.N.ffn_up.weight         [ffn_h, D]          {ffn_h, D}        up_proj   ✓
 * blk.N.ffn_down.weight       [D, ffn_h]          {D, ffn_h}        down_proj ✓
 */

struct LlamaHyperParams {
    size_t vocab_size;
    size_t input_dim;       // embedding_length    e.g. 576
    size_t block_size;      // context_length      e.g. 8192
    size_t n_heads;         // attention.head_count      e.g. 9
    size_t n_kv_heads;      // attention.head_count_kv   e.g. 3
    size_t n_layer;         // block_count         e.g. 30
    size_t ffn_hidden;      // feed_forward_length e.g. 1536
};

template <typename T, template<typename> class LinearT>
class LlamaGGUFLoader : public GGUFLoader<T> {
    private:
        using Base = GGUFLoader<T>;

    // ── attention head loading (supports GQA) ────────────────────────────────

    void load_attention_heads(MultiHeadAttention<T>& mha,
                              const LlamaHyperParams& hp,
                              const std::string& pfx) {
        const size_t n_heads    = hp.n_heads;
        const size_t n_kv_heads = hp.n_kv_heads;
        const size_t head_size  = hp.input_dim / n_heads;
        const size_t kv_ratio   = n_heads / n_kv_heads;

        // PyTorch [nH*hs, D] → load_raw → {nH*hs, D}.
        // Matrix::slice_row(h*hs, (h+1)*hs) → {hs, D} = {out, in} — no transpose needed.
        Matrix<T> Q_full = this->load_raw(pfx + "attn_q.weight");
        Matrix<T> K_full = this->load_raw(pfx + "attn_k.weight");
        Matrix<T> V_full = this->load_raw(pfx + "attn_v.weight");

        for (size_t h = 0; h < n_heads; h++) {
            Head<T>& head = *mha.Heads[h];
            size_t   kv_h = h / kv_ratio;   // GQA: many Q heads share one KV head

            if (Q_full.get_size() > 0)
                this->copy_into(head.Q.weight,
                    Q_full.slice_row(h * head_size, (h + 1) * head_size),
                    pfx + "attn_q[h=" + std::to_string(h) + "]");

            if (K_full.get_size() > 0)
                this->copy_into(head.K.weight,
                    K_full.slice_row(kv_h * head_size, (kv_h + 1) * head_size),
                    pfx + "attn_k[kv=" + std::to_string(kv_h) + "]");

            if (V_full.get_size() > 0)
                this->copy_into(head.V.weight,
                    V_full.slice_row(kv_h * head_size, (kv_h + 1) * head_size),
                    pfx + "attn_v[kv=" + std::to_string(kv_h) + "]");
        }
    }

    // ── block loading ────────────────────────────────────────────────────────

    void load_block(Llama<T, Linear>& model, const LlamaHyperParams& hp, size_t blk_idx) {
        Block<T>&   block = *model.decoder_blocks[blk_idx];
        std::string pfx   = "blk." + std::to_string(blk_idx) + ".";

        // ── RMSNorms (1-D) ───────────────────────────────────────────────────
        this->copy_into(block.rms1.gamma,
                        this->load_raw(pfx + "attn_norm.weight"),
                        pfx + "attn_norm.weight");
        this->copy_into(block.rms2.gamma,
                        this->load_raw(pfx + "ffn_norm.weight"),
                        pfx + "ffn_norm.weight");

        // ── Attention output projection ──────────────────────────────────────
        // PyTorch [D, D] → load_raw → {D, D} = {out, in} ✓
        this->copy_into(block.mha.proJ.weight,
                        this->load_raw(pfx + "attn_output.weight"),
                        pfx + "attn_output.weight");

        // ── Q / K / V (with GQA support) ────────────────────────────────────
        load_attention_heads(block.mha, hp, pfx);

        // ── SwiGLU FFN ───────────────────────────────────────────────────────
        // gate_proj: PyTorch [ffn_h, D] → load_raw → {ffn_h, D} = {out, in} ✓
        // up_proj:   PyTorch [ffn_h, D] → load_raw → {ffn_h, D} = {out, in} ✓
        // down_proj: PyTorch [D, ffn_h] → load_raw → {D, ffn_h} = {out, in} ✓
        this->copy_into(block.ffwd.gate_proj.weight,
                        this->load_raw(pfx + "ffn_gate.weight"),
                        pfx + "ffn_gate.weight");
        this->copy_into(block.ffwd.up_proj.weight,
                        this->load_raw(pfx + "ffn_up.weight"),
                        pfx + "ffn_up.weight");
        this->copy_into(block.ffwd.down_proj.weight,
                        this->load_raw(pfx + "ffn_down.weight"),
                        pfx + "ffn_down.weight");
    }

    // ── full weight loading ───────────────────────────────────────────────────

    void load_weights(Llama<T, Linear>& model, const LlamaHyperParams& hp) {

        // Token embedding: PyTorch [vocab, D] → load_raw → {vocab, D} ✓
        this->copy_into(model.embedding_table.embeddings,
                        this->load_raw("token_embd.weight"),
                        "token_embd.weight");

        // Final RMSNorm (1-D)
        this->copy_into(model.rms.gamma,
                        this->load_raw("output_norm.weight"),
                        "output_norm.weight");

        // LM head: PyTorch [vocab, D] → load_raw → {vocab, D} = {out, in} ✓
        // Tied-weight fallback: token_embd → {vocab, D} = {out, in} ✓
        {
            Matrix<T> w = this->load_raw("output.weight", false);
            if (w.get_size() > 0) {
                this->copy_into(model.lm_head.weight, w, "output.weight");
            } else {
                Matrix<T> embd = this->load_raw("token_embd.weight");
                if (embd.get_size() > 0)
                    this->copy_into(model.lm_head.weight, embd, "output.weight (tied)");
            }
        }

        for (size_t blk = 0; blk < hp.n_layer; blk++)
            load_block(model, hp, blk);
    }

public:
    Llama<T, Linear> load_model(const std::string& path, const LlamaHyperParams& hp) {
        this->open(path);
        std::cout << "[LlamaGGUFLoader] Building Llama skeleton"
                  << " layers=" << hp.n_layer
                  << " d="      << hp.input_dim
                  << " heads="  << hp.n_heads
                  << " kv_h="   << hp.n_kv_heads << "\n";
        Llama<T, Linear> model(hp.vocab_size, hp.input_dim, hp.block_size,
                       hp.n_heads, hp.n_layer, hp.ffn_hidden);
        load_weights(model, hp);
        this->report_params(model);
        std::cout << "[LlamaGGUFLoader] Done.\n";
        return model;
    }

    LlamaTokenizer load_tokenizer() {
        auto vocab  = this->get_metadata_array("tokenizer.ggml.tokens");
        auto merges = this->get_metadata_array("tokenizer.ggml.merges");
        if (vocab.empty() || merges.empty()) {
            std::cerr << "[LlamaGGUFLoader] Warning: tokenizer metadata not found, "
                         "returning empty tokenizer.\n";
            return LlamaTokenizer();
        }
        std::vector<std::pair<std::string, std::string>> merge_pairs;
        for (const auto& m : merges) {
            size_t space = m.find(' ');
            if (space != std::string::npos)
                merge_pairs.emplace_back(m.substr(0, space), m.substr(space + 1));
        }
        LlamaTokenizer tokenizer;
        tokenizer.load_from_gguf(vocab, merge_pairs);
        return tokenizer;
    }
};