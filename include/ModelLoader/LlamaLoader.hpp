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


// ─────────────────────────────────────────────────────────────────────────────
// LlamaGGUFLoader  (Llama / SmolLM / Mistral family)
// ─────────────────────────────────────────────────────────────────────────────
/*
 * GGUF → Llama mapping
 * ────────────────────────────────────────────────────────────────────────────
 * GGUF tensor name                   Llama<T> field             Notes
 * ──────────────────────────────────────────────────────────────────────────────
 * token_embd.weight   [vocab,D]       embedding_table.embeddings  (no transpose)
 * output_norm.weight  [D]             rms.gamma
 * output.weight       [D,vocab]       lm_head.weight              (→[D,vocab]T)
 * blk.N.attn_norm.weight [D]          blocks[N].rms1.gamma
 * blk.N.ffn_norm.weight  [D]          blocks[N].rms2.gamma
 * blk.N.attn_q.weight    [nH*hs,D]   heads[h].Q.weight           (→[D,hs] per h)
 * blk.N.attn_k.weight    [nKV*hs,D]  heads[h].K.weight           (GQA expansion)
 * blk.N.attn_v.weight    [nKV*hs,D]  heads[h].V.weight           (GQA expansion)
 * blk.N.attn_output.weight [D,D]      mha.proJ.weight             (→[D,D])
 * blk.N.ffn_gate.weight  [ffn_h,D]   ffwd.gate_proj.weight       (→[D,ffn_h])
 * blk.N.ffn_up.weight    [ffn_h,D]   ffwd.up_proj.weight         (→[D,ffn_h])
 * blk.N.ffn_down.weight  [D,ffn_h]   ffwd.down_proj.weight       (→[ffn_h,D])
 */


struct LlamaHyperParams {
    size_t vocab_size;
    size_t input_dim;       // embedding_length   e.g. 576
    size_t block_size;      // context_length     e.g. 8192
    size_t n_heads;         // attention.head_count        e.g. 9
    size_t n_kv_heads;      // attention.head_count_kv     e.g. 3
    size_t n_layer;         // block_count        e.g. 30
    size_t ffn_hidden;      // feed_forward_length e.g. 1536
};

template<typename T>
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

        // GGUF [n_heads*hs, D] → transpose → [D, n_heads*hs]
        Matrix<T> Q_full = this->load_raw_t(pfx + "attn_q.weight");
        Matrix<T> K_full = this->load_raw_t(pfx + "attn_k.weight");
        Matrix<T> V_full = this->load_raw_t(pfx + "attn_v.weight");

        for (size_t h = 0; h < n_heads; h++) {
            Head<T>& head  = *mha.Heads[h];
            size_t   kv_h  = h / kv_ratio;   // GQA: many Q heads share one KV head

            if (Q_full.get_size() > 0)
                this->copy_into(head.Q.weight,
                    Q_full.slice_cols(h * head_size, (h + 1) * head_size),
                    pfx + "attn_q[h=" + std::to_string(h) + "]");

            if (K_full.get_size() > 0)
                this->copy_into(head.K.weight,
                    K_full.slice_cols(kv_h * head_size, (kv_h + 1) * head_size),
                    pfx + "attn_k[kv=" + std::to_string(kv_h) + "]");

            if (V_full.get_size() > 0)
                this->copy_into(head.V.weight,
                    V_full.slice_cols(kv_h * head_size, (kv_h + 1) * head_size),
                    pfx + "attn_v[kv=" + std::to_string(kv_h) + "]");
        }
    }

    // ── block loading ────────────────────────────────────────────────────────

    
    void load_block(Llama<T>& model, const LlamaHyperParams& hp,
                    size_t blk_idx) {
        Block<T>&    block = *model.decoder_blocks[blk_idx];
        std::string  pfx   = "blk." + std::to_string(blk_idx) + ".";

        // RMSNorms 
        this->copy_into(block.rms1.gamma,
                        this->load_raw(pfx + "attn_norm.weight"),
                        pfx + "attn_norm.weight");
        this->copy_into(block.rms2.gamma,
                        this->load_raw(pfx + "ffn_norm.weight"),
                        pfx + "ffn_norm.weight");

        // Attention output projection
        this->copy_into(block.mha.proJ.weight,
                        this->load_raw_t(pfx + "attn_output.weight"),
                        pfx + "attn_output.weight");

        // Q / K / V (with GQA support)
        load_attention_heads(block.mha, hp, pfx);

        // SwiGLU FFN: gate, up, down
        this->copy_into(block.ffwd.gate_proj.weight,
                        this->load_raw_t(pfx + "ffn_gate.weight"),
                        pfx + "ffn_gate.weight");
        this->copy_into(block.ffwd.up_proj.weight,
                        this->load_raw_t(pfx + "ffn_up.weight"),
                        pfx + "ffn_up.weight");
        this->copy_into(block.ffwd.down_proj.weight,
                        this->load_raw_t(pfx + "ffn_down.weight"),
                        pfx + "ffn_down.weight");
    }

    // ── full weight loading ───────────────────────────────────────────────────

    void load_weights(Llama<T>& model, const LlamaHyperParams& hp) {
        // Token embedding: GGUF [vocab,D] matches Embedding directly
        this->copy_into(model.embedding_table.embeddings,
                        this->load_raw("token_embd.weight"),
                        "token_embd.weight");

        // Final RMSNorm
        this->copy_into(model.rms.gamma,
                        this->load_raw("output_norm.weight"),
                        "output_norm.weight");

        // LM head (or weight tying)
        {
            Matrix<T> w = this->load_raw("output.weight", false);
            if (w.get_size() > 0) {
                this->copy_into(model.lm_head.weight, w.transpose(), "output.weight");
            } else {
                Matrix<T> embd = this->load_raw("token_embd.weight");
                if (embd.get_size() > 0)
                    this->copy_into(model.lm_head.weight,
                                    embd.transpose(), "output.weight (tied)");
            }
        }

        for (size_t blk = 0; blk < hp.n_layer; blk++)
            load_block(model, hp, blk);
    }

public:
    Llama<T> load_model(const std::string& path, const LlamaHyperParams& hp) {
        this->open(path);

        std::cout << "[LlamaGGUFLoader] Building Llama skeleton"
                  << " layers=" << hp.n_layer
                  << " d="      << hp.input_dim
                  << " heads="  << hp.n_heads
                  << " kv_h="   << hp.n_kv_heads << "\n";

        Llama<T> model(hp.vocab_size, hp.input_dim, hp.block_size,
                       hp.n_heads, hp.n_layer, hp.ffn_hidden);

        load_weights(model, hp);
        this->report_params(model);
        std::cout << "[LlamaGGUFLoader] Done.\n";
        return model;
    }
};

