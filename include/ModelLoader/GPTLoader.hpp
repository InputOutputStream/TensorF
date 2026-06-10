#pragma once
#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>

#include "GGUFLoader.hpp"

#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../DataLoader/GGUF.hpp"
#include "../Modules/Transformer/GPT/GPT.hpp"


// ─────────────────────────────────────────────────────────────────────────────
// GPTGGUFLoader  (GPT-2 family)
// ─────────────────────────────────────────────────────────────────────────────
/*
 * GGUF → GPT-2 mapping
 * ────────────────────────────────────────────────────────────────────────────
 * GGUF tensor name               GPT<T> field              Notes
 * ─────────────────────────────────────────────────────────────────────────────
 * token_embd.weight   [D,vocab]  embedding_table.embeddings  (transpose→[vocab,D])
 * position_embd.weight[D,ctx]    position_embedding_table.weight (→[ctx,D])
 * output_norm.weight  [D]        ln_f.gamma
 * output_norm.bias    [D]        ln_f.beta
 * output.weight       [D,vocab]  lm_head.weight              (→[D,vocab] after T)
 *                                                             weight-tying if absent
 * blk.N.attn_norm.weight [D]     blocks[N].ln1.gamma
 * blk.N.attn_norm.bias   [D]     blocks[N].ln1.beta
 * blk.N.ffn_norm.weight  [D]     blocks[N].ln2.gamma
 * blk.N.ffn_norm.bias    [D]     blocks[N].ln2.beta
 * blk.N.attn_qkv.weight  [3D,D]  Head[h].Q/K/V.weight        (→[D,3D], split)
 * blk.N.attn_qkv.bias    [3D]    Head[h].Q/K/V.bias          (split)
 * blk.N.attn_output.weight[D,D]  mha.proJ.weight             (→[D,D])
 * blk.N.attn_output.bias  [D]    mha.proJ.bias
 * blk.N.ffn_up.weight   [4D,D]   ffwd.up.weight              (→[D,4D])
 * blk.N.ffn_up.bias     [4D]     ffwd.up.bias
 * blk.N.ffn_down.weight [D,4D]   ffwd.down.weight              (→[4D,D])
 * blk.N.ffn_down.bias   [D]      ffwd.down.bias
 */


struct GPT2HyperParams {
    size_t vocab_size = 50257;
    size_t d_model    = 768;
    size_t block_size = 1024;
    size_t n_layer    = 12;
    size_t n_head     = 12;
};

template<typename T>
class GPTGGUFLoader : public GGUFLoader<T> 
{
    private:
        using Base = GGUFLoader<T>;

    // ── block loading ────────────────────────────────────────────────────────

    void load_block(GPT<T>& model, const GPT2HyperParams& hp, size_t n) {
        const std::string pfx = "blk." + std::to_string(n) + ".";
        Block<T>&  blk     = *model.decoder_blocks[n];
        const size_t D     = hp.d_model;
        const size_t nh    = hp.n_head;
        const size_t hs    = D / nh;   // per-head size

        // LayerNorms
        this->copy_into(blk.ln1.gamma, this->load_raw(pfx + "attn_norm.weight"),     pfx + "attn_norm.weight");
        this->copy_into(blk.ln1.beta,  this->load_raw(pfx + "attn_norm.bias", false),pfx + "attn_norm.bias");
        this->copy_into(blk.ln2.gamma, this->load_raw(pfx + "ffn_norm.weight"),      pfx + "ffn_norm.weight");
        this->copy_into(blk.ln2.beta,  this->load_raw(pfx + "ffn_norm.bias", false), pfx + "ffn_norm.bias");

        // Fused QKV: GGUF [3D,D] → transpose → [D,3D] → split into Q|K|V
        Matrix<T> qkv_w = this->load_raw(pfx + "attn_qkv.weight").transpose(); // [D, 3D]
        Matrix<T> qkv_b = this->load_raw(pfx + "attn_qkv.bias", false);       // [3D] (1D vector)

        Matrix<T> Q_all = qkv_w.slice_cols(0, D);          // columns 0 .. D-1
        Matrix<T> K_all = qkv_w.slice_cols(D, 2*D);        // columns D .. 2D-1
        Matrix<T> V_all = qkv_w.slice_cols(2*D, 3*D);      // columns 2D .. 3D-1

        for (size_t h = 0; h < nh; h++) {
            Head<T>& head = *blk.mha.Heads[h];
            size_t c0 = h * hs;

            this->copy_into(head.Q.weight, Q_all.slice_cols(c0, c0+hs), pfx + "Q[" + std::to_string(h) + "]");
            this->copy_into(head.K.weight, K_all.slice_cols(c0, c0+hs), pfx + "K[" + std::to_string(h) + "]");
            this->copy_into(head.V.weight, V_all.slice_cols(c0, c0+hs), pfx + "V[" + std::to_string(h) + "]");

            // Bias handling – qkv_b is 1‑D, shape [3*D]
            if (qkv_b.get_size() > 0) {
                std::vector<T> bias_data = qkv_b.get_data();  // length = 3*D
                
                // Q bias
                std::vector<T> q_bias(bias_data.begin() + c0, bias_data.begin() + c0 + hs);
                Matrix<T> q_bias_mat(q_bias, {1, hs});  
                this->copy_into(head.Q.bias, q_bias_mat, pfx + "Qb");
                
                // K bias (offset D)
                std::vector<T> k_bias(bias_data.begin() + D + c0, bias_data.begin() + D + c0 + hs);
                Matrix<T> k_bias_mat(k_bias, {1, hs});
                this->copy_into(head.K.bias, k_bias_mat, pfx + "Kb");
                
                // V bias (offset 2D)
                std::vector<T> v_bias(bias_data.begin() + 2*D + c0, bias_data.begin() + 2*D + c0 + hs);
                Matrix<T> v_bias_mat(v_bias, {1, hs});
                this->copy_into(head.V.bias, v_bias_mat, pfx + "Vb");
            }
        }
    }

    // ── full weight loading ───────────────────────────────────────────────────

    void load_weights(GPT<T>& model, const GPT2HyperParams& hp) {
        // Token embedding: GGUF stores it already in [vocab, D] order after
        // parse — copy_into handles the size check.
        this->copy_into(model.embedding_table.embeddings,
                        this->load_raw("token_embd.weight"),
                        "token_embd.weight");

        // Position embedding: same layout [ctx, D]
        this->copy_into(model.position_embedding_table.weight,
                        this->load_raw("position_embd.weight"),
                        "position_embd.weight");

        // Final LayerNorm
        this->copy_into(model.ln_f.gamma, this->load_raw("output_norm.weight"),        "output_norm.weight");
        this->copy_into(model.ln_f.beta,  this->load_raw("output_norm.bias", false),   "output_norm.bias");

        // LM head (weight tying if absent)
        {
            Matrix<T> w = this->load_raw("output.weight", false);
            if (w.get_size() > 0) {
                // GGUF [D,vocab] → transpose → [vocab,D]  (lm_head is Linear [D,vocab])
                // lm_head.forward: x{1,D} @ W{D,vocab} → {1,vocab}
                // So weight must be [D, vocab]; load_raw gives [D,vocab] already
                // after the GGUF dimension reversal in load_tensor.
                this->copy_into(model.lm_head.weight, w.transpose(), "output.weight");
            } else {
                // Weight tying: lm_head.weight = embedding^T = [D, vocab]
                Matrix<T> embd = this->load_raw("token_embd.weight");
                if (embd.get_size() > 0)
                    this->copy_into(model.lm_head.weight, embd.transpose(), "output.weight (tied)");
            }
        }

        for (size_t n = 0; n < hp.n_layer; n++)
            load_block(model, hp, n);
    }

public:
    GPT<T> load_model(const std::string& path, const GPT2HyperParams& hp) {
        this->open(path);

        std::cout << "[GPTGGUFLoader] Building GPT-2 skeleton"
                  << " layers=" << hp.n_layer
                  << " d="      << hp.d_model
                  << " heads="  << hp.n_head << "\n";

        GPT<T> model(hp.vocab_size, hp.d_model, hp.block_size,
                     hp.n_head, hp.n_layer);

        load_weights(model, hp);
        this->report_params(model);
        std::cout << "[GPTGGUFLoader] Done.\n";
        return model;
    }
};

