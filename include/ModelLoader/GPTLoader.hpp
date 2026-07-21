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
#include "../Tokenizer/GPT2Tokenizer.hpp"
#include "../Modules/Transformer/GPT/GPT.hpp"

// ─────────────────────────────────────────────────────────────────────────────
// GPTGGUFLoader  (GPT-2 family)
// ─────────────────────────────────────────────────────────────────────────────
/*
 * GGUF → GPT-2 weight mapping
 * ─────────────────────────────────────────────────────────────────────────────
 * GGUF stores [out, in] PyTorch weights with fastest dim first, i.e. as [in, out].
 * load_tensor() reverses → Matrix{out, in} = PyTorch shape.
 * New Linear.weight = {out, in}  →  NO .transpose() for any Linear weight.
 * LoRALinear.weight is also {out, in} now (see LoRALinear.hpp) so this loader
 * works unmodified whether LinearT is Linear or LoRALinear — either way we're
 * filling in the frozen/base weight path; LoRA's A/B adapters are untouched
 * here and only get initialized by LoRALinear's own constructor.
 *
 * GGUF name                  PyTorch shape   load_raw shape   stored as
 * ──────────────────────────────────────────────────────────────────────
 * token_embd.weight          [vocab, D]      {vocab, D}       Embedding ✓
 * position_embd.weight       [ctx,   D]      {ctx,   D}       PE weight ✓
 * output_norm.weight/bias    [D]             {D}              1-D       ✓
 * output.weight              [vocab, D]      {vocab, D}       lm_head   ✓
 * blk.N.attn_norm.weight/bias[D]             {D}              1-D       ✓
 * blk.N.ffn_norm.weight/bias [D]             {D}              1-D       ✓
 * blk.N.attn_qkv.weight      [3D, D]         {3D, D}          slice_row per head → {hs,D} ✓
 * blk.N.attn_qkv.bias        [3D]            {3D}             split flat per head ✓
 * blk.N.attn_output.weight   [D,  D]         {D,  D}          proJ      ✓
 * blk.N.attn_output.bias     [D]             {D}              1-D       ✓
 * blk.N.ffn_up.weight        [4D, D]         {4D, D}          up        ✓
 * blk.N.ffn_up.bias          [4D]            {4D}             1-D       ✓
 * blk.N.ffn_down.weight      [D,  4D]        {D,  4D}         down      ✓
 * blk.N.ffn_down.bias        [D]             {D}              1-D       ✓
 */

struct GPT2HyperParams {
    size_t vocab_size = 50257;
    size_t d_model    = 768;
    size_t block_size = 1024;
    size_t n_layer    = 12;
    size_t n_head     = 12;
};

template <typename T, template<typename> class LinearT>
class GPTGGUFLoader : public GGUFLoader<T> {
    static_assert(std::is_same_v<LinearT<T>, Linear<T>>,
        "GPTGGUFLoader only supports LinearT=Linear — GGUF files store dense "
        "pretrained weights, not LoRA adapters. To fine-tune with LoRA: "
        "load a dense model via GPTGGUFLoader<T,Linear>, construct a "
        "separate GPT<T,LoRALinear> student, then call "
        "student.load_backbone_from(pretrained) and "
        "student.load_head_from(pretrained).");

    private:
        using Base = GGUFLoader<T>;

    // ── block loading ────────────────────────────────────────────────────────

    void load_block(GPT<T, LinearT>& model, const GPT2HyperParams& hp, size_t n) {
        const std::string pfx = "blk." + std::to_string(n) + ".";
        // model.decoder_blocks is private on GPT<T,LinearT> — GPT's friend
        // declares GPTGGUFLoader<T,LinearT> a friend, so this direct access
        // is fine (unlike blk's own internals below, which we now go through
        // accessors for since Block/MHA/FeedForward's friend list is the same
        // templated-friend pattern but the fields we need — Heads, proJ,
        // up/down, ln1/ln2 — are reached via get_*() rather than raw field
        // access, matching how load_backbone_from() in GPT.hpp does it).
        Block<T, LinearT>& blk = *model.decoder_blocks[n];
        const size_t D   = hp.d_model;
        const size_t nh  = hp.n_head;
        const size_t hs  = D / nh;   // per-head size

        // ── LayerNorms (1-D) ─────────────────────────────────────────────────
        this->copy_into(blk.get_ln1().gamma, this->load_raw(pfx + "attn_norm.weight"),      pfx + "attn_norm.weight");
        this->copy_into(blk.get_ln1().beta,  this->load_raw(pfx + "attn_norm.bias", false), pfx + "attn_norm.bias");
        this->copy_into(blk.get_ln2().gamma, this->load_raw(pfx + "ffn_norm.weight"),       pfx + "ffn_norm.weight");
        this->copy_into(blk.get_ln2().beta,  this->load_raw(pfx + "ffn_norm.bias", false),  pfx + "ffn_norm.bias");

        // ── Fused QKV ────────────────────────────────────────────────────────
        // PyTorch [3D, D] → load_raw → {3D, D}.
        // Row layout: rows [0,D)=Q, [D,2D)=K, [2D,3D)=V, each split into nh heads.
        // Matrix::slice_row(start, end) → {hs, D} = {out, in} — no transpose needed.
        Matrix<T> qkv_w = this->load_raw(pfx + "attn_qkv.weight");      // {3D, D}
        Matrix<T> qkv_b = this->load_raw(pfx + "attn_qkv.bias", false); // {3D} flat

        auto& heads = blk.get_mha().get_heads();
        for (size_t h = 0; h < nh; h++) {
            Head<T, LinearT>& head = *heads[h];
            size_t r0 = h * hs;

            this->copy_into(head.get_Q().weight, qkv_w.slice_row(r0,       r0 + hs),
                            pfx + "Q[" + std::to_string(h) + "]");
            this->copy_into(head.get_K().weight, qkv_w.slice_row(D  + r0,  D  + r0 + hs),
                            pfx + "K[" + std::to_string(h) + "]");
            this->copy_into(head.get_V().weight, qkv_w.slice_row(2*D + r0, 2*D + r0 + hs),
                            pfx + "V[" + std::to_string(h) + "]");

            // Bias: flat {3D}; slice per head per projection
            if (qkv_b.get_size() > 0) {
                const std::vector<T>& bd = qkv_b.get_data();
                std::vector<T> qb(bd.begin() + r0,        bd.begin() + r0 + hs);
                std::vector<T> kb(bd.begin() + D  + r0,   bd.begin() + D  + r0 + hs);
                std::vector<T> vb(bd.begin() + 2*D + r0,  bd.begin() + 2*D + r0 + hs);
                this->copy_into(head.get_Q().bias, Matrix<T>(qb, {hs}), pfx + "Qb[" + std::to_string(h) + "]");
                this->copy_into(head.get_K().bias, Matrix<T>(kb, {hs}), pfx + "Kb[" + std::to_string(h) + "]");
                this->copy_into(head.get_V().bias, Matrix<T>(vb, {hs}), pfx + "Vb[" + std::to_string(h) + "]");
            }
        }

        // ── Attention output projection ──────────────────────────────────────
        // PyTorch [D, D] → load_raw → {D, D} = {out, in} ✓
        this->copy_into(blk.get_mha().get_proj().weight,
                        this->load_raw(pfx + "attn_output.weight"),
                        pfx + "attn_output.weight");
        this->copy_into(blk.get_mha().get_proj().bias,
                        this->load_raw(pfx + "attn_output.bias", false),
                        pfx + "attn_output.bias");

        // ── FFN weights ──────────────────────────────────────────────────────
        // up  : PyTorch [4D, D] → load_raw → {4D, D} = {out, in} ✓
        // down: PyTorch [D, 4D] → load_raw → {D, 4D} = {out, in} ✓
        this->copy_into(blk.get_ffwd().get_up().weight,
                        this->load_raw(pfx + "ffn_up.weight"),
                        pfx + "ffn_up.weight");
        this->copy_into(blk.get_ffwd().get_up().bias,
                        this->load_raw(pfx + "ffn_up.bias", false),
                        pfx + "ffn_up.bias");
        this->copy_into(blk.get_ffwd().get_down().weight,
                        this->load_raw(pfx + "ffn_down.weight"),
                        pfx + "ffn_down.weight");
        this->copy_into(blk.get_ffwd().get_down().bias,
                        this->load_raw(pfx + "ffn_down.bias", false),
                        pfx + "ffn_down.bias");
    }

    // ── full weight loading ───────────────────────────────────────────────────

    void load_weights(GPT<T, LinearT>& model, const GPT2HyperParams& hp) {

        // Token embedding: PyTorch [vocab, D] → load_raw → {vocab, D} ✓
        this->copy_into(model.get_embedding_table().embeddings,
                        this->load_raw("token_embd.weight"),
                        "token_embd.weight");

        // Position embedding: PyTorch [ctx, D] → load_raw → {ctx, D} ✓
        this->copy_into(model.get_position_embedding().weight,
                        this->load_raw("position_embd.weight"),
                        "position_embd.weight");

        // Final LayerNorm (1-D)
        this->copy_into(model.get_ln_f().gamma, this->load_raw("output_norm.weight"),       "output_norm.weight");
        this->copy_into(model.get_ln_f().beta,  this->load_raw("output_norm.bias", false),  "output_norm.bias");

        // LM head: PyTorch [vocab, D] → load_raw → {vocab, D} = {out, in} ✓
        // Tied-weight fallback: token_embd → {vocab, D} = {out, in} ✓
        {
            Matrix<T> w = this->load_raw("output.weight", false);
            if (w.get_size() > 0) {
                this->copy_into(model.get_lm_head().weight, w, "output.weight");
            } else {
                Matrix<T> embd = this->load_raw("token_embd.weight");
                if (embd.get_size() > 0)
                    this->copy_into(model.get_lm_head().weight, embd, "output.weight (tied)");
            }
        }

        for (size_t n = 0; n < hp.n_layer; n++)
            load_block(model, hp, n);
    }

public:
    GPT<T, LinearT> load_model(const std::string& path, const GPT2HyperParams& hp) {
        this->open(path);
        std::cout << "[GPTGGUFLoader] Building GPT-2 skeleton"
                  << " layers=" << hp.n_layer
                  << " d="      << hp.d_model
                  << " heads="  << hp.n_head << "\n";
        GPT<T, LinearT> model(hp.vocab_size, hp.d_model, hp.block_size, hp.n_head, hp.n_layer);
        load_weights(model, hp);
        this->report_params(model);
        std::cout << "[GPTGGUFLoader] Done.\n";
        return model;
    }

    GPT2Tokenizer load_tokenizer() {
        auto vocab  = this->get_metadata_array("tokenizer.ggml.tokens");
        auto merges = this->get_metadata_array("tokenizer.ggml.merges");
        if (vocab.empty() || merges.empty()) {
            std::cerr << "[GPTGGUFLoader] Warning: tokenizer metadata not found, "
                         "returning empty tokenizer.\n";
            return GPT2Tokenizer();
        }
        GPT2Tokenizer tokenizer;
        tokenizer.load_from_arrays(vocab, merges);
        return tokenizer;
    }
};