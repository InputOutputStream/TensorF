#ifndef __GGUF__LOADER__
#define __GGUF__LOADER__

/*
 * GGUF Loader — GPT-2 family (vanilla transformer, arch = "gpMatrix<T>::transpose")
 *
 * Modèles supportés :
 *   GPT-2 small  : 12 layers, 12 heads, d_model=768,  vocab=50257
 *   GPT-2 medium : 24 layers, 16 heads, d_model=1024, vocab=50257
 *   GPT-2 large  : 36 layers, 20 heads, d_model=1280, vocab=50257
 *   GPT-2 XL     : 48 layers, 25 heads, d_model=1600, vocab=50257
 *
 * Mapping GGUF → votre framework
 * ─────────────────────────────────────────────────────────────────
 *  GGUF tensor name               | Votre paramètre                | Remarque
 *  -------------------------------|--------------------------------|------------------------
 *  token_embd.weight   [D, vocab] | embedding_table.embeddings     | Matrix<T>::transpose → [vocab, D]
 *  position_embd.weight [D, ctx]  | position_embedding_table.weight| Matrix<T>::transpose → [ctx, D]
 *  blk.N.attn_norm.weight [D]    | blocks[N].ln1.gamma            |
 *  blk.N.attn_norm.bias   [D]    | blocks[N].ln1.beta             |
 *  blk.N.ffn_norm.weight  [D]    | blocks[N].ln2.gamma            |
 *  blk.N.ffn_norm.bias    [D]    | blocks[N].ln2.beta             |
 *  blk.N.attn_qkv.weight [3D, D] | Q/K/V de chaque Head           | Matrix<T>::transpose → [D,3D] → split
 *  blk.N.attn_qkv.bias   [3D]   | bias Q/K/V par head            | split
 *  blk.N.attn_output.weight [D,D]| mha.proJ.weight                | Matrix<T>::transpose
 *  blk.N.attn_output.bias  [D]  | mha.proJ.bias                  |
 *  blk.N.ffn_up.weight   [4D, D] | ffwd.l1.weight                 | Matrix<T>::transpose → [D, 4D]
 *  blk.N.ffn_up.bias     [4D]   | ffwd.l1.bias                   |
 *  blk.N.ffn_down.weight [D, 4D] | ffwd.l2.weight                 | Matrix<T>::transpose → [4D, D]
 *  blk.N.ffn_down.bias   [D]    | ffwd.l2.bias                   |
 *  output_norm.weight     [D]    | ln_f.gamma                     |
 *  output_norm.bias       [D]    | ln_f.beta                      |
 *  output.weight          [D,vocab]| lm_head.weight               | absent = weight tying
 *
 * NOTE TRANSPOSITION :
 *   llama.cpp stocke les Linear en [out, in] (convention PyTorch).
 *   Votre Linear attend [in, out]  (y = x @ W).
 *   → on Matrix<T>::transpose toutes les matrices 2-D au chargement.
 *
 * NOTE attn_qkv :
 *   GPT-2 fusionne Q, K, V dans une seule matrice [3*D, D] (après Matrix<T>::transpose : [D, 3D]).
 *   On découpe en 3 blocs de [D, D] puis on répartit par head.
 */

#include <string>
#include <vector>
#include <map>
#include <stdexcept>
#include <iostream>

#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../Modules/Transformer/GPT.hpp"
#include "../DataLoader/GGUF.hpp"

// ─────────────────────────────────────────────
//  Hyperparamètres
// ─────────────────────────────────────────────
struct GPT2HyperParams {
    size_t vocab_size = 50257;
    size_t d_model    = 768;    // n_embd
    size_t block_size = 1024;   // n_ctx
    size_t n_layer    = 12;
    size_t n_head     = 12;
    // ffn_hidden = 4 * d_model
};

template<typename T>
class GGUFLoader {
private:
    GGUF gguf;
    std::map<std::string, size_t> name_to_idx;

    void build_index() {
        name_to_idx.clear();
        for (size_t i = 0; i < gguf.tensors.size(); i++)
            name_to_idx[gguf.tensors[i].name] = i;
    }

    // Charge un tenseur par nom. Retourne matrice vide si absent.
    Matrix<T> load_raw(const std::string& name, bool required = true) {
        auto it = name_to_idx.find(name);
        if (it == name_to_idx.end()) {
            if (required)
                std::cerr << "[GGUFLoader] MANQUANT : " << name << "\n";
            return Matrix<T>();
        }
        const TensorInfo& info = gguf.tensors[it->second];
        std::cout << "[GGUFLoader] " << name << "  [";
        for (size_t d = 0; d < info.dimensions.size(); d++) {
            std::cout << info.dimensions[d];
            if (d + 1 < info.dimensions.size()) std::cout << "x";
        }
        std::cout << "]  ggml_type=" << info.ggml_type << "\n";
        return gguf.load_tensor<T>(gguf.file, info, gguf.data_start_offset);
    }

    // Copie securisee avec verification de taille
    void copy_into(Tensor_t<T> dst, Matrix<T> src, std::string name) {
        if (src.get_size() == 0) return;
        if (src.get_size() != dst->val.get_size()) {
            std::cerr << "[GGUFLoader] TAILLE INCOMPATIBLE '" << name
                      << "' : GGUF=" << src.get_size()
                      << " modele=" << dst->val.get_size() << " -> ignore\n";
            return;
        }
        dst->val.copy_from(src);
        dst->shape = src.shape;
    }

    // Slice de colonnes [rows, total] -> [rows, n_cols] a partir de col_start
    Matrix<T> col_slice(const Matrix<T>& src, size_t col_start, size_t n_cols) {
        if (src.shape.size() != 2)
            throw std::runtime_error("col_slice: matrice 2D attendue");
        size_t rows = src.shape[0], total = src.shape[1];
        if (col_start + n_cols > total)
            throw std::runtime_error("col_slice: hors limites");
        std::vector<T> out(rows * n_cols);
        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < n_cols; c++)
                out[r * n_cols + c] = src.data[r * total + col_start + c];
        return Matrix<T>(out, {rows, n_cols});
    }

    // Slice d'un vecteur 1D
    Matrix<T> vec_slice(Matrix<T> src, size_t start, size_t len) {
        if (src.get_size() == 0) return src;
        std::vector<T> out(len);
        for (size_t i = 0; i < len; i++) out[i] = src.data[start + i];
        return Matrix<T>(out, {len});
    }

    // ─────────────────────────────────────────────────────────────
    void load_weights(GPT<T>& model, const GPT2HyperParams& hp) {

        assert(model.position_embedding_table.weight->val.get_size() == hp.block_size * hp.d_model
        && "position_embedding_table weight shape mismatch — check GPT constructor arg order");
        assert(model.embedding_table.embeddings->val.get_size() == hp.vocab_size * hp.d_model
        && "embedding_table shape mismatch");

        // Token embedding : GGUF [D, vocab] -> T2 -> [vocab, D]
        copy_into(model.embedding_table.embeddings,
                load_raw("token_embd.weight"),
                "token_embd.weight");

        copy_into(model.position_embedding_table.weight,
                load_raw("position_embd.weight"),
                "position_embd.weight");
        
        // Final LayerNorm
        copy_into(model.ln_f.gamma, load_raw("output_norm.weight"), "output_norm.weight");
        copy_into(model.ln_f.beta,  load_raw("output_norm.bias", false), "output_norm.bias");

        // LM Head (weight tying si output.weight absent)
        {
            Matrix<T> w = load_raw("output.weight", false);
            if (w.get_size() > 0) {
                copy_into(model.lm_head.weight, w.transpose(), "output.weight");
            } else {
                // GGUF [D, vocab] -> Matrix<T>::transpose -> [vocab, D] -> Matrix<T>::transpose -> [D, vocab]
                Matrix<T> tied = load_raw("token_embd.weight").transpose(); // [d_model, vocab_size]
                copy_into(model.lm_head.weight, tied, "output.weight (weight tying)");
            }
        }

        for (size_t n = 0; n < hp.n_layer; n++)
            load_block(model, hp, n);
    }

    // ─────────────────────────────────────────────────────────────
    void load_block(GPT<T>& model, const GPT2HyperParams& hp, size_t n) {
        const std::string pfx = "blk." + std::to_string(n) + ".";
        Block<T>& blk         = *model.decoder_blocks[n];
        const size_t D        = hp.d_model;
        const size_t n_heads  = hp.n_head;
        const size_t hs       = D / n_heads;   // head_size

        // LayerNorm 1 (pre-attention)
        copy_into(blk.ln1.gamma, load_raw(pfx + "attn_norm.weight"),      pfx + "attn_norm.weight");
        copy_into(blk.ln1.beta,  load_raw(pfx + "attn_norm.bias", false), pfx + "attn_norm.bias");

        // LayerNorm 2 (pre-FFN)
        copy_into(blk.ln2.gamma, load_raw(pfx + "ffn_norm.weight"),       pfx + "ffn_norm.weight");
        copy_into(blk.ln2.beta,  load_raw(pfx + "ffn_norm.bias", false),  pfx + "ffn_norm.bias");

        // QKV fusionne
        // GGUF stocke [3D, D] (out=3D, in=D) -> apres Matrix<T>::transpose : [D, 3D]
        // Ordre : Q | K | V (colonnes 0..D-1, D..2D-1, 2D..3D-1)
        {
            Matrix<T> qkv_w = load_raw(pfx + "attn_qkv.weight").transpose();  // [D, 3D]
            Matrix<T> qkv_b = load_raw(pfx + "attn_qkv.bias", false); // [3D]

            Matrix<T> Q_all = col_slice(qkv_w, 0,   D);  // [D, D]
            Matrix<T> K_all = col_slice(qkv_w, D,   D);  // [D, D]
            Matrix<T> V_all = col_slice(qkv_w, 2*D, D);  // [D, D]

            for (size_t h = 0; h < n_heads; h++) {
                Head<T>& head = *blk.mha.Heads[h];
                size_t c0 = h * hs;

                copy_into(head.Q.weight, col_slice(Q_all, c0, hs),
                          pfx + "Q[" + std::to_string(h) + "]");
                copy_into(head.K.weight, col_slice(K_all, c0, hs),
                          pfx + "K[" + std::to_string(h) + "]");
                copy_into(head.V.weight, col_slice(V_all, c0, hs),
                          pfx + "V[" + std::to_string(h) + "]");

                if (qkv_b.get_size() > 0) {
                    copy_into(head.Q.bias, vec_slice(qkv_b, c0,       hs), pfx + "Qb");
                    copy_into(head.K.bias, vec_slice(qkv_b, D   + c0, hs), pfx + "Kb");
                    copy_into(head.V.bias, vec_slice(qkv_b, 2*D + c0, hs), pfx + "Vb");
                }
            }
        }

        // Projection de sortie attention : [D, D] -> Matrix<T>::transpose -> [D, D]
        copy_into(blk.mha.proJ.weight,
                  load_raw(pfx + "attn_output.weight").transpose(),
                  pfx + "attn_output.weight");
        copy_into(blk.mha.proJ.bias,
                  load_raw(pfx + "attn_output.bias", false),
                  pfx + "attn_output.bias");

        // FFN up : GGUF [4D, D] -> Matrix<T>::transpose -> [D, 4D]
        copy_into(blk.ffwd.l1.weight,
                  load_raw(pfx + "ffn_up.weight").transpose(),
                  pfx + "ffn_up.weight");
        copy_into(blk.ffwd.l1.bias,
                  load_raw(pfx + "ffn_up.bias", false),
                  pfx + "ffn_up.bias");

        // FFN down : GGUF [D, 4D] -> Matrix<T>::transpose -> [4D, D]
        copy_into(blk.ffwd.l2.weight,
                  load_raw(pfx + "ffn_down.weight").transpose(),
                  pfx + "ffn_down.weight");
        copy_into(blk.ffwd.l2.bias,
                  load_raw(pfx + "ffn_down.bias", false),
                  pfx + "ffn_down.bias");
    }

public:
    // ─────────────────────────────────────────────────────────────
    //  Point d'entree
    // ─────────────────────────────────────────────────────────────
    GPT<T> load_model(const std::string& path, const GPT2HyperParams& hp) {
        std::cout << "[GGUFLoader] Parsing : " << path << "\n";
        gguf.parse_gguf(path);
        build_index();

        std::cout << "[GGUFLoader] Instanciation GPT-2 (layers=" << hp.n_layer
                  << " d=" << hp.d_model << " heads=" << hp.n_head << ")\n";
        GPT<T> model(hp.vocab_size, hp.d_model, hp.block_size, hp.n_head, hp.n_layer);

        load_weights(model, hp);
        std::cout << "[GGUFLoader] Chargement termine.\n";
        size_t nb = 0;
        for (auto m : model.parameters()) nb += m->val.get_size();
        std::cout <<"N° de Parametres: " << nb << "\n";
        return model;
    }

    // Inspecter un fichier GGUF inconnu
    void inspect(const std::string& path) {
        gguf.parse_gguf(path);
        std::cout << "\n=== " << path << " (" << gguf.tensors.size() << " tenseurs) ===\n";
        for (const auto& t : gguf.tensors) {
            std::cout << "  " << t.name << "  [";
            for (size_t d = 0; d < t.dimensions.size(); d++) {
                std::cout << t.dimensions[d];
                if (d + 1 < t.dimensions.size()) std::cout << "x";
            }
            std::cout << "]  type=" << t.ggml_type << "\n";
        }
        std::cout << "\n";
    }
};

#endif // __GGUF__LOADER__