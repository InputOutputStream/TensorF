/* client.cpp  —  TensorF Federated Learning Client
 * ==================================================
 *
 *  Architectural contract:
 *    • The client owns ONLY the student model.
 *    • In FedAvg:     train locally → send weight deltas → receive global weights.
 *    • In FedDistill: compute logits on proxy batch → send to server →
 *                     receive averaged consensus → distill_logits() against it.
 *    • The "teacher" in FedDistill is the server-averaged consensus tensor
 *      received over the wire; the client never loads or references a teacher model.
 *    • All training math (forward/backward/step, serialisation) lives in Trainer.
 *      The client supplies a batch callback and reads back the final loss.
 *
 *  Networking is delegated entirely to Client<float> (Client.hpp):
 *    connect_to_server()  → net.connect_to_server()
 *    send_deltas()        → streams student.parameters() via net.sendChunked()
 *    receive_weights()    → net.receiveChunked() straight into student.parameters()
 *    (no full-model flat buffer is built on either side — see --chunk-mb)
 *
 *  Build:
 *    make client         (CLIENT_SRC=include/Network/client.cpp in Makefile)
 *
 *  Usage:
 *    ./bin/client [--server 127.0.0.1] [--port 8080]
 *                [--model SLM/SmolLM2-135M-Instruct-f16.gguf]
 *                [--dataset Dataset]
 *                [--iters 50] [--eval 10]
 *                [--batch 4] [--block 512]
 *                [--prompt "the data type is int"]
 *                [--tokens 40]
 *                [--no-profile]      skip hardware profiling
 *                [--json-log]        write profiler JSON to client_profile.json
 *                [--no-federated]    train only locally, no server connection
 *                [--feddistill]      use FedDistill round instead of FedAvg
 *                [--client-id default]   namespaces default checkpoint paths
 *                [--save-path PATH]      where to save the student (default:
 *                                        checkpoints/client_<id>/student.tnsf)
 *                [--load-path PATH]      resume the student from a checkpoint
 *                [--save-every 1]        save every N rounds, 0=never
 *                [--rounds 0]            stop after N rounds, 0=run forever
 *                [--quantize none]       also save a compressed copy: none|fp8|fp4
 */

// ─── Standard library ────────────────────────────────────────────────────────
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <functional>
#include <optional>
#include <filesystem>
#include <algorithm>

// ─── POSIX networking ────────────────────────────────────────────────────────
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

// ─── TensorF core ────────────────────────────────────────────────────────────
#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"

#include "../Modules/Transformer/GPT/GPT.hpp"
#include "../ModelLoader/GPTLoader.hpp"
#include "../Tokenizer/GPT2Tokenizer.hpp"

// Dataset
#include "../DataLoader/DataLoading.hpp"

// ─── TensorF profiler ────────────────────────────────────────────────────────
#include "../Profiler/Profiler.hpp"
#include "../Profiler/HyperparamAdvisor.hpp"

// ─── Network utilities + Client base class ───────────────────────────────────
// Client.hpp includes io_utils.hpp internally.
#include "Client.hpp"

// ─── Central training + aggregation impl ─────────────────────────────────────
// Trainer<Model, T> owns forward/backward/step, weight serialisation,
// and (server-side) aggregation. The client uses:
//   finetune()       — local training loop (FEDAVG mode)
//   compute_logits() — proxy-batch forward for FedDistill phase 1
//   distill_logits() — train against server consensus  (FedDistill phase 2)
//   get_flat_weights() / set_flat_weights() — wire serialisation
#include "Trainner.hpp"

// ════════════════════════════════════════════════════════════════════════════
//  Wall-clock timer
// ════════════════════════════════════════════════════════════════════════════

struct Timer {
    using Clock = std::chrono::steady_clock;
    Clock::time_point t0 = Clock::now();

    void reset() { t0 = Clock::now(); }

    double ms() const {
        return std::chrono::duration<double, std::milli>(
            Clock::now() - t0).count();
    }
    double sec() const { return ms() / 1000.0; }

    std::string str() const {
        double m = ms();
        std::ostringstream s;
        if (m < 1000.0)
            s << std::fixed << std::setprecision(1) << m << " ms";
        else
            s << std::fixed << std::setprecision(2) << sec() << " s";
        return s.str();
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  Checkpoint path helper
// ════════════════════════════════════════════════════════════════════════════

/// std::ofstream doesn't create directories — make sure the checkpoint's
/// parent directory exists before Trainer/Module tries to write into it.
static void ensure_parent_dir(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty())
        std::filesystem::create_directories(parent);
}

// ════════════════════════════════════════════════════════════════════════════
//  CLI options
// ════════════════════════════════════════════════════════════════════════════

struct ClientOptions {
    std::string server_ip    = "127.0.0.1";
    uint16_t    port         = 8080;
    std::string model_path   = "SLM/gpt2-small-f32.gguf";
    std::string dataset_path = "Dataset";
    size_t      iters        = 50;
    size_t      eval_iters   = 10;
    size_t      batch_size   = 1;
    size_t      block_size   = 128;
    float       lr           = 1e-4f;
    std::string prompt       = "the data type is int";
    size_t      gen_tokens   = 40;
    bool        no_profile   = false;
    bool        json_log     = false;
    bool        no_federated = false;
    bool        feddistill   = false;   // use FedDistill instead of FedAvg
    bool        verbose      = true;
    bool        help         = false;

    // ── Checkpointing ────────────────────────────────────────────────────
    // The client owns the student model (see file header), so it owns
    // saving/loading it too. Defaults are namespaced by --client-id so
    // multiple clients on one machine don't clobber each other's state —
    // this matters most in FedDistill, where each client's student is its
    // own personalized model, not a shared global one.
    std::string client_id    = "default";
    std::string save_path;              // default derived from client_id below
    std::string load_path;              // empty = start from --model GGUF only
    size_t      save_every   = 1;       // save every N rounds (0 = never)
    size_t      max_rounds   = 0;       // 0 = run forever (existing behaviour)
    std::string quantize     = "none";  // "none" | "fp8" | "fp4" — extra compressed save

    // ── Chunked transfer (low-RAM machines) ───────────────────────────────
    // Sizes THIS client's outgoing send_deltas() chunks. Smaller = less peak
    // memory for the round-trip (weight deltas are chunked straight from/
    // into student.parameters() — no full-model flat buffer is ever built),
    // more per-chunk framing overhead (8 bytes/chunk, negligible above a
    // few hundred KB). Doesn't need to match the server's own --chunk-mb —
    // recv_chunked() reads whatever chunk size the sender announces. Lower
    // this (e.g. 1-2) on the weak machine specifically; the server's
    // --chunk-mb separately controls how big the BROADCAST chunks back to
    // this client are, so lower that too if THIS client is the weak one.
    double      chunk_mb     = 8.0;
};

GPT2HyperParams GPTp {
        .vocab_size = 50257,
        .d_model    = 768,
        .block_size = 1024,
        .n_layer    = 12,
        .n_head     = 12
    };

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --server <ip>       Server IP (default: 127.0.0.1)\n"
        "  --port <n>          Server port (default: 8080)\n"
        "  --model <path>      GGUF model file (SLM/...)\n"
        "  --dataset <path>    Text dataset directory\n"
        "  --iters <n>         Training iterations per round (default: 50)\n"
        "  --eval <n>          Eval interval (default: 10)\n"
        "  --batch <n>         Batch size (default: 4)\n"
        "  --block <n>         Block size / context length (default: 512)\n"
        "  --lr <f>            Learning rate (default: 1e-4)\n"
        "  --prompt <text>     Test prompt for inference after each round\n"
        "  --tokens <n>        Tokens to generate (default: 40)\n"
        "  --no-profile        Skip hardware profiling\n"
        "  --json-log          Write profiler JSON to client_profile.json\n"
        "  --no-federated      Local training only, skip server connection\n"
        "  --feddistill        Use FedDistill round (logit exchange) instead of FedAvg\n"
        "  --client-id <name>  Namespaces default checkpoint paths (default: \"default\")\n"
        "  --save-path <path>  Where to save the student checkpoint\n"
        "                      (default: checkpoints/client_<id>/student.tnsf)\n"
        "  --load-path <path>  Resume the student from a checkpoint before training\n"
        "                      (default: none — start from --model GGUF weights)\n"
        "  --save-every <n>    Save every n rounds, 0=never (default: 1)\n"
        "  --rounds <n>        Max rounds, 0=run forever (default: 0)\n"
        "  --quantize <fmt>    Also save a compressed checkpoint: none|fp8|fp4 (default: none)\n"
        "  --chunk-mb <n>      Outgoing weight-delta chunk size in MB (default: 8). Lower\n"
        "                      this on a low-RAM machine; also lower the SERVER's own\n"
        "                      --chunk-mb if this client is the one that's weak (that\n"
        "                      controls the broadcast chunk size coming back to it).\n"
        "  --quiet             Suppress verbose output\n"
        "  --help              Show this help\n",
        prog);
}

static ClientOptions parse_args(int argc, char* argv[]) {
    ClientOptions o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--help")         o.help         = true;
        else if (a == "--no-profile")   o.no_profile   = true;
        else if (a == "--json-log")     o.json_log     = true;
        else if (a == "--no-federated") o.no_federated = true;
        else if (a == "--feddistill")   o.feddistill   = true;
        else if (a == "--quiet")        o.verbose      = false;
        else if (a == "--server"  && i+1 < argc) o.server_ip   = argv[++i];
        else if (a == "--port"    && i+1 < argc) o.port        = (uint16_t)atoi(argv[++i]);
        else if (a == "--model"   && i+1 < argc) o.model_path  = argv[++i];
        else if (a == "--dataset" && i+1 < argc) o.dataset_path= argv[++i];
        else if (a == "--iters"   && i+1 < argc) o.iters       = atoi(argv[++i]);
        else if (a == "--eval"    && i+1 < argc) o.eval_iters  = atoi(argv[++i]);
        else if (a == "--batch"   && i+1 < argc) o.batch_size  = atoi(argv[++i]);
        else if (a == "--block"   && i+1 < argc) o.block_size  = atoi(argv[++i]);
        else if (a == "--lr"      && i+1 < argc) o.lr          = (float)atof(argv[++i]);
        else if (a == "--prompt"  && i+1 < argc) o.prompt      = argv[++i];
        else if (a == "--tokens"  && i+1 < argc) o.gen_tokens  = atoi(argv[++i]);
        else if (a == "--client-id"  && i+1 < argc) o.client_id   = argv[++i];
        else if (a == "--save-path"  && i+1 < argc) o.save_path   = argv[++i];
        else if (a == "--load-path"  && i+1 < argc) o.load_path   = argv[++i];
        else if (a == "--save-every" && i+1 < argc) o.save_every  = atoi(argv[++i]);
        else if (a == "--rounds"     && i+1 < argc) o.max_rounds  = atoi(argv[++i]);
        else if (a == "--quantize"   && i+1 < argc) o.quantize    = argv[++i];
        else if (a == "--chunk-mb"   && i+1 < argc) o.chunk_mb    = atof(argv[++i]);
        else { fprintf(stderr, "[client] Unknown option: %s\n", a.c_str()); o.help = true; }
    }
    // Default checkpoint path, namespaced by client-id so concurrent clients
    // on one machine (e.g. FedDistill, where each owns a personalized model)
    // don't overwrite each other.
    if (o.save_path.empty())
        o.save_path = "checkpoints/client_" + o.client_id + "/student.tnsf";
    return o;
}

// ════════════════════════════════════════════════════════════════════════════
//  Per-iteration and per-round stats
// ════════════════════════════════════════════════════════════════════════════

struct RoundStats {
    size_t  round_no        = 0;
    double  train_total_ms  = 0;
    double  train_loss      = 0;
    double  send_ms         = 0;
    double  recv_ms         = 0;
    double  encode_ms       = 0;
    double  infer_ms        = 0;
    double  decode_ms       = 0;
    size_t  prompt_tokens   = 0;
    size_t  gen_tokens      = 0;
    double  tokens_per_sec  = 0;
    double  rss_mb          = 0;
    std::string generated;

    // ── Network bandwidth (this round) ───────────────────────────────────
    // send = bytes written during send_deltas()/sendLogits()
    // recv = bytes read during receive_weights()/receiveLogits()
    // Captured via io_utils.hpp's global counters (snapshot-diffed around
    // each call) — see send_deltas()/receive_weights() and
    // run_feddistill_round() below.
    double  send_mb         = 0;
    double  send_mbps       = 0;
    double  recv_mb         = 0;
    double  recv_mbps       = 0;
};

static void print_round_stats(const RoundStats& s) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  CLIENT  Round %-4zu                                      ║\n", s.round_no);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Training total   : %8.1f ms                          ║\n", s.train_total_ms);
    printf("║  Training loss    : %8.4f                             ║\n", s.train_loss);
    printf("║  Send deltas      : %8.1f ms   (%6.2f MB, %6.1f MB/s)    ║\n",
           s.send_ms, s.send_mb, s.send_mbps);
    printf("║  Receive weights  : %8.1f ms   (%6.2f MB, %6.1f MB/s)    ║\n",
           s.recv_ms, s.recv_mb, s.recv_mbps);
    printf("║  Encode (prompt)  : %8.3f ms  [%zu tokens]            ║\n",
           s.encode_ms, s.prompt_tokens);
    printf("║  Inference        : %8.1f ms  [%zu tokens]            ║\n",
           s.infer_ms, s.gen_tokens);
    printf("║  Decode           : %8.3f ms                          ║\n", s.decode_ms);
    printf("║  Tokens/sec       : %8.1f                             ║\n", s.tokens_per_sec);
    printf("║  RSS              : %8.0f MB                          ║\n", s.rss_mb);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Generated: %-44s║\n", s.generated.substr(0, 44).c_str());
    if (s.generated.size() > 44)
        printf("║             %-44s║\n", s.generated.substr(44, 44).c_str());
    printf("╚══════════════════════════════════════════════════════════╝\n");
}

// ════════════════════════════════════════════════════════════════════════════
//  FederatedClient
// ════════════════════════════════════════════════════════════════════════════
//
//  Owns:
//    • student  — the only model on the client. No teacher model here.
//    • trainer  — wraps student with the optimizer; drives local training
//                 (finetune, compute_logits, distill_logits) and serialisation.
//    • net      — Client<float> for all socket I/O.
//
//  In FedDistill the "teacher" signal is the consensus tensor the server
//  broadcasts after averaging all clients' proxy-batch logits. It arrives
//  as a std::vector<float> over the wire and is passed directly to
//  trainer.distill_logits() — there is no teacher model on the client.

class FederatedClient {
    // ── Student model ─────────────────────────────────────────────────────────
    // The ONLY model the client owns. No teacher.
    GPT2HyperParams        hp;
    GPTGGUFLoader<float>  loader;
    GPT<float>             student;
    GPT2Tokenizer          tokenizer;

    // ── Trainer ───────────────────────────────────────────────────────────────
    // Mode = FEDAVG (default) or FEDDISTILL (set below based on opts).
    // No teacher pointer — federated clients never hold a local teacher model.
    // distill_logits() accepts the teacher signal as an external tensor.
    Trainer<GPT<float>, float> trainer;

    // ── Dataset ──────────────────────────────────────────────────────────────
    TextDataset<float>     dataset;

    // ── Networking — delegated to Client<float> ──────────────────────────────
    Client<float>          net;
    ClientOptions          opts;

    // ── Profiler ─────────────────────────────────────────────────────────────
    Profiler               profiler;
    MemoryProfiler         mem;
    HyperparamConfig       hw_config;

    // ── Network helpers ───────────────────────────────────────────────────────

    bool connect_to_server() {
        bool ok = net.connect_to_server(opts.server_ip, opts.port);
        if (ok)
            printf("[client] Connected to %s:%u\n", opts.server_ip.c_str(), opts.port);
        return ok;
    }

    // ── Flat-offset ↔ tensor mapping for chunked I/O ──────────────────────────
    // Mirrors Server.hpp's FlatParamCursor exactly (same logical layout as
    // Trainer::get_flat_weights()/set_flat_weights(), just walked
    // incrementally instead of via one big flat buffer). Duplicated here
    // rather than shared because Client.hpp deliberately stays model-
    // agnostic (see its own header comment) — this cursor needs
    // Tensor_t<T>, so it lives where the model is actually visible.
    struct FlatParamCursor {
        const std::vector<Tensor_t<float>>& params;
        size_t   tensor_idx    = 0;
        size_t   pos_in_tensor = 0;
        uint64_t flat_pos      = 0;

        explicit FlatParamCursor(const std::vector<Tensor_t<float>>& p) : params(p) {}

        void seek(uint64_t target) {
            while (flat_pos < target && tensor_idx < params.size()) {
                size_t tsize = params[tensor_idx]->val.get_size();
                size_t remaining = tsize - pos_in_tensor;
                uint64_t need = target - flat_pos;
                size_t step = static_cast<size_t>(std::min<uint64_t>(need, remaining));
                pos_in_tensor += step;
                flat_pos      += step;
                if (pos_in_tensor >= tsize) { tensor_idx++; pos_in_tensor = 0; }
            }
        }

        /// Copy `len` elements OUT of params at flat offset `offset` into
        /// `dst` — used when SENDING (building an outgoing chunk).
        void read_into(uint64_t offset, size_t len, float* dst) {
            seek(offset);
            size_t written = 0;
            while (written < len) {
                auto& data = params[tensor_idx]->val.data;
                size_t avail = data.size() - pos_in_tensor;
                size_t take  = std::min(avail, len - written);
                std::copy(data.begin() + pos_in_tensor,
                          data.begin() + pos_in_tensor + take,
                          dst + written);
                written       += take;
                pos_in_tensor += take;
                flat_pos      += take;
                if (pos_in_tensor >= data.size()) { tensor_idx++; pos_in_tensor = 0; }
            }
        }

        /// Copy `len` elements from `src` INTO params at flat offset
        /// `offset` — used when RECEIVING (applying an incoming chunk).
        void write_from(uint64_t offset, const float* src, size_t len) {
            seek(offset);
            size_t written = 0;
            while (written < len) {
                auto& data = params[tensor_idx]->val.data;
                size_t avail = data.size() - pos_in_tensor;
                size_t take  = std::min(avail, len - written);
                std::copy(src + written, src + written + take,
                          data.begin() + pos_in_tensor);
                written       += take;
                pos_in_tensor += take;
                flat_pos      += take;
                if (pos_in_tensor >= data.size()) { tensor_idx++; pos_in_tensor = 0; }
            }
        }
    };

    // ── Network I/O result: elapsed time + bytes actually moved ──────────────
    // bytes comes from io_utils.hpp's global counters (snapshot-diffed around
    // the call), so it reflects real wire bytes including protocol headers,
    // not just sizeof(flat)*N.
    struct NetIoStats {
        double   ms   = 0;     // -1 on failure (receive_weights() only)
        uint64_t bytes = 0;
        double   mb   = 0;
        double   mbps = 0;
    };

    /// Stream student.parameters() to the server in opts.chunk_mb-sized
    /// pieces — no full-model flat buffer is ever built (that's what
    /// trainer.get_flat_weights() would do; this reads straight from the
    /// source tensors instead). This is what keeps peak memory for the
    /// round-trip down to ~chunk_mb regardless of model size, the actual
    /// fix for "client freezes/gets killed after receiving weights" on a
    /// low-RAM machine.
    NetIoStats send_deltas() {
        auto params = student.parameters();
        uint64_t total = 0;
        for (auto& p : params) total += p->val.get_size();
        size_t chunk_elems = std::max<size_t>(
            1, static_cast<size_t>(opts.chunk_mb * 1024.0 * 1024.0 / sizeof(float)));

        FlatParamCursor cursor(params);
        uint64_t before = net_bytes_sent();
        Timer t;
        bool ok = net.sendChunked(total, chunk_elems,
            [&cursor](uint64_t offset, size_t len, float* out) {
                cursor.read_into(offset, len, out);
            });
        NetIoStats r;
        r.ms    = ok ? t.ms() : -1.0;
        r.bytes = net_bytes_sent() - before;
        r.mb    = r.bytes / 1e6;
        r.mbps  = r.ms > 0 ? r.mb / (r.ms / 1000.0) : 0.0;
        return r;
    }

    /// Receive updated global weights in chunks, writing each one straight
    /// into student.parameters() as it arrives — no full-model flat buffer
    /// on this side either. r.ms == -1 on connection drop or a size
    /// mismatch (wrong model architecture for this checkpoint/server).
    NetIoStats receive_weights() {
        auto params = student.parameters();
        uint64_t total = 0;
        for (auto& p : params) total += p->val.get_size();

        FlatParamCursor cursor(params);
        uint64_t before = net_bytes_received();
        Timer t;
        NetIoStats r;
        bool ok;
        try {
            ok = net.receiveChunked(total,
                [&cursor](uint64_t offset, const float* data, size_t len) {
                    cursor.write_from(offset, data, len);
                });
        } catch (const std::exception& e) {
            printf("[client] receive_weights: %s\n", e.what());
            r.ms = -1.0;
            return r;
        }
        if (!ok) { r.ms = -1.0; return r; }
        r.ms    = t.ms();
        r.bytes = net_bytes_received() - before;
        r.mb    = r.bytes / 1e6;
        r.mbps  = r.ms > 0 ? r.mb / (r.ms / 1000.0) : 0.0;
        return r;
    }

    // ── Inference test ────────────────────────────────────────────────────────

    RoundStats run_inference(size_t round_no, double train_ms, double train_loss,
                             double send_ms, double recv_ms,
                             double send_mb = 0, double recv_mb = 0) {
        RoundStats rs;
        rs.round_no       = round_no;
        rs.train_total_ms = train_ms;
        rs.train_loss     = train_loss;
        rs.send_ms        = send_ms;
        rs.recv_ms        = recv_ms;
        rs.send_mb        = send_mb;
        rs.recv_mb        = recv_mb;
        rs.send_mbps      = send_ms > 0 ? send_mb / (send_ms / 1000.0) : 0.0;
        rs.recv_mbps      = recv_ms > 0 ? recv_mb / (recv_ms / 1000.0) : 0.0;

        Timer enc_t;
        auto token_ids  = tokenizer.encode(opts.prompt);
        rs.encode_ms    = enc_t.ms();
        rs.prompt_tokens= token_ids.size();

        std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
        Tensor_t<float> context = make_tensor<float>(
            Matrix<float>(ctx_data, {1, ctx_data.size()}));

        Timer inf_t;
        auto out = student.generate(context, (int)opts.gen_tokens, 0.7f, 40);
        rs.infer_ms = inf_t.ms();

        size_t prompt_len = token_ids.size();
        std::vector<int> gen_ids;
        for (size_t i = prompt_len; i < out->val.data.size(); i++)
            gen_ids.push_back((int)out->val.data[i]);

        Timer dec_t;
        rs.generated  = tokenizer.decode(gen_ids);
        rs.decode_ms  = dec_t.ms();
        rs.gen_tokens = gen_ids.size();

        if (rs.infer_ms > 0)
            rs.tokens_per_sec = (double)rs.gen_tokens / (rs.infer_ms / 1000.0);

        mem.snapshot(MemStage::INFER);
        auto report = mem.report();
        rs.rss_mb = (double)report.infer_rss_mb;

        return rs;
    }

    // ── Checkpointing ──────────────────────────────────────────────────────────
    //
    // The client decides WHEN to save (here); Trainer knows HOW (save_checkpoint /
    // save_quantized_checkpoint, both delegating to student.save() / Quantizer.hpp).
    // Always writes the plain float checkpoint first — that's the one you resume
    // distillation from via --load-path. The quantized copy (if requested) is an
    // additional, smaller artifact alongside it, not a replacement.
    void checkpoint(size_t round_no, bool final_save = false) {
        ensure_parent_dir(opts.save_path);
        trainer.save_checkpoint(opts.save_path);
        printf("[client] Saved student checkpoint -> %s (round %zu%s)\n",
               opts.save_path.c_str(), round_no, final_save ? ", final" : "");

        if (opts.quantize == "none") return;

        std::string qpath = opts.save_path + "." + opts.quantize;
        ensure_parent_dir(qpath);
        if (opts.quantize == "fp8")
            trainer.save_quantized_checkpoint<fp8_e4m3>(qpath);
        else if (opts.quantize == "fp4")
            trainer.save_quantized_checkpoint<fp4_e2m1>(qpath);
        else
            printf("[client] Unknown --quantize format '%s' (expected fp8|fp4) — skipping.\n",
                   opts.quantize.c_str());
    }

    // ── FedAvg round ──────────────────────────────────────────────────────────

    void run_fedavg_round(size_t round_no) {
        printf("\n[client] ── FedAvg Round %zu ────────────────────────────\n", round_no);

        Timer train_t;
        auto get_batch = [&](const std::string& split) {
            return dataset.get_batch(split);
        };
        // trainer.finetune() owns the optimizer loop — client just supplies data.
        double last_loss = trainer.finetune(get_batch, (int)opts.iters,
                                            (int)opts.eval_iters, opts.verbose);
        double train_ms = train_t.ms();
        mem.snapshot(MemStage::TRAIN);

        double send_ms = 0, recv_ms = 0, send_mb = 0, recv_mb = 0;

        if (!opts.no_federated && net.fd() >= 0) {
            // streams student.parameters() in chunks — net.sendChunked()
            auto send_io = send_deltas();
            send_ms = send_io.ms; send_mb = send_io.mb;
            printf("[client] Sent deltas in %.1f ms (%.2f MB, %.1f MB/s)\n",
                   send_ms, send_mb, send_io.mbps);

            // net.receiveChunked() straight into student.parameters()
            auto recv_io = receive_weights();
            if (recv_io.ms < 0) {
                printf("[client] Server disconnected during receive.\n");
                return;
            }
            recv_ms = recv_io.ms; recv_mb = recv_io.mb;
            printf("[client] Received weights in %.1f ms (%.2f MB, %.1f MB/s)\n",
                   recv_ms, recv_mb, recv_io.mbps);
        }

        auto rs = run_inference(round_no, train_ms, last_loss, send_ms, recv_ms,
                                send_mb, recv_mb);
        if (opts.verbose)
            print_round_stats(rs);
    }

    // ── FedDistill round ──────────────────────────────────────────────────────
    //
    // Phase 1: compute student logits on the local batch (proxy batch).
    //          No backward pass, no teacher model. Send logits to server.
    // Phase 2: receive averaged consensus from server.
    //          Pass it to trainer.distill_logits() as the teacher signal.
    //          The client still trains on its OWN local targets — the consensus
    //          just provides the soft-label component of the distillation loss.

    void run_feddistill_round(size_t round_no) {
        printf("\n[client] ── FedDistill Round %zu ─────────────────────────\n", round_no);

        if (opts.no_federated || net.fd() < 0) {
            // No server: fall back to local fine-tuning.
            run_fedavg_round(round_no);
            return;
        }

        // ── Phase 1: forward-only to get proxy logits ─────────────────────────
        auto [proxy_inputs, local_targets] = dataset.get_batch("train");

        // trainer.compute_logits() does student.forward(inputs, nullptr, true),
        // reshapes to {B*S, vocab_size}, resets graph — no grad retained.
        Matrix<float> logit_mat = trainer.compute_logits(proxy_inputs);

        uint64_t n_examples = static_cast<uint64_t>(logit_mat.shape[0]); // B*S
        uint64_t vocab_size = static_cast<uint64_t>(logit_mat.shape[1]); // V

        std::vector<float> flat_logits(logit_mat.data.begin(), logit_mat.data.end());

        uint64_t send_bytes_before = net_bytes_sent();
        Timer send_t;
        // Client::sendLogits — wire: [n_examples][vocab_size][bytes][data]
        if (!net.sendLogits(flat_logits, n_examples, vocab_size)) {
            printf("[client] Failed to send logits — dropping round.\n");
            return;
        }
        double send_ms = send_t.ms();
        double send_mb = (net_bytes_sent() - send_bytes_before) / 1e6;
        printf("[client] Sent logits (%zu examples × %zu vocab) in %.1f ms (%.2f MB, %.1f MB/s)\n",
               (size_t)n_examples, (size_t)vocab_size, send_ms,
               send_mb, send_ms > 0 ? send_mb / (send_ms / 1000.0) : 0.0);

        // ── Phase 2: receive consensus and distill ────────────────────────────
        std::vector<float> consensus_flat;
        uint64_t recv_n = 0, recv_v = 0;

        uint64_t recv_bytes_before = net_bytes_received();
        Timer recv_t;
        // Client::receiveLogits — wire: [n_examples][vocab_size][bytes][data]
        if (!net.receiveLogits(consensus_flat, recv_n, recv_v)) {
            printf("[client] Failed to receive consensus — dropping round.\n");
            return;
        }
        double recv_ms = recv_t.ms();
        double recv_mb = (net_bytes_received() - recv_bytes_before) / 1e6;
        printf("[client] Received consensus in %.1f ms (%.2f MB, %.1f MB/s)\n",
               recv_ms, recv_mb, recv_ms > 0 ? recv_mb / (recv_ms / 1000.0) : 0.0);

        // Reshape flat consensus into a Tensor [B*S, V] — this IS the teacher
        // signal. No teacher model involved; it came from the server over the wire.
        auto teacher_logits = make_tensor<float>(
            Matrix<float>(consensus_flat, {recv_n, recv_v}));

        // trainer.distill_logits() — alpha*hard_CE(local_targets) + (1-alpha)*soft_CE(consensus)
        Timer distill_t;
        float loss = trainer.distill_logits(proxy_inputs, local_targets, teacher_logits);
        printf("[client] Distillation loss: %.4f  (%.1f ms)\n", loss, distill_t.ms());

        mem.snapshot(MemStage::TRAIN);
        auto rs = run_inference(round_no, distill_t.ms(), loss, send_ms, recv_ms,
                                send_mb, recv_mb);
        if (opts.verbose)
            print_round_stats(rs);
    }

    // ── Dispatch ──────────────────────────────────────────────────────────────

    void run_round(size_t round_no) {
        if (opts.feddistill)
            run_feddistill_round(round_no);
        else
            run_fedavg_round(round_no);
    }

public:
    FederatedClient(const ClientOptions& o)
        : hp(GPTp),
          student(loader.load_model(o.model_path, hp)),
          // Trainer: student only. No teacher pointer.
          // FEDAVG mode by default; distill_logits() handles FedDistill
          // server-consensus path without needing a local teacher.
          trainer(student, o.lr, o.feddistill ? FEDDISTILL : FEDAVG),
          dataset(o.dataset_path, o.block_size, o.batch_size),
          opts(o)
    {
        // load_tokenizer() reads metadata populated by load_model() above.
        // Assigning in the body guarantees load_model() has already run.
        tokenizer = loader.load_tokenizer();
        if (tokenizer.encoder.empty())
            throw std::runtime_error(
                "FederatedClient: failed to load tokenizer from GGUF: " + o.model_path);

        dataset.load();

        // ── Resume from a previous checkpoint, if one was requested ───────────
        // Ownership: the client owns the student, so the client is what
        // decides whether to resume — not the server, and not Trainer (which
        // only knows HOW to serialise, via save_checkpoint()/load_checkpoint(),
        // not WHEN). Falls back to the freshly-loaded GGUF weights above if no
        // checkpoint path was given or the file doesn't exist yet.
        if (!opts.load_path.empty()) {
            if (std::filesystem::exists(opts.load_path)) {
                trainer.load_checkpoint(opts.load_path);
                printf("[client] Resumed student from checkpoint: %s\n", opts.load_path.c_str());
            } else {
                printf("[client] --load-path %s not found — starting from %s instead.\n",
                       opts.load_path.c_str(), o.model_path.c_str());
            }
        }
    }

    // ── Hardware profiling ────────────────────────────────────────────────────

    void run_profiler() {
        printf("\n[client] ══ Hardware Profiling ═════════════════════════\n");

        mem.snapshot(MemStage::BASELINE);
        profiler.run(hp.vocab_size, opts.verbose);
        hw_config = profiler.config();
        mem.snapshot(MemStage::LOADED);

        printf("[client] Recommended batch_size : %zu\n",   hw_config.batch_size);
        printf("[client] Recommended block_size : %zu\n",   hw_config.block_size);
        printf("[client] Recommended quant      : %s\n",    quant_name(hw_config.algo.quant));
        printf("[client] Estimated param RAM    : %llu MB\n",
               (unsigned long long)hw_config.estimated_param_mb);

        if (opts.json_log) {
            std::ofstream f("client_profile.json");
            f << profiler.to_json();
            printf("[client] Profile written to client_profile.json\n");
        }
    }

    // ── Main training + federated loop ────────────────────────────────────────

    void run() {
        if (!opts.no_federated) {
            if (!connect_to_server()) {
                printf("[client] Could not connect — running in local mode.\n");
                opts.no_federated = true;
            }
        }

        // Baseline inference before any training
        printf("\n[client] ── Baseline inference (before training) ──────\n");
        {
            Timer enc_t;
            auto ids    = tokenizer.encode(opts.prompt);
            double enc_ms = enc_t.ms();
            printf("[client] Encode  '%s' → %zu tokens  [%.3f ms]\n",
                   opts.prompt.c_str(), ids.size(), enc_ms);

            std::vector<float> ctx(ids.begin(), ids.end());
            Tensor_t<float> context = make_tensor<float>(
                Matrix<float>(ctx, {1, ctx.size()}));

            Timer inf_t;
            auto out = student.generate(context, (int)opts.gen_tokens, 0.7f, 40);
            double inf_ms = inf_t.ms();

            std::vector<int> gen_ids;
            for (size_t i = ids.size(); i < out->val.data.size(); i++)
                gen_ids.push_back((int)out->val.data[i]);

            Timer dec_t;
            std::string decoded = tokenizer.decode(gen_ids);
            double dec_ms = dec_t.ms();

            printf("[client] Infer   %zu tokens  [%.1f ms  →  %.1f tok/s]\n",
                   gen_ids.size(), inf_ms,
                   (double)gen_ids.size() / (inf_ms / 1000.0));
            printf("[client] Decode  [%.3f ms]\n", dec_ms);
            printf("[client] Output: %s\n", decoded.c_str());
        }

        // Federated training rounds
        for (size_t round = 1; opts.max_rounds == 0 || round <= opts.max_rounds; ++round) {
            run_round(round);
            if (opts.json_log)
                mem.print_summary();

            if (opts.save_every > 0 && round % opts.save_every == 0)
                checkpoint(round);
        }

        // "After the distillation process the student model should be saved":
        // this is that point. Only reached when --rounds is finite — with
        // --rounds 0 (the default) the loop above runs forever and this client
        // process is expected to be stopped externally, same as before.
        checkpoint(opts.max_rounds, /*final_save=*/true);

        net.disconnect();
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    ClientOptions opts = parse_args(argc, argv);
    if (opts.help) { print_usage(argv[0]); return 0; }

    printf("═══════════════════════════════════════════════════════\n");
    printf("  TensorF Federated Client\n");
    printf("═══════════════════════════════════════════════════════\n");

    Timer startup;
    printf("[client] Loading model: %s\n", opts.model_path.c_str());
    FederatedClient client(opts);
    printf("[client] Model ready in %s\n", startup.str().c_str());

    if (!opts.no_profile)
        client.run_profiler();

    client.run();
    return 0;
}