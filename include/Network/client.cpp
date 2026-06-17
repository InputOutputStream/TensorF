/* client.cpp  —  TensorF Federated Learning Client
 * ==================================================
 *
 *  Features integrated:
 *    • Profiler     — runs at startup, fingerprints client hardware, advises
 *                     hyperparams (batch_size, block_size, quant policy)
 *    • MemoryProfiler — tracks RSS at baseline / after load / during train /
 *                       during infer
 *    • LlamaTokenizer / GPT2Tokenizer — encodes prompt, decodes output with
 *                       wall-clock measurements
 *    • WallClock      — measures per-iteration training time, encode/decode
 *                       latency, send/receive time, tokens/sec
 *    • FedAvg client  — trains on local data, sends deltas, receives updated
 *                       global weights
 *
 *  Networking is delegated entirely to Client<float> (Client.hpp):
 *    connect_to_server()  → net.connect_to_server()
 *    send_deltas()        → net.send(flat_params)
 *    receive_weights()    → net.receive(flat) + unpack into model.parameters()
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
 *                [--no-profile]  skip hardware profiling
 *                [--json-log]    write profiler JSON to client_profile.json
 *                [--no-federated]  train only locally, no server connection
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

// ─── POSIX networking ────────────────────────────────────────────────────────
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

// ─── TensorF core ────────────────────────────────────────────────────────────
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"

// Llama model + loader
#include "Modules/Transformer/Llama/Llama.hpp"
#include "ModelLoader/LlamaLoader.hpp"
#include "Tokenizer/LlamaTokenizer.hpp"

// Dataset
#include "DataLoader/DataLoading.hpp"

// ─── TensorF profiler ────────────────────────────────────────────────────────
#include "Profiler/Profiler.hpp"
#include "Profiler/HyperparamAdvisor.hpp"

// ─── Network utilities + Client base class ───────────────────────────────────
// Client.hpp includes io_utils.hpp internally; do not include io_utils.hpp
// separately here to avoid duplicate-definition errors.
#include "Network/Client.hpp"

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
//  CLI options
// ════════════════════════════════════════════════════════════════════════════

struct ClientOptions {
    std::string server_ip    = "127.0.0.1";
    uint16_t    port         = 8080;
    std::string model_path   = "SLM/SmolLM2-135M-Instruct-f16.gguf";
    std::string dataset_path = "Dataset";
    size_t      iters        = 50;
    size_t      eval_iters   = 10;
    size_t      batch_size   = 4;
    size_t      block_size   = 512;
    std::string prompt       = "the data type is int";
    size_t      gen_tokens   = 40;
    bool        no_profile   = false;
    bool        json_log     = false;
    bool        no_federated = false;
    bool        verbose      = true;
    bool        help         = false;
};


static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --server <ip>       Server IP (default: 127.0.0.1)\n"
        "  --port <n>          Server port (default: 8080)\n"
        "  --model <path>      GGUF model file (Llama/SmolLM2)\n"
        "  --dataset <path>    Text dataset directory\n"
        "  --iters <n>         Training iterations per round (default: 50)\n"
        "  --eval <n>          Eval interval (default: 10)\n"
        "  --batch <n>         Batch size (default: 4)\n"
        "  --block <n>         Block size / context length (default: 512)\n"
        "  --prompt <text>     Test prompt for inference after each round\n"
        "  --tokens <n>        Tokens to generate (default: 40)\n"
        "  --no-profile        Skip hardware profiling\n"
        "  --json-log          Write profiler JSON to client_profile.json\n"
        "  --no-federated      Local training only, skip server connection\n"
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
        else if (a == "--quiet")        o.verbose      = false;
        else if (a == "--server"  && i+1 < argc) o.server_ip   = argv[++i];
        else if (a == "--port"    && i+1 < argc) o.port        = (uint16_t)atoi(argv[++i]);
        else if (a == "--model"   && i+1 < argc) o.model_path  = argv[++i];
        else if (a == "--dataset" && i+1 < argc) o.dataset_path= argv[++i];
        else if (a == "--iters"   && i+1 < argc) o.iters       = atoi(argv[++i]);
        else if (a == "--eval"    && i+1 < argc) o.eval_iters  = atoi(argv[++i]);
        else if (a == "--batch"   && i+1 < argc) o.batch_size  = atoi(argv[++i]);
        else if (a == "--block"   && i+1 < argc) o.block_size  = atoi(argv[++i]);
        else if (a == "--prompt"  && i+1 < argc) o.prompt      = argv[++i];
        else if (a == "--tokens"  && i+1 < argc) o.gen_tokens  = atoi(argv[++i]);
        else { fprintf(stderr, "[client] Unknown option: %s\n", a.c_str()); o.help = true; }
    }
    return o;
}

// ════════════════════════════════════════════════════════════════════════════
//  Per-iteration and per-round stats
// ════════════════════════════════════════════════════════════════════════════

struct IterStats {
    size_t iter;
    double train_ms;
    double loss;
};

struct RoundStats {
    size_t  round_no        = 0;
    double  train_total_ms  = 0;
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
};

static void print_round_stats(const RoundStats& s) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  CLIENT  Round %-4zu                                      ║\n", s.round_no);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Training total   : %8.1f ms                          ║\n", s.train_total_ms);
    printf("║  Send deltas      : %8.1f ms                          ║\n", s.send_ms);
    printf("║  Receive weights  : %8.1f ms                          ║\n", s.recv_ms);
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
//  Networking is handled entirely by the Client<float> member `net`.
//  This class only owns the model, tokenizer, dataset, and profiler.

class FederatedClient {
    // ── Model ────────────────────────────────────────────────────────────────
    LlamaHyperParams       hp;
    LlamaGGUFLoader<float> loader;
    Llama<float>           model;
    LlamaTokenizer         tokenizer;

    // ── Dataset ──────────────────────────────────────────────────────────────
    TextDataset<float>     dataset;

    // ── Networking — delegated to Client<float> ──────────────────────────────
    Client<float>          net;   // ← from Client.hpp
    ClientOptions          opts;

    // ── Profiler ─────────────────────────────────────────────────────────────
    Profiler               profiler;
    MemoryProfiler         mem;
    HyperparamConfig       hw_config;

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// Total elements across all model parameter tensors.
    size_t total_param_elems() const {
        size_t n = 0;
        for (auto& p : model.parameters()) n += p->val.get_size();
        return n;
    }

    // ── Network helpers — all delegate to net (Client<float>) ────────────────

    /// Establish connection; delegates to Client<float>::connect_to_server().
    bool connect_to_server() {
        bool ok = net.connect_to_server(opts.server_ip, opts.port);
        if (ok)
            printf("[client] Connected to %s:%u\n", opts.server_ip.c_str(), opts.port);
        return ok;
    }

    /// Flatten model parameters and send via Client<float>::send().
    /// Returns elapsed milliseconds.
    double send_deltas() {
        std::vector<float> flat;
        flat.reserve(total_param_elems());
        for (auto& p : model.parameters())
            flat.insert(flat.end(), p->val.data.begin(), p->val.data.end());

        Timer t;
        net.send(flat);   // ← Client<float>::send([uint64 bytes][data...])
        return t.ms();
    }

    /// Receive updated global weights via Client<float>::receive(), then
    /// unpack the flat buffer back into model.parameters().
    /// Returns elapsed milliseconds, or -1 on connection drop.
    double receive_weights() {
        std::vector<float> flat;
        Timer t;
        if (!net.receive(flat)) return -1.0;   // ← Client<float>::receive()
        double recv_ms = t.ms();

        auto params  = model.parameters();
        size_t offset = 0;
        for (auto& p : params) {
            size_t n = p->val.get_size();
            if (offset + n > flat.size()) break;
            std::copy(flat.begin() + offset, flat.begin() + offset + n,
                      p->val.data.begin());
            offset += n;
        }
        return recv_ms;
    }

    // ── Inference test ────────────────────────────────────────────────────────

    RoundStats run_inference(size_t round_no, double train_ms,
                             double send_ms, double recv_ms) {
        RoundStats rs;
        rs.round_no       = round_no;
        rs.train_total_ms = train_ms;
        rs.send_ms        = send_ms;
        rs.recv_ms        = recv_ms;

        // Encode
        Timer enc_t;
        auto token_ids  = tokenizer.encode(opts.prompt);
        rs.encode_ms    = enc_t.ms();
        rs.prompt_tokens= token_ids.size();

        // Build context tensor
        std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
        Tensor_t<float> context = make_tensor<float>(
            Matrix<float>(ctx_data, {1, ctx_data.size()}));

        // Generate
        Timer inf_t;
        auto out = model.generate(context, (int)opts.gen_tokens, 0.7f, 40);
        rs.infer_ms = inf_t.ms();

        size_t prompt_len = token_ids.size();
        std::vector<int> gen_ids;
        for (size_t i = prompt_len; i < out->val.data.size(); i++)
            gen_ids.push_back((int)out->val.data[i]);

        // Decode
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

    // ── One federated round ───────────────────────────────────────────────────

    void run_round(size_t round_no) {
        printf("\n[client] ── Round %zu ──────────────────────────────────\n", round_no);

        // Train on local data
        Timer train_t;
        double last_loss = 0.0;

        auto get_batch = [&](const std::string& split) {
            return dataset.get_batch(split);
        };

        Optimizer<float> op(model.parameters(), 1e-4f, ADAMw);

        for (size_t it = 0; it < opts.iters; it++) {
            auto [inputs, targets] = get_batch("train");
            op.zero_grad();
            auto loss = model.forward(inputs, targets, true);
            loss->backward(Matrix<float>(1.0f));
            op.step();
            last_loss = loss->val.data[0];
            loss->reset_graph();

            if (opts.verbose && (it % opts.eval_iters == 0)) {
                printf("[client]   iter=%zu  loss=%.4f  elapsed=%s\n",
                       it, last_loss, train_t.str().c_str());
            }
        }
        double train_ms = train_t.ms();
        mem.snapshot(MemStage::TRAIN);

        double send_ms = 0, recv_ms = 0;

        // Only attempt federated sync when a server connection is active.
        // net.fd() returns the raw socket fd; -1 means disconnected.
        if (!opts.no_federated && net.fd() >= 0) {
            send_ms = send_deltas();   // Client<float>::send()
            printf("[client] Sent deltas in %.1f ms\n", send_ms);

            recv_ms = receive_weights();   // Client<float>::receive()
            if (recv_ms < 0) {
                printf("[client] Server disconnected during receive.\n");
                return;
            }
            printf("[client] Received weights in %.1f ms\n", recv_ms);
        }

        auto rs = run_inference(round_no, train_ms, send_ms, recv_ms);
        if (opts.verbose)
            print_round_stats(rs);
    }

public:
    FederatedClient(const ClientOptions& o)
        : hp { .vocab_size = 49152, .input_dim = 576, .block_size = (o.block_size),
               .n_heads = 9, .n_kv_heads = 3, .n_layer = 30, .ffn_hidden = 1536 },
          model(loader.load_model(o.model_path, hp)),
          dataset(o.dataset_path, o.block_size, o.batch_size),
          opts(o)
    {
        dataset.load();
        // tokenizer populated by LlamaGGUFLoader; copy here if loader exposes it:
        // tokenizer = loader.tokenizer;
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
            // connect_to_server() calls Client<float>::connect_to_server()
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
            auto out = model.generate(context, (int)opts.gen_tokens, 0.7f, 40);
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
        for (size_t round = 1; ; ++round) {
            run_round(round);
            if (opts.json_log)
                mem.print_summary();
        }

        // Disconnect via Client<float>::disconnect()
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