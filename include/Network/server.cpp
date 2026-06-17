/*
 * server.cpp  —  TensorF Federated Learning Server
 * =================================================
 *
 *  Features integrated:
 *    • Profiler     — runs once at startup, fingerprints the server hardware
 *    • MemoryProfiler — tracks RSS before / after model load, during rounds
 *    • LlamaTokenizer  — encodes prompts, decodes generated tokens for logs
 *    • WallClock      — measures per-round latency, encode/decode time,
 *                       broadcast time, tokens/sec
 *    • FedAvg         — aggregates client gradients, broadcasts updated weights
 *
 *  FederatedServer<T> inherits Server<T> (Server.hpp) and calls:
 *    build_param_layout()   — caches model parameter sizes (base, constructor)
 *    handleClient(sock)     — receives one client's update (base, thread pool)
 *    federatedAverage()     — averages all received updates (base, run_round)
 *    applyUpdate(avg)       — writes avg into model.parameters() (base, run_round)
 *    broadcast()            — sends updated weights to all clients (base, run_round)
 *
 *  Build:
 *    make server          (SERVER_SRC=include/Network/server.cpp in Makefile)
 *
 *  Usage:
 *    ./bin/server [--port 8080] [--clients 2] [--rounds 10]
 *               [--model SLM/SmolLM2-135M-Instruct-f16.gguf]
 *               [--vocab SLM/llama-tokenizer/vocab.json]
 *               [--merges SLM/llama-tokenizer/merges.txt]
 *               [--prompt "Hello world"]
 *               [--no-profile]   skip hardware profiling
 *               [--json-log]     write profiler JSON to server_profile.json
 */

// ─── Standard library ────────────────────────────────────────────────────────
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <iomanip>

// ─── POSIX networking ────────────────────────────────────────────────────────
#include <unistd.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>

// ─── TensorF core ────────────────────────────────────────────────────────────
#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"
#include "Modules/Transformer/Llama/Llama.hpp"
#include "ModelLoader/LlamaLoader.hpp"
#include "Tokenizer/LlamaTokenizer.hpp"

// ─── TensorF profiler ────────────────────────────────────────────────────────
#include "Profiler/Profiler.hpp"
#include "Profiler/HyperparamAdvisor.hpp"

// ─── Server base class ───────────────────────────────────────────────────────
// Server.hpp includes ThreadPool.hpp and io_utils.hpp internally.
#include "Network/Server.hpp"

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

struct ServerOptions {
    uint16_t    port          = 8080;
    size_t      min_clients   = 2;
    size_t      max_rounds    = 0;       // 0 = run forever
    std::string model_path    = "SLM/SmolLM2-135M-Instruct-f16.gguf";
    std::string vocab_path    = "SLM/llama-tokenizer/vocab.json";
    std::string merges_path   = "SLM/llama-tokenizer/merges.txt";
    std::string prompt        = "The data type is int";
    size_t      gen_tokens    = 30;
    bool        no_profile    = false;
    bool        json_log      = false;
    bool        verbose       = true;
    bool        help          = false;
};

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --port <n>          Listen port (default: 8080)\n"
        "  --clients <n>       Min clients per federated round (default: 2)\n"
        "  --rounds <n>        Max rounds, 0=infinite (default: 0)\n"
        "  --model <path>      GGUF model file\n"
        "  --vocab <path>      vocab.json\n"
        "  --merges <path>     merges.txt\n"
        "  --prompt <text>     Prompt for test inference after each round\n"
        "  --tokens <n>        Tokens to generate per inference test (default: 30)\n"
        "  --no-profile        Skip hardware profiling at startup\n"
        "  --json-log          Write profiler JSON to server_profile.json\n"
        "  --quiet             Suppress verbose output\n"
        "  --help              Show this help\n",
        prog);
}

static ServerOptions parse_args(int argc, char* argv[]) {
    ServerOptions o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if      (a == "--help")       o.help       = true;
        else if (a == "--no-profile") o.no_profile = true;
        else if (a == "--json-log")   o.json_log   = true;
        else if (a == "--quiet")      o.verbose    = false;
        else if (a == "--port"    && i+1 < argc) o.port        = (uint16_t)atoi(argv[++i]);
        else if (a == "--clients" && i+1 < argc) o.min_clients = atoi(argv[++i]);
        else if (a == "--rounds"  && i+1 < argc) o.max_rounds  = atoi(argv[++i]);
        else if (a == "--model"   && i+1 < argc) o.model_path  = argv[++i];
        else if (a == "--vocab"   && i+1 < argc) o.vocab_path  = argv[++i];
        else if (a == "--merges"  && i+1 < argc) o.merges_path = argv[++i];
        else if (a == "--prompt"  && i+1 < argc) o.prompt      = argv[++i];
        else if (a == "--tokens"  && i+1 < argc) o.gen_tokens  = atoi(argv[++i]);
        else { fprintf(stderr, "[server] Unknown option: %s\n", a.c_str()); o.help = true; }
    }
    return o;
}

// ════════════════════════════════════════════════════════════════════════════
//  Per-round stats
// ════════════════════════════════════════════════════════════════════════════

struct RoundStats {
    size_t  round_no         = 0;
    size_t  n_clients        = 0;
    double  recv_ms          = 0;
    double  fedavg_ms        = 0;
    double  broadcast_ms     = 0;
    double  infer_ms         = 0;
    double  encode_ms        = 0;
    double  decode_ms        = 0;
    size_t  prompt_tokens    = 0;
    size_t  generated_tokens = 0;
    double  tokens_per_sec   = 0;
    double  total_round_ms   = 0;
    std::string generated_text;
};

static void print_round_stats(const RoundStats& s) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  Round %-4zu  |  %zu clients                              ║\n",
           s.round_no, s.n_clients);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Receive updates  : %8.1f ms                          ║\n", s.recv_ms);
    printf("║  FedAvg           : %8.1f ms                          ║\n", s.fedavg_ms);
    printf("║  Broadcast        : %8.1f ms                          ║\n", s.broadcast_ms);
    printf("║  Encode (prompt)  : %8.3f ms  [%zu tokens]            ║\n",
           s.encode_ms, s.prompt_tokens);
    printf("║  Inference        : %8.1f ms  [%zu tokens]            ║\n",
           s.infer_ms, s.generated_tokens);
    printf("║  Decode (output)  : %8.3f ms                          ║\n", s.decode_ms);
    printf("║  Tokens/sec       : %8.1f                             ║\n", s.tokens_per_sec);
    printf("║  Round total      : %8.1f ms                          ║\n", s.total_round_ms);
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Generated: %-44s║\n", s.generated_text.substr(0, 44).c_str());
    if (s.generated_text.size() > 44)
        printf("║             %-44s║\n", s.generated_text.substr(44, 44).c_str());
    printf("╚══════════════════════════════════════════════════════════╝\n");
}

// ════════════════════════════════════════════════════════════════════════════
//  FederatedServer<T>  —  inherits Server<T>
// ════════════════════════════════════════════════════════════════════════════
//
//  Inherits from Server<T> (Server.hpp) to reuse:
//    • model, tokenizer, locker, round_cv, updates, client_sockets,
//      param_sizes, total_param_elems, min_clients  (protected members)
//    • build_param_layout()   called in Server<T> constructor
//    • handleClient(sock)     dispatched from the thread pool
//    • federatedAverage()     called each round
//    • applyUpdate(avg)       called each round
//    • broadcast()            called each round
//
//  Adds: profiling, per-round stats, and the main accept-loop / round-runner.

template<typename T>
class FederatedServer : public Server<T> {

    // Bring inherited protected names into scope (required for template bases).
    using Server<T>::model;
    using Server<T>::tokenizer;
    using Server<T>::server_fd;
    using Server<T>::pool;
    using Server<T>::locker;
    using Server<T>::round_cv;
    using Server<T>::updates;
    using Server<T>::client_sockets;
    using Server<T>::min_clients;
    using Server<T>::total_param_elems;

    // ── State added by this subclass ─────────────────────────────────────────
    std::atomic<size_t> total_rounds{0};
    size_t              max_rounds;

    Profiler            profiler;
    MemoryProfiler      mem;
    bool                profiled = false;

    ServerOptions       opts;

    // ── Inference test after each round ──────────────────────────────────────

    RoundStats run_inference_test(size_t round_no, size_t n_clients,
                                  double recv_ms, double fedavg_ms, double bcast_ms) {
        RoundStats rs;
        rs.round_no     = round_no;
        rs.n_clients    = n_clients;
        rs.recv_ms      = recv_ms;
        rs.fedavg_ms    = fedavg_ms;
        rs.broadcast_ms = bcast_ms;

        // Encode prompt
        Timer enc_t;
        auto token_ids   = tokenizer.encode(opts.prompt);
        rs.encode_ms     = enc_t.ms();
        rs.prompt_tokens = token_ids.size();

        // Build context tensor
        std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
        Tensor_t<T> context = make_tensor<T>(
            Matrix<T>(ctx_data, {1, ctx_data.size()}));

        // Generate
        Timer inf_t;
        auto out     = model.generate(context, (int)opts.gen_tokens, 0.7f, 40);
        rs.infer_ms  = inf_t.ms();

        // Decode
        size_t prompt_len = token_ids.size();
        std::vector<int> gen_ids;
        for (size_t i = prompt_len; i < out->val.data.size(); i++)
            gen_ids.push_back((int)out->val.data[i]);

        Timer dec_t;
        rs.generated_text    = tokenizer.decode(gen_ids);
        rs.decode_ms         = dec_t.ms();
        rs.generated_tokens  = gen_ids.size();

        if (rs.infer_ms > 0)
            rs.tokens_per_sec = (double)rs.generated_tokens / (rs.infer_ms / 1000.0);

        rs.total_round_ms = recv_ms + fedavg_ms + bcast_ms + rs.infer_ms;
        return rs;
    }

    // ── One federated round ───────────────────────────────────────────────────

    void run_round() {
        size_t round_no = ++total_rounds;
        Timer  round_t;

        // Wait for min_clients updates — handleClient() (from Server<T>) notifies.
        size_t n_clients;
        {
            std::unique_lock<std::mutex> lk(locker);
            round_cv.wait(lk, [this]{ return updates.size() >= min_clients; });
            n_clients = updates.size();
        }
        double recv_ms = round_t.ms();

        // FedAvg: calls Server<T>::federatedAverage() + applyUpdate()
        Timer avg_t;
        {
            std::lock_guard<std::mutex> lk(locker);
            auto avg = this->federatedAverage();   // ← Server<T>::federatedAverage()
            this->applyUpdate(avg);                // ← Server<T>::applyUpdate()
            updates.clear();
        }
        double fedavg_ms = avg_t.ms();

        mem.snapshot(MemStage::TRAIN);

        // Broadcast: calls Server<T>::broadcast()
        double bcast_ms = this->broadcast();       // ← Server<T>::broadcast()

        {
            std::lock_guard<std::mutex> lk(locker);
            client_sockets.clear();
        }

        // Inference test + stats
        auto rs = run_inference_test(round_no, n_clients, recv_ms, fedavg_ms, bcast_ms);
        mem.snapshot(MemStage::INFER);

        if (opts.verbose)
            print_round_stats(rs);

        // Stop after max_rounds if set
        if (max_rounds > 0 && round_no >= max_rounds) {
            printf("[server] Reached max_rounds=%zu. Shutting down.\n", max_rounds);
            if (opts.json_log)
                mem.print_summary();
            ::close(server_fd);
            server_fd = -1;
            exit(0);
        }
    }

public:
    FederatedServer(const ServerOptions& o)
        : Server<T>(
            o.model_path,
            o.vocab_path,
            o.merges_path,
            LlamaHyperParams {
                .vocab_size = 50257,
                .d_model    = 768,
                .block_size = 1024,
                .n_layer    = 12,
                .n_head     = 12
            },
            o.min_clients),
          max_rounds(o.max_rounds),
          opts(o)
    {}

    // ── Hardware profiling ────────────────────────────────────────────────────

    void run_profiler() {
        printf("\n[server] ══ Hardware Profiling ══════════════════════════\n");

        mem.snapshot(MemStage::BASELINE);
        profiler.run(this->hp.vocab_size, opts.verbose);
        mem.snapshot(MemStage::LOADED);
        profiled = true;

        if (opts.json_log) {
            std::ofstream f("server_profile.json");
            f << profiler.to_json();
            printf("[server] Profile written to server_profile.json\n");
        }
    }

    // ── Main accept loop ──────────────────────────────────────────────────────

    void start() {
        server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0) { perror("socket"); exit(1); }

        int opt = 1;
        ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(opts.port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (::bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind"); exit(1);
        }
        ::listen(server_fd, 32);
        printf("[server] Listening on port %u (min_clients=%zu)\n",
               opts.port, min_clients);

        // Round runner on a background thread.
        std::thread round_thread([this]() {
            while (true) run_round();
        });
        round_thread.detach();

        // Accept loop: dispatch each connection to the thread pool.
        // handleClient() is inherited from Server<T>.
        while (server_fd >= 0) {
            int client = ::accept(server_fd, nullptr, nullptr);
            if (client < 0) continue;
            printf("[server] New connection fd=%d\n", client);
            pool.enqueue([this, client]() {
                this->handleClient(client);   // ← Server<T>::handleClient()
            });
        }
    }
};

// ════════════════════════════════════════════════════════════════════════════
//  main
// ════════════════════════════════════════════════════════════════════════════

int main(int argc, char* argv[]) {
    ServerOptions opts = parse_args(argc, argv);
    if (opts.help) { print_usage(argv[0]); return 0; }

    printf("═══════════════════════════════════════════════════════\n");
    printf("  TensorF Federated Server\n");
    printf("═══════════════════════════════════════════════════════\n");

    Timer startup;
    printf("[server] Loading model: %s\n", opts.model_path.c_str());
    FederatedServer<float> server(opts);
    printf("[server] Model ready in %s\n", startup.str().c_str());

    if (!opts.no_profile)
        server.run_profiler();

    server.start();
    return 0;
}