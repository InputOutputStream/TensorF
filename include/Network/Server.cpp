/*
 * server.cpp  —  TensorF Federated Learning Server
 * =================================================
 *
 *  Architectural contract:
 *    • Server owns the global model + a Trainer (lr=0, aggregation-only).
 *    • FedDistill logits aggregation math lives in Trainer:
 *        trainer.aggregate_logits(logits)    → FedDistill consensus
 *    • FedAvg weight aggregation is chunked, done directly in Server.hpp/cpp
 *      (NOT Trainer::aggregate(), which expects a fully-materialized
 *      per-client updates[] vector — see Server.hpp's round_accum):
 *        handleClient()         → recv_chunked() straight into round_accum
 *                                  (a running sum across clients), never
 *                                  materializing any one client's full delta.
 *        run_fedavg_round()     → divide round_accum by n_clients, write
 *                                  into model.parameters() — same math as
 *                                  Trainer::aggregate(), computed in place.
 *        broadcast()            → send_chunked() straight from
 *                                  model.parameters(), no flat copy built first.
 *    • handleClientLogits()/broadcastLogits() (FedDistill) are unchanged —
 *      still the original single-shot flat protocol; logits payloads are
 *      smaller and chunking them is a separate, not-yet-done follow-up.
 *    • FederatedServer (this file) adds: profiling, per-round stats, accept loop.
 *
 *  Build:
 *    make server          (SERVER_SRC=include/Network/server.cpp in Makefile)
 *
 *  Usage:
 *    ./bin/server [--port 8080] [--clients 2] [--rounds 10]
 *               [--model SLM/SmolLM2-135M-Instruct-f16.gguf]
 *               [--prompt "Hello world"]
 *               [--feddistill]   run FedDistill rounds instead of FedAvg
 *               [--save-path checkpoints/global/model.tnsf]  FedAvg checkpoint path
 *               [--load-path PATH]  resume the global model from a checkpoint
 *               [--save-every 1]    save every N FedAvg rounds, 0=never
 *               [--quantize none]   also save a compressed copy: none|fp8|fp4
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
#include <filesystem>

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

// ─── TensorF profiler ────────────────────────────────────────────────────────
#include "../Profiler/Profiler.hpp"
#include "../Profiler/HyperparamAdvisor.hpp"

// ─── Server base + Trainer ───────────────────────────────────────────────────
// Server.hpp now includes Trainner.hpp. The server's public `trainer` member
// (Trainer<GPT<T>,T>, lr=0) owns FedDistill's aggregation math:
//   trainer.aggregate_logits(logits)  — FedDistill consensus
// FedAvg weight aggregation is chunked and lives directly in Server.hpp/cpp
// instead (round_accum) — see the file header above for why.
#include "Server.hpp"

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

struct ServerOptions {
    uint16_t    port          = 8080;
    size_t      min_clients   = 1;
    size_t      max_rounds    = 0;       // 0 = run forever
    std::string model_path    = "SLM/gpt2-small-f32.gguf";
    std::string prompt        = "The data type is int";
    size_t      gen_tokens    = 30;
    bool        feddistill    = false;   // run FedDistill rounds
    bool        no_profile    = false;
    bool        json_log      = false;
    bool        verbose       = true;
    bool        help          = false;

    // ── Checkpointing ────────────────────────────────────────────────────
    // The server owns the GLOBAL model (see Server.hpp's ownership contract above),
    // so in FedAvg mode the server is the canonical place this gets saved —
    // it's the one model every client's update was averaged into. (FedDistill
    // is different: the server's model.parameters() are never updated by
    // aggregate_logits(), so there's nothing new to checkpoint there — see
    // the comment in run_feddistill_round() below.)
    std::string save_path     = "checkpoints/global/model.tnsf";
    std::string load_path;               // empty = start from --model GGUF only
    size_t      save_every    = 1;       // save every N FedAvg rounds, 0=never
    std::string quantize      = "none";  // "none" | "fp8" | "fp4" — extra compressed save

    // ── Chunked transfer/aggregation (low-RAM machines) ──────────────────
    // Sizes THIS server's outgoing broadcast chunks. Smaller = less peak
    // memory per broadcast, more per-chunk framing overhead (8 bytes/chunk,
    // negligible above a few hundred KB). Doesn't need to match clients'
    // own --chunk-mb — recv_chunked() reads whatever chunk size the sender
    // announces per chunk. Server-side aggregation peak memory is now
    // O(model_size) regardless of this value (see Server.hpp's round_accum).
    double      chunk_mb      = 8.0;
};

static void print_usage(const char* prog) {
    printf(
        "Usage: %s [options]\n"
        "  --port <n>          Listen port (default: 8080)\n"
        "  --clients <n>       Min clients per federated round (default: 2)\n"
        "  --rounds <n>        Max rounds, 0=infinite (default: 0)\n"
        "  --model <path>      GGUF model file\n"
        "  --prompt <text>     Prompt for test inference after each round\n"
        "  --tokens <n>        Tokens to generate per inference test (default: 30)\n"
        "  --feddistill        Run FedDistill rounds (logit exchange) instead of FedAvg\n"
        "  --save-path <path>  Where to save the global model (default:\n"
        "                      checkpoints/global/model.tnsf). FedAvg only — see\n"
        "                      run_feddistill_round() for why FedDistill doesn't save here.\n"
        "  --load-path <path>  Resume the global model from a checkpoint at startup\n"
        "  --save-every <n>    Save every n FedAvg rounds, 0=never (default: 1)\n"
        "  --quantize <fmt>    Also save a compressed checkpoint: none|fp8|fp4 (default: none)\n"
        "  --chunk-mb <n>      Broadcast chunk size in MB (default: 8). Lower this on a\n"
        "                      low-RAM server; doesn't need to match clients' own --chunk-mb.\n"
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
        if      (a == "--help")        o.help        = true;
        else if (a == "--no-profile")  o.no_profile  = true;
        else if (a == "--json-log")    o.json_log    = true;
        else if (a == "--quiet")       o.verbose     = false;
        else if (a == "--feddistill")  o.feddistill  = true;
        else if (a == "--port"    && i+1 < argc) o.port        = (uint16_t)atoi(argv[++i]);
        else if (a == "--clients" && i+1 < argc) o.min_clients = atoi(argv[++i]);
        else if (a == "--rounds"  && i+1 < argc) o.max_rounds  = atoi(argv[++i]);
        else if (a == "--model"   && i+1 < argc) o.model_path  = argv[++i];
        else if (a == "--prompt"  && i+1 < argc) o.prompt      = argv[++i];
        else if (a == "--tokens"  && i+1 < argc) o.gen_tokens  = atoi(argv[++i]);
        else if (a == "--save-path"  && i+1 < argc) o.save_path  = argv[++i];
        else if (a == "--load-path"  && i+1 < argc) o.load_path  = argv[++i];
        else if (a == "--save-every" && i+1 < argc) o.save_every = atoi(argv[++i]);
        else if (a == "--quantize"   && i+1 < argc) o.quantize   = argv[++i];
        else if (a == "--chunk-mb"   && i+1 < argc) o.chunk_mb   = atof(argv[++i]);
        else { fprintf(stderr, "[server] Unknown option: %s\n", a.c_str()); o.help = true; }
    }
    return o;
}

// ════════════════════════════════════════════════════════════════════════════
//  Per-round stats
// ════════════════════════════════════════════════════════════════════════════

// Per-client bandwidth breakdown for one round: diff each participating fd's
// counters against a snapshot taken right before the round started. Server-
// side only — there's exactly one fd per connected client, and it stays the
// same for that client's entire session (see Server.hpp's persistent
// handleClient loop), so "per fd" already IS "per client" here with no
// protocol change or client-ID handshake needed.
static void print_per_client_bandwidth(const std::vector<int>& fds,
                                       const std::unordered_map<int, NetIoCounters>& before) {
    for (int fd : fds) {
        NetIoCounters now = net_per_fd(fd);
        auto it = before.find(fd);
        NetIoCounters bef = (it != before.end()) ? it->second : NetIoCounters{};
        double recv_mb = (now.received - bef.received) / 1e6;
        double sent_mb = (now.sent     - bef.sent)     / 1e6;
        printf("[server]   client fd=%-3d  recv %6.2f MB   sent %6.2f MB\n",
               fd, recv_mb, sent_mb);
    }
}

struct RoundStats {
    size_t  round_no         = 0;
    size_t  n_clients        = 0;
    double  recv_ms          = 0;
    double  aggregate_ms     = 0;   // renamed from fedavg_ms — covers both modes
    double  broadcast_ms     = 0;
    double  infer_ms         = 0;
    double  encode_ms        = 0;
    double  decode_ms        = 0;
    size_t  prompt_tokens    = 0;
    size_t  generated_tokens = 0;
    double  tokens_per_sec   = 0;
    double  total_round_ms   = 0;
    std::string generated_text;
    std::string mode;               // "FedAvg" | "FedDistill"

    // ── Network bandwidth (this round) ───────────────────────────────────
    // recv  = bytes read from clients while waiting for their updates/logits
    // bcast = bytes written back out during broadcast()/broadcastLogits()
    // Captured via io_utils.hpp's global counters (snapshot-diffed around
    // each phase) — see run_fedavg_round()/run_feddistill_round() below.
    double  recv_mb           = 0;
    double  recv_mbps         = 0;
    double  bcast_mb          = 0;
    double  bcast_mbps        = 0;
};

static void print_round_stats(const RoundStats& s) {
    printf("\n╔══════════════════════════════════════════════════════════╗\n");
    printf("║  [%s] Round %-4zu  |  %zu clients%*s║\n",
           s.mode.c_str(), s.round_no, s.n_clients,
           (int)(28 - (int)s.mode.size()), " ");
    printf("╠══════════════════════════════════════════════════════════╣\n");
    printf("║  Receive updates  : %8.1f ms   (%6.2f MB, %6.1f MB/s)    ║\n",
           s.recv_ms, s.recv_mb, s.recv_mbps);
    printf("║  Aggregate        : %8.1f ms                          ║\n", s.aggregate_ms);
    printf("║  Broadcast        : %8.1f ms   (%6.2f MB, %6.1f MB/s)    ║\n",
           s.broadcast_ms, s.bcast_mb, s.bcast_mbps);
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
//  Inherits from Server<T> (Server.hpp):
//    • model, tokenizer, trainer        (public)
//    • locker, round_cv, round_accum, clients_done,
//      logit_updates, client_sockets,
//      min_clients, param_sizes         (protected)
//    • build_param_layout()             called in Server<T> constructor
//    • handleClient(sock)               dispatched from the thread pool (FedAvg)
//    • handleClientLogits(sock)         dispatched from the thread pool (FedDistill)
//    • broadcast()                      streams model.parameters() out chunked
//    • broadcastLogits(consensus)       sends consensus flat vector to all clients
//    • closeClientSocks()               cleanup after each round
//    • clearLogitUpdates()              cleanup after each FedDistill round
//
//  Adds: profiling, per-round stats, accept loop, round runners.

template<typename T>
class FederatedServer : public Server<T> {

    // Bring inherited protected names into scope (required for template bases).
    using Server<T>::model;
    using Server<T>::tokenizer;
    using Server<T>::trainer;
    using Server<T>::server_fd;
    using Server<T>::pool;
    using Server<T>::locker;
    using Server<T>::round_cv;
    using Server<T>::round_accum;
    using Server<T>::clients_done;
    using Server<T>::logit_updates;
    using Server<T>::client_sockets;
    using Server<T>::min_clients;
    using Server<T>::total_param_elems;
    using Server<T>::param_sizes;
    using Server<T>::round_done_cv;
    using Server<T>::round_generation;

    // ── State added by this subclass ─────────────────────────────────────────
    std::atomic<size_t> total_rounds{0};
    size_t              max_rounds;

    Profiler            profiler;
    MemoryProfiler      mem;
    bool                profiled = false;

    ServerOptions       opts;

    // ── Inference test after each round ──────────────────────────────────────

    RoundStats run_inference_test(const std::string& mode,
                                  size_t round_no, size_t n_clients,
                                  double recv_ms, double aggregate_ms,
                                  double bcast_ms,
                                  double recv_mb, double bcast_mb) {
        RoundStats rs;
        rs.round_no      = round_no;
        rs.n_clients     = n_clients;
        rs.recv_ms       = recv_ms;
        rs.aggregate_ms  = aggregate_ms;
        rs.broadcast_ms  = bcast_ms;
        rs.mode          = mode;

        rs.recv_mb    = recv_mb;
        rs.bcast_mb   = bcast_mb;
        rs.recv_mbps  = recv_ms  > 0 ? recv_mb  / (recv_ms  / 1000.0) : 0.0;
        rs.bcast_mbps = bcast_ms > 0 ? bcast_mb / (bcast_ms / 1000.0) : 0.0;

        Timer enc_t;
        auto token_ids   = tokenizer.encode(opts.prompt);
        rs.encode_ms     = enc_t.ms();
        rs.prompt_tokens = token_ids.size();

        std::vector<float> ctx_data(token_ids.begin(), token_ids.end());
        Tensor_t<T> context = make_tensor<T>(
            Matrix<T>(ctx_data, {1, ctx_data.size()}));

        Timer inf_t;
        auto out     = model.generate(context, (int)opts.gen_tokens, 0.7f, 40);
        rs.infer_ms  = inf_t.ms();

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

        rs.total_round_ms = recv_ms + aggregate_ms + bcast_ms + rs.infer_ms;
        return rs;
    }

    // ── Checkpointing ──────────────────────────────────────────────────────────
    //
    // The server decides WHEN to save (here, right after aggregate() — that's
    // the moment the global model actually changes); Trainer knows HOW
    // (save_checkpoint / save_quantized_checkpoint). Only called from the
    // FedAvg round: in FedDistill, trainer.aggregate_logits() never touches
    // model.parameters(), so there's nothing new here to persist — see the
    // comment in run_feddistill_round() for what a server-side FedDistill
    // checkpoint would require instead.
    void checkpoint(size_t round_no) {
        ensure_parent_dir(opts.save_path);
        trainer.save_checkpoint(opts.save_path);
        printf("[server] Saved global model checkpoint -> %s (round %zu)\n",
               opts.save_path.c_str(), round_no);

        if (opts.quantize == "none") return;

        std::string qpath = opts.save_path + "." + opts.quantize;
        ensure_parent_dir(qpath);
        if (opts.quantize == "fp8")
            trainer.template save_quantized_checkpoint<fp8_e4m3>(qpath);
        else if (opts.quantize == "fp4")
            trainer.template save_quantized_checkpoint<fp4_e2m1>(qpath);
        else
            printf("[server] Unknown --quantize format '%s' (expected fp8|fp4) — skipping.\n",
                   opts.quantize.c_str());
    }


    // ── FedAvg round ──────────────────────────────────────────────────────────
    //
    //  1. Wait until min_clients clients have FULLY submitted (handleClient
    //     notifies once each client's chunked stream finishes). Their data is
    //     already summed into round_accum chunk-by-chunk as it arrived — see
    //     Server.hpp's handleClient() — so there's nothing left to collect.
    //  2. Divide round_accum by n_clients, write straight into
    //     model.parameters() — same math as Trainer::aggregate(), just done
    //     here directly since round_accum (not a per-client updates[] vector)
    //     is what's holding the data now.
    //  3. broadcast() — streams model.parameters() out in chunks.

    void run_fedavg_round() {
        size_t round_no = ++total_rounds;
        Timer  round_t;

        // Wait for enough clients to fully finish their chunked submission.
        // handleClient() (base class) increments clients_done and notifies.
        uint64_t recv_bytes_before = net_bytes_received();
        auto     per_fd_before     = net_per_fd_snapshot();
        size_t n_clients;
        {
            std::unique_lock<std::mutex> lk(locker);
            round_cv.wait(lk, [this]{ return clients_done >= min_clients; });
            n_clients = clients_done;
        }
        double recv_ms   = round_t.ms();
        double recv_mb   = (net_bytes_received() - recv_bytes_before) / 1e6;

        // ── Aggregate: divide the running sum, write into the global model ────
        // Equivalent to Trainer::aggregate(updates)'s
        //   avg[l] = (Σ over clients of updates[c][l]) / n_clients
        // just computed from round_accum (one flat running sum, already built
        // up as chunks arrived) instead of n_clients separate full copies.
        // NOTE: floating-point summation order differs from the old
        // client-by-client sequential sum (chunks from different clients can
        // interleave depending on thread scheduling) — both compute the same
        // sum up to ordinary floating-point rounding, which is not
        // meaningfully different for FedAvg-style averaging.
        Timer agg_t;
        {
            std::lock_guard<std::mutex> lk(locker);
            auto params = model.parameters();
            size_t offset = 0;
            for (size_t i = 0; i < params.size(); ++i) {
                size_t n = param_sizes[i];
                for (size_t j = 0; j < n; ++j)
                    params[i]->val.data[j] = round_accum[offset + j] / static_cast<T>(n_clients);
                offset += n;
            }
            // Reset for the NEXT round before any handleClient thread can
            // wake up and start writing into it again (that only happens
            // after round_generation increments, further below).
            std::fill(round_accum.begin(), round_accum.end(), T(0));
            clients_done = 0;
        }
        double aggregate_ms = agg_t.ms();

        if (opts.save_every > 0 && round_no % opts.save_every == 0)
            checkpoint(round_no);

        mem.snapshot(MemStage::TRAIN);

        // ── Broadcast: streams model.parameters() out in chunks ──────────────
        uint64_t sent_bytes_before = net_bytes_sent();
        double bcast_ms = this->broadcast();
        double bcast_mb = (net_bytes_sent() - sent_bytes_before) / 1e6;

        std::vector<int> round_fds;
        {
            std::lock_guard<std::mutex> lk(locker);
            round_fds = client_sockets;     // copy — about to be cleared below
            client_sockets.clear();
            // Let every handleClient() thread parked in round_done_cv know
            // round_no is done, so each loops back to read its client's
            // NEXT round delta from the same persistent connection.
            round_generation++;
        }
        round_done_cv.notify_all();
        if (opts.verbose) print_per_client_bandwidth(round_fds, per_fd_before);

        auto rs = run_inference_test("FedAvg", round_no, n_clients,
                                     recv_ms, aggregate_ms, bcast_ms,
                                     recv_mb, bcast_mb);
        mem.snapshot(MemStage::INFER);
        if (opts.verbose) print_round_stats(rs);

        maybe_shutdown(round_no);
    }

    // ── FedDistill round ──────────────────────────────────────────────────────
    //
    //  1. Wait until min_clients logit submissions arrive (handleClientLogits
    //     notifies). Each client sent its proxy-batch logits in phase 1.
    //  2. trainer.aggregate_logits(logit_updates) — consensus math in Trainer.
    //  3. broadcastLogits(consensus) — send averaged signal back to all clients
    //     so each can run trainer.distill_logits() locally (phase 2).
    //
    //  Note: the server's model.parameters() are NOT updated in a FedDistill
    //  round — only the consensus signal is exchanged. To also update the
    //  global model you would run a FedAvg round or use a server-side forward
    //  pass, which is an extension left to FederatedServer subclasses.
    //
    //  This is also why checkpoint() (server-side) is only called from
    //  run_fedavg_round(): there is no single canonical "the" model to save
    //  here. In FedDistill, each CLIENT's student is the thing that's
    //  actually trained and personalized — that's what client.cpp's
    //  --save-path checkpoints, one file per client.

    void run_feddistill_round() {
        size_t round_no = ++total_rounds;
        Timer  round_t;

        // Wait for enough logit submissions. handleClientLogits() notifies.
        uint64_t recv_bytes_before = net_bytes_received();
        auto     per_fd_before     = net_per_fd_snapshot();
        size_t n_clients;
        {
            std::unique_lock<std::mutex> lk(locker);
            round_cv.wait(lk, [this]{ return logit_updates.size() >= min_clients; });
            n_clients = logit_updates.size();
        }
        double recv_ms = round_t.ms();
        double recv_mb = (net_bytes_received() - recv_bytes_before) / 1e6;

        // ── Aggregate logits: math lives in Trainer ───────────────────────────
        Timer agg_t;
        std::vector<T> consensus;
        {
            std::lock_guard<std::mutex> lk(locker);
            // trainer.aggregate_logits() averages per-client flat logit vectors
            // into a single consensus vector. Architecture-agnostic: only the
            // proxy-batch × vocab shape needs to match across clients.
            consensus = trainer.aggregate_logits(logit_updates);
        }
        double aggregate_ms = agg_t.ms();

        // ── Broadcast consensus: Server networking, not Trainer ───────────────
        // broadcastLogits() sends [n_examples][vocab_size][bytes][data] to each
        // connected client. Clients will call trainer.distill_logits() with it.
        uint64_t sent_bytes_before = net_bytes_sent();
        double bcast_ms = this->broadcastLogits(consensus);
        double bcast_mb = (net_bytes_sent() - sent_bytes_before) / 1e6;

        this->clearLogitUpdates();
        std::vector<int> round_fds;
        {
            std::lock_guard<std::mutex> lk(locker);
            round_fds = client_sockets;     // copy — about to be cleared below
            client_sockets.clear();
            // Same reasoning as run_fedavg_round(): wake every
            // handleClientLogits() thread parked in round_done_cv so each
            // loops back to read its client's next-round logits.
            round_generation++;
        }
        round_done_cv.notify_all();
        if (opts.verbose) print_per_client_bandwidth(round_fds, per_fd_before);

        // Inference test reflects current global model (unchanged this round
        // in pure FedDistill — the update happened locally on each client).
        auto rs = run_inference_test("FedDistill", round_no, n_clients,
                                     recv_ms, aggregate_ms, bcast_ms,
                                     recv_mb, bcast_mb);
        mem.snapshot(MemStage::INFER);
        if (opts.verbose) print_round_stats(rs);

        maybe_shutdown(round_no);
    }

    // ── Round dispatch ────────────────────────────────────────────────────────

    void run_round() {
        if (opts.feddistill)
            run_feddistill_round();
        else
            run_fedavg_round();
    }

    void maybe_shutdown(size_t round_no) {
        if (max_rounds > 0 && round_no >= max_rounds) {
            printf("[server] Reached max_rounds=%zu. Shutting down.\n", max_rounds);
            if (opts.json_log)
                mem.print_summary();
            ::close(server_fd);
            server_fd = -1;
            exit(0);
        }
    }

    // Hardcoded hyperparams — kept as a local member so FederatedServer can
    // pass them to the Server<T> base constructor without depending on a global.
    // NOTE: this member initialises AFTER the base class constructor runs
    // (C++ always initialises base before members). The base constructor
    // receives the VALUE of hp via the init-list expression below, which
    // evaluates the aggregate literal directly — it does NOT read this member.
    // Concretely: `Server<T>(o.model_path, {50257,768,1024,12,12}, ...)` is safe.
    GPT2HyperParams hp {
            .vocab_size = 50257,
            .d_model    = 768,
            .block_size = 1024,
            .n_layer    = 12,
            .n_head     = 12
    };

public:
    FederatedServer(const ServerOptions& o)
        : Server<T>(
            o.model_path,
            // Pass the hyperparams as a literal so we don't depend on `hp`
            // being initialised (it isn't yet when the base ctor runs).
            GPT2HyperParams{50257, 768, 1024, 12, 12},
            o.min_clients,
            o.chunk_mb),
          max_rounds(o.max_rounds),
          opts(o)
    {
        // Ownership: the server owns the global model, so the server decides
        // whether to resume it — same reasoning as the client's --load-path,
        // mirrored on the other side of the FedAvg round. Falls back to the
        // GGUF weights the base Server<T> constructor already loaded above
        // if no checkpoint path was given or the file doesn't exist yet.
        if (!opts.load_path.empty()) {
            if (std::filesystem::exists(opts.load_path)) {
                trainer.load_checkpoint(opts.load_path);
                printf("[server] Resumed global model from checkpoint: %s\n",
                       opts.load_path.c_str());
            } else {
                printf("[server] --load-path %s not found — starting from %s instead.\n",
                       opts.load_path.c_str(), o.model_path.c_str());
            }
        }
    }

    // ── Hardware profiling ────────────────────────────────────────────────────

    void run_profiler() {
        printf("\n[server] ══ Hardware Profiling ══════════════════════════\n");

        mem.snapshot(MemStage::BASELINE);
        profiler.run(hp.vocab_size, opts.verbose);
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

        int opt_val = 1;
        ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt_val, sizeof(opt_val));

        sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(opts.port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (::bind(server_fd, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("bind"); exit(1);
        }
        ::listen(server_fd, 32);
        printf("[server] Listening on port %u (min_clients=%zu, mode=%s)\n",
               opts.port, min_clients,
               opts.feddistill ? "FedDistill" : "FedAvg");

        // Round runner on a background thread — blocks until min_clients arrive.
        std::thread round_thread([this]() {
            while (true) run_round();
        });
        round_thread.detach();

        // Accept loop: dispatch each connection to the thread pool.
        while (server_fd >= 0) {
            int client_fd = ::accept(server_fd, nullptr, nullptr);
            if (client_fd < 0) continue;
            printf("[server] New connection fd=%d\n", client_fd);

            if (opts.feddistill) {
                pool->enqueue([this, client_fd]() {
                    this->handleClientLogits(client_fd);   // ← Server<T>::handleClientLogits()
                });
            } else {
                pool->enqueue([this, client_fd]() {
                    this->handleClient(client_fd);          // ← Server<T>::handleClient()
                });
            }
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