#pragma once

#include "ThreadPool.hpp"
#include "io_utils.hpp"
#include "Trainner.hpp"   // ← Trainer owns all aggregation impl

#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <algorithm>

#include "Types/types.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"

#include "Modules/Transformer/GPT/GPT.hpp"
#include "ModelLoader/GPTLoader.hpp"
#include "Tokenizer/GPT2Tokenizer.hpp"

// ── Server<T> ─────────────────────────────────────────────────────────────────
//
// Base class for federated learning servers. Owns:
//   • GPT<T, LinearT> model          — global model (weights averaged each FedAvg round;
//                              also supplies the teacher signal in FedDistill if
//                              needed, though in pure FedDistill the consensus is
//                              the averaged client logits, not this model's output)
//   • Trainer<GPT<T, LinearT>,T> trainer — owns FedDistill's aggregation math
//                              (aggregate_logits()) plus checkpointing.
//                              FedAvg weight aggregation is chunked and
//                              lives directly in this class instead
//                              (round_accum, below) — Trainer's own
//                              aggregate()/get_flat_weights()/
//                              set_flat_weights() still exist (e.g. for
//                              round-trip tests) but the FedAvg round
//                              path here doesn't call them.
//   • TCP listener socket + ThreadPool for concurrent client connections
//   • FedAvg state: round_accum (running-sum accumulator), client_sockets[], param layout
//
// Responsibility split:
//   Server   — pure networking + socket/round lifecycle + chunked FedAvg
//              aggregation (round_accum).
//   Trainer  — FedDistill consensus math, checkpointing/serialisation.
//   FederatedServer (server.cpp) — profiling, per-round stats, accept loop.
//
// Access policy:
//   public    — interface used from main() and FederatedServer
//   protected — called by FederatedServer subclass

template <typename T, template<typename> class LinearT>
class Server {

protected:
    // ── Model loader + tokenizer ─────────────────────────────────────────────
    GPT2HyperParams   hp;
    // GGUF files always store dense pretrained weights (see GPTGGUFLoader's
    // static_assert) — regardless of LinearT, we always load a dense
    // GPT<T,Linear> from disk and copy it into `model` below (see
    // constructor), so this loader is fixed at LinearT=Linear.
    GPTGGUFLoader<T, Linear> loader;
    GPT2Tokenizer     tokenizer;

    // ── Networking ───────────────────────────────────────────────────────────
    int    server_fd = -1;
    // IMPORTANT: handleClient()/handleClientLogits() now loop for the entire
    // lifetime of each client connection (see round_done_cv below), so every
    // connected client permanently occupies one pool thread for its whole
    // session — this is no longer "however many short-lived reads happen to
    // overlap." Sized to min_clients (+ a little headroom) in the constructor
    // body below, once min_clients is actually known — a flat default here
    // would either starve a larger cluster or, as a flat 16, sit mostly idle
    // for a small one. unique_ptr because ThreadPool has no default ctor and
    // isn't movable/copyable (owns std::thread/mutex/cv members), so it can't
    // be a plain member built before min_clients exists.
    std::unique_ptr<ThreadPool> pool;

    // ── FedAvg round state ────────────────────────────────────────────────────
    // round_accum/clients_done replace what used to be `updates` (a vector
    // holding every client's COMPLETE delta tensors simultaneously — peak
    // memory O(n_clients * model_size)). Each client's chunks are now added
    // directly into ONE shared running-sum accumulator as they arrive (see
    // handleClient()), so peak memory here is O(model_size) regardless of
    // n_clients — this is the aggregation half of low-RAM chunking; the
    // transfer half is send_chunked()/recv_chunked() in io_utils.hpp.
    std::mutex                            locker;
    std::condition_variable               round_cv;
    std::vector<T>                        round_accum;     // flat, size = total_param_elems
    size_t                                clients_done = 0;
    std::vector<int>                      client_sockets;
    size_t                                min_clients = 2;

    // Chunk size (in elements of T) used by THIS process's outgoing chunked
    // sends (server → client broadcast). The receiving side doesn't need to
    // match this — recv_chunked() just reads whatever chunk_len the sender
    // announces per chunk — so a low-RAM client and a beefy server can each
    // pick their own value independently. See ServerOptions::chunk_mb.
    size_t                                chunk_elems = (8ull * 1024 * 1024) / sizeof(T);

    // ── Round-completion barrier ──────────────────────────────────────────────
    // Clients open ONE connection and reuse it for every round (see
    // FederatedClient::run() in client.cpp — it connects once, then loops
    // run_round() and only disconnects at the very end). handleClient() /
    // handleClientLogits() must therefore loop for the lifetime of the
    // connection rather than reading a single message and returning —
    // otherwise round 2 has nothing left server-side to read the client's
    // next update from, and both client and server stall waiting on each
    // other indefinitely.
    //
    // round_generation increments by one each time the round-runner thread
    // finishes a round (right after broadcast()/broadcastLogits()). A
    // per-client handler thread captures the generation value at the moment
    // it submits its update, then blocks on round_done_cv until the counter
    // moves past that value — i.e. until ITS update has actually been
    // aggregated and broadcast back — before looping around to read that
    // same client's next-round payload. This is what keeps each client's
    // single persistent socket in lock-step with the round it's currently on.
    std::condition_variable               round_done_cv;
    std::atomic<size_t>                   round_generation{0};

    // ── Federated-distillation round state ────────────────────────────────────
    // Same round-membership rules as FedAvg. Payload: each client's soft
    // logits on a shared proxy batch instead of weight deltas, so clients
    // with different architectures can participate (only vocab + proxy shape
    // must match).
    std::vector<std::vector<T>>           logit_updates;
    uint64_t                              proxy_n_examples = 0;
    uint64_t                              proxy_vocab_size = 0;

    // ── Parameter layout cache (networking only) ──────────────────────────────
    // Used by handleClient() to reconstruct per-layer tensors from the flat
    // wire buffer. This is networking metadata, not training state — Trainer
    // doesn't need it because it receives already-reconstructed tensors.
    std::vector<size_t> param_sizes;
    size_t              total_param_elems = 0;

    // ── Networking methods ────────────────────────────────────────────────────

    /// Cache the element count of each parameter tensor.
    /// Called once after model is loaded (constructor body).
    void build_param_layout() {
        param_sizes.clear();
        total_param_elems = 0;
        for (auto& p : model.parameters()) {
            size_t n = p->val.get_size();
            param_sizes.push_back(n);
            total_param_elems += n;
        }
        printf("[server] Model: %zu param tensors, %zu floats (~%.1f MB)\n",
               param_sizes.size(), total_param_elems,
               static_cast<double>(total_param_elems * sizeof(T)) / (1024.0 * 1024.0));
    }

    /// Broadcast updated global model weights to all connected clients,
    /// streamed in chunk_elems-sized pieces straight from model.parameters()
    /// — no intermediate flat copy of the whole model (that's what
    /// trainer.get_flat_weights() would build; chunked broadcast reads
    /// directly from the source tensors instead).
    /// Wire (per client): [uint64 total_elems] then repeated
    ///                     [uint64 chunk_len][T × chunk_len]
    /// Returns elapsed milliseconds.
    double broadcast() {
        auto params = model.parameters();

        std::vector<int> fds;
        {
            std::lock_guard<std::mutex> lk(locker);
            fds = client_sockets;   // copy — broadcast itself doesn't need the lock
        }

        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        for (int fd : fds) {
            // Fresh cursor per fd: it only walks forward, so it can't be
            // reused across multiple independent streams.
            FlatParamCursor cursor(params);
            send_chunked<T>(fd, total_param_elems, chunk_elems,
                [&cursor](uint64_t offset, size_t len, T* out) {
                    cursor.read_into(offset, len, out);
                });
        }

        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    /// Close a client socket and drop its per-fd bandwidth counters (see
    /// io_utils.hpp's net_per_fd map) so a long-running server doesn't keep
    /// accumulating map entries for clients that disconnected long ago.
    void close_client(int sock) {
        net_forget_fd(sock);
        ::close(sock);
    }

    // ── Flat-offset ↔ tensor mapping for chunked I/O ──────────────────────────
    // get_flat_weights()/set_flat_weights() in Trainner.hpp do this same
    // mapping, just all at once (concatenate everything into one big
    // std::vector<T> first). This version walks it incrementally so chunked
    // send/receive never has to materialize that big vector — it only ever
    // needs one chunk_elems-sized buffer, regardless of total model size.
    //
    // send_chunked()/recv_chunked() always call with strictly increasing
    // offsets within a single stream, so this keeps a small cursor (current
    // tensor index + position within it) instead of rescanning params from
    // the start on every chunk. Construct a fresh one per stream — it only
    // walks forward.
    struct FlatParamCursor {
        const std::vector<Tensor_t<T>>& params;
        size_t   tensor_idx    = 0;
        size_t   pos_in_tensor = 0;
        uint64_t flat_pos      = 0;

        explicit FlatParamCursor(const std::vector<Tensor_t<T>>& p) : params(p) {}

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

        /// Copy `len` elements OUT of params starting at flat offset `offset`
        /// into `dst` — used when SENDING (building an outgoing chunk).
        void read_into(uint64_t offset, size_t len, T* dst) {
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
    };

    /// Persistent per-client FedAvg loop. The client keeps ONE socket open
    /// for every round (see comment on round_done_cv above), so this method
    /// must too: read a round's delta — via recv_chunked(), accumulating
    /// straight into round_accum chunk by chunk, never materializing this
    /// client's full delta as its own buffer — then BLOCK until the
    /// round-runner has aggregated + broadcast this round, and only then
    /// loop back to read this same client's next-round delta. Returns (and
    /// closes the socket) only once the client disconnects or the server
    /// shuts down.
    ///
    /// Wire (per round): [uint64 total_elems] then repeated
    ///                    [uint64 chunk_len][T × chunk_len]  (see io_utils.hpp)
    void handleClient(int sock) {
        while (true) {
            bool ok;
            try {
                // Each chunk gets added directly into the shared running
                // sum. Multiple clients' handleClient threads do this
                // concurrently on DIFFERENT chunks/sockets, all touching the
                // SAME round_accum, so each chunk's accumulation is guarded
                // by `locker` — float addition isn't atomic, and chunks are
                // big enough (thousands of elements) that the lock/unlock
                // overhead is negligible next to the actual network read.
                ok = recv_chunked<T>(sock, total_param_elems,
                    [this](uint64_t offset, const T* data, size_t len) {
                        std::lock_guard<std::mutex> lk(locker);
                        for (size_t i = 0; i < len; ++i)
                            round_accum[offset + i] += data[i];
                    });
            } catch (const std::exception& e) {
                // Most likely a stale/mismatched client (wrong model
                // architecture or chunk total) — log and drop it rather than
                // let a malformed stream desync everything after it.
                std::cerr << "[server] fd=" << sock << ": " << e.what() << "\n";
                close_client(sock);
                return;
            }
            if (!ok) { close_client(sock); return; }

            size_t my_round;
            {
                std::lock_guard<std::mutex> lk(locker);
                clients_done++;
                client_sockets.push_back(sock);
                my_round = round_generation;
                printf("[server] Client fd=%d submitted update (%zu/%zu)\n",
                       sock, clients_done, min_clients);
            }
            round_cv.notify_one();

            // Block until THIS round's aggregate()+broadcast() has actually
            // happened (round_generation increments past my_round), so we
            // don't race back into recv_chunked() for round N+1 before the
            // server has even finished round N. Also wakes on shutdown
            // (server_fd < 0) so this thread doesn't block forever.
            std::unique_lock<std::mutex> lk(locker);
            round_done_cv.wait(lk, [this, my_round] {
                return round_generation > my_round || server_fd < 0;
            });
            if (server_fd < 0) { lk.unlock(); close_client(sock); return; }
            lk.unlock();
            // Loop back and read this same client's next-round delta.
        }
    }

    /// Persistent per-client FedDistill loop — same reasoning as
    /// handleClient() above (one socket, reused for every round). Reads one
    /// round's proxy-batch logits, submits them, waits for this round's
    /// aggregate_logits()+broadcastLogits() to complete, then loops back to
    /// read this client's next-round logits.
    /// Wire (per round): [uint64 n_examples][uint64 vocab_size][uint64 total_bytes][T × N]
    void handleClientLogits(int sock) {
        while (true) {
            uint64_t n_examples = 0, vocab_size = 0, total_bytes = 0;
            if (!read_exact(sock, &n_examples, sizeof(uint64_t)) ||
                !read_exact(sock, &vocab_size,  sizeof(uint64_t)) ||
                !read_exact(sock, &total_bytes, sizeof(uint64_t))) {
                close_client(sock);
                return;
            }

            std::vector<T> flat(total_bytes / sizeof(T));
            if (!read_exact(sock, flat.data(), total_bytes)) {
                close_client(sock);
                return;
            }

            size_t my_round;
            {
                std::lock_guard<std::mutex> lk(locker);
                if (logit_updates.empty()) {
                    // First contributor this round defines the proxy-batch shape.
                    // Mismatches are caught in trainer.aggregate_logits().
                    proxy_n_examples = n_examples;
                    proxy_vocab_size = vocab_size;
                }
                logit_updates.push_back(std::move(flat));
                client_sockets.push_back(sock);
                my_round = round_generation;
                printf("[server] Client fd=%d submitted logits (%zu/%zu)\n",
                       sock, logit_updates.size(), min_clients);
            }
            round_cv.notify_one();

            std::unique_lock<std::mutex> lk(locker);
            round_done_cv.wait(lk, [this, my_round] {
                return round_generation > my_round || server_fd < 0;
            });
            if (server_fd < 0) { lk.unlock(); close_client(sock); return; }
            lk.unlock();
            // Loop back and read this same client's next-round logits.
        }
    }

    /// Broadcast the averaged consensus logits to all connected clients
    /// (FedDistill phase 2). The consensus vector comes from
    /// trainer.aggregate_logits() called in the round runner.
    /// Wire: [uint64 n_examples][uint64 vocab_size][uint64 total_bytes][T × N]
    /// Returns elapsed milliseconds.
    double broadcastLogits(const std::vector<T>& consensus) {
        uint64_t total_bytes = consensus.size() * sizeof(T);

        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        std::lock_guard<std::mutex> lk(locker);
        for (int fd : client_sockets) {
            write_exact(fd, &proxy_n_examples, sizeof(uint64_t));
            write_exact(fd, &proxy_vocab_size, sizeof(uint64_t));
            write_exact(fd, &total_bytes,      sizeof(uint64_t));
            write_exact(fd, consensus.data(),  total_bytes);
        }

        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    /// Close and clear all open client sockets; reset pending FedAvg state.
    void closeClientSocks() {
        std::lock_guard<std::mutex> lk(locker);
        for (int fd : client_sockets) close_client(fd);
        client_sockets.clear();
        clients_done = 0;
        std::fill(round_accum.begin(), round_accum.end(), T(0));
    }

    /// Clear logit update state between FedDistill rounds.
    void clearLogitUpdates() {
        std::lock_guard<std::mutex> lk(locker);
        logit_updates.clear();
        proxy_n_examples = 0;
        proxy_vocab_size = 0;
    }

public:
    // ── Global model + Trainer ────────────────────────────────────────────────
    //
    // model    — the authoritative global model updated each round.
    // trainer  — owns ALL aggregation math (aggregate, aggregate_logits,
    //            get_flat_weights, set_flat_weights). The server's Trainer is
    //            constructed with lr = T(0): it never calls backward() or step(),
    //            so the optimizer is present but inert. Only the aggregation and
    //            serialisation methods are used server-side.
    //
    // Declaration order matters: trainer holds a reference to model, so model
    // must be declared (and therefore initialised) first.
    GPT<T, LinearT>                  model;
    Trainer<GPT<T, LinearT>, T>      trainer;   // initialised after model in init-list

    // ── Constructor ──────────────────────────────────────────────────────────

    /// Load model and tokenizer from the same GGUF file.
    /// min_clients_per_round controls how many updates are awaited per round.
    /// chunk_mb sizes THIS process's outgoing broadcast chunks (see
    /// chunk_elems above) — 0 falls back to a 1-element minimum rather than
    /// "unchunked," since the wire protocol is chunked unconditionally now.
    template <typename... ModelArgs>
    Server(const std::string& model_path,
           const GPT2HyperParams& hyper,
           size_t min_clients_per_round,
           double chunk_mb,
           ModelArgs&&... model_args)
        : hp(hyper),
          min_clients(min_clients_per_round),
          model(hp.vocab_size, hp.d_model, hp.block_size, hp.n_head, hp.n_layer,
                std::forward<ModelArgs>(model_args)...),
          // trainer takes a reference to model (already constructed above since
          // model is declared before trainer in the class body). lr = T(0):
          // server never runs backward, so the optimizer is a no-op.
          trainer(model, T(0), FEDAVG)
    {
        // Load the dense GGUF checkpoint and copy it into `model`. For
        // LinearT=Linear this is a dense-to-dense copy; for LoRALinear the
        // backbone is copied verbatim and the LoRA {A,B} adapters (already
        // randomly initialised above) are left untouched.
        GPT<T, Linear> pretrained = loader.load_model(model_path, hp);
        model.load_backbone_from(pretrained);
        model.load_head_from(pretrained);
        // Sized here (not as a default member initializer) because it needs
        // min_clients, which only exists once the init-list above has run.
        // +4 is just headroom for a stray extra/retrying connection — not a
        // hard cap on participants, since handleClient()/handleClientLogits()
        // now hold their thread for the connection's entire lifetime (see
        // round_done_cv), unlike before when each task finished in one read.
        pool = std::make_unique<ThreadPool>(min_clients + 4);

        size_t requested = static_cast<size_t>(chunk_mb * 1024.0 * 1024.0 / sizeof(T));
        chunk_elems = std::max<size_t>(1, requested);

        // Tokenizer is assigned in the body (not init-list) because
        // load_tokenizer() reads metadata that load_model() populates — the
        // body runs after all member initialisers, guaranteeing ordering.
        tokenizer = loader.load_tokenizer();
        if (tokenizer.encoder.empty())
            throw std::runtime_error(
                "Server: failed to load tokenizer from GGUF: " + model_path);
        build_param_layout();

        // Sized now that build_param_layout() has set total_param_elems —
        // this is the ONE model-sized allocation FedAvg aggregation needs,
        // replacing what used to be n_clients separate full-model copies.
        round_accum.assign(total_param_elems, T(0));
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void stop() {
        closeClientSocks();
        if (server_fd >= 0) { ::close(server_fd); server_fd = -1; }
        // Unblock any handleClient/handleClientLogits threads currently
        // parked in the round_done_cv wait — they check `server_fd < 0` and
        // exit cleanly instead of blocking forever on a server that's gone.
        round_done_cv.notify_all();
    }
};
