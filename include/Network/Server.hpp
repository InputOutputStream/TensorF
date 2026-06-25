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
#include <string>
#include <cstdint>
#include <stdexcept>

#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"

#include "../Modules/Transformer/GPT/GPT.hpp"
#include "../ModelLoader/GPTLoader.hpp"
#include "../Tokenizer/GPT2Tokenizer.hpp"

// ── Server<T> ─────────────────────────────────────────────────────────────────
//
// Base class for federated learning servers. Owns:
//   • GPT<T> model          — global model (weights averaged each FedAvg round;
//                              also supplies the teacher signal in FedDistill if
//                              needed, though in pure FedDistill the consensus is
//                              the averaged client logits, not this model's output)
//   • Trainer<GPT<T>,T> trainer — owns ALL aggregation math:
//                              aggregate()        → FedAvg (replaces federatedAverage+applyUpdate)
//                              aggregate_logits() → FedDistill consensus
//                              get_flat_weights() → serialise model for broadcast
//                              set_flat_weights() → apply received weights (not called
//                                                   here, but available for round-trip tests)
//   • TCP listener socket + ThreadPool for concurrent client connections
//   • FedAvg state: updates[], client_sockets[], param layout
//
// Responsibility split:
//   Server   — pure networking + socket/round lifecycle.
//   Trainer  — all training math (aggregation, serialisation, distillation).
//   FederatedServer (server.cpp) — profiling, per-round stats, accept loop.
//
// Access policy:
//   public    — interface used from main() and FederatedServer
//   protected — called by FederatedServer subclass

template<typename T>
class Server {

protected:
    // ── Model loader + tokenizer ─────────────────────────────────────────────
    GPT2HyperParams   hp;
    GPTGGUFLoader<T> loader;
    GPT2Tokenizer     tokenizer;

    // ── Networking ───────────────────────────────────────────────────────────
    int    server_fd = -1;
    ThreadPool pool{4};

    // ── FedAvg round state ────────────────────────────────────────────────────
    std::mutex                            locker;
    std::condition_variable               round_cv;
    std::vector<std::vector<Tensor_t<T>>> updates;
    std::vector<int>                      client_sockets;
    size_t                                min_clients = 2;

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

    /// Broadcast updated global model weights to all connected clients.
    /// Delegates serialisation to trainer.get_flat_weights().
    /// Wire: [uint64 total_bytes][T × N]
    /// Returns elapsed milliseconds.
    double broadcast() {
        // Trainer serialises — server just writes to sockets.
        auto flat = trainer.get_flat_weights();
        uint64_t total_bytes = flat.size() * sizeof(T);

        using Clock = std::chrono::steady_clock;
        auto t0 = Clock::now();

        std::lock_guard<std::mutex> lk(locker);
        for (int fd : client_sockets) {
            write_exact(fd, &total_bytes, sizeof(uint64_t));
            write_exact(fd, flat.data(), total_bytes);
        }

        return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    }

    /// Receive one client's FedAvg gradient delta, reconstruct per-layer tensors,
    /// push onto updates[], and notify the round condition variable.
    /// Wire (receive): [uint64 total_bytes][T × N]
    ///
    /// Networking concern only — reconstruction uses param_sizes[] built at
    /// startup. The actual aggregation math lives in trainer.aggregate().
    void handleClient(int sock) {
        uint64_t total_bytes = 0;
        if (!read_exact(sock, &total_bytes, sizeof(uint64_t))) {
            ::close(sock); return;
        }

        std::vector<T> flat(total_bytes / sizeof(T));
        if (!read_exact(sock, flat.data(), total_bytes)) {
            ::close(sock); return;
        }

        auto params = model.parameters();
        std::vector<Tensor_t<T>> deltas;
        deltas.reserve(params.size());

        size_t offset = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            size_t n = param_sizes[i];
            if (offset + n > flat.size()) {
                std::cerr << "[server] Truncated data from fd=" << sock << "\n";
                ::close(sock); return;
            }
            std::vector<T> slice(flat.begin() + offset, flat.begin() + offset + n);
            deltas.push_back(make_tensor<T>(Matrix<T>(slice, params[i]->shape)));
            offset += n;
        }

        {
            std::lock_guard<std::mutex> lk(locker);
            updates.push_back(std::move(deltas));
            client_sockets.push_back(sock);
            printf("[server] Client fd=%d submitted update (%zu/%zu)\n",
                   sock, updates.size(), min_clients);
        }
        round_cv.notify_one();
    }

    /// Receive one client's proxy-batch logits (FedDistill phase 1), push onto
    /// logit_updates[], and notify the round condition variable.
    /// Wire: [uint64 n_examples][uint64 vocab_size][uint64 total_bytes][T × N]
    void handleClientLogits(int sock) {
        uint64_t n_examples = 0, vocab_size = 0, total_bytes = 0;
        if (!read_exact(sock, &n_examples, sizeof(uint64_t)) ||
            !read_exact(sock, &vocab_size,  sizeof(uint64_t)) ||
            !read_exact(sock, &total_bytes, sizeof(uint64_t))) {
            ::close(sock); return;
        }

        std::vector<T> flat(total_bytes / sizeof(T));
        if (!read_exact(sock, flat.data(), total_bytes)) {
            ::close(sock); return;
        }

        {
            std::lock_guard<std::mutex> lk(locker);
            if (logit_updates.empty()) {
                // First contributor defines the proxy-batch shape for the round.
                // Mismatches are caught in trainer.aggregate_logits().
                proxy_n_examples = n_examples;
                proxy_vocab_size = vocab_size;
            }
            logit_updates.push_back(std::move(flat));
            client_sockets.push_back(sock);
            printf("[server] Client fd=%d submitted logits (%zu/%zu)\n",
                   sock, logit_updates.size(), min_clients);
        }
        round_cv.notify_one();
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

    /// Close and clear all open client sockets; clear pending FedAvg updates.
    void closeClientSocks() {
        std::lock_guard<std::mutex> lk(locker);
        for (int fd : client_sockets) ::close(fd);
        client_sockets.clear();
        updates.clear();
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
    GPT<T>                  model;
    Trainer<GPT<T>, T>      trainer;   // initialised after model in init-list

    // ── Constructor ──────────────────────────────────────────────────────────

    /// Load model and tokenizer from the same GGUF file.
    /// min_clients_per_round controls how many updates are awaited per round.
    Server(const std::string& model_path,
           const GPT2HyperParams& hyper,
           size_t min_clients_per_round = 2)
        : hp(hyper),
          min_clients(min_clients_per_round),  
          model(loader.load_model(model_path, hp)),
          // trainer takes a reference to model (already constructed above since
          // model is declared before trainer in the class body). lr = T(0):
          // server never runs backward, so the optimizer is a no-op.
          trainer(model, T(0), FEDAVG)
    {
        // Tokenizer is assigned in the body (not init-list) because
        // load_tokenizer() reads metadata that load_model() populates — the
        // body runs after all member initialisers, guaranteeing ordering.
        tokenizer = loader.load_tokenizer();
        if (tokenizer.encoder.empty())
            throw std::runtime_error(
                "Server: failed to load tokenizer from GGUF: " + model_path);
        build_param_layout();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void stop() {
        closeClientSocks();
        if (server_fd >= 0) { ::close(server_fd); server_fd = -1; }
    }
};
