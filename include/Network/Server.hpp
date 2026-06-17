#pragma once

#include "ThreadPool.hpp"
#include "../Network/io_utils.hpp"

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
#include "../Modules/Transformer/Llama/Llama.hpp"
#include "../ModelLoader/LlamaLoader.hpp"
#include "../Tokenizer/LlamaTokenizer.hpp"

// ── Server<T> ─────────────────────────────────────────────────────────────────
//
// Base class for federated learning servers.  Owns:
//   • Llama<T> model + LlamaTokenizer + LlamaGGUFLoader
//   • TCP listener socket + ThreadPool for concurrent client connections
//   • FedAvg state: updates[], client_sockets[], param layout
//
// All heavy lifting (federatedAverage, applyUpdate, broadcast, handleClient,
// build_param_layout) lives here so FederatedServer (server.cpp) only needs
// to add profiling, per-round stats and the main accept-loop/round-runner.
//
// Access policy:
//   public    — interface used from main()
//   protected — called by FederatedServer subclass

template<typename T>
class Server {

protected:
    // ── Model ────────────────────────────────────────────────────────────────
    LlamaHyperParams   hp;
    LlamaGGUFLoader<T> loader;
    LlamaTokenizer     tokenizer;

    // ── Networking ───────────────────────────────────────────────────────────
    int    server_fd = -1;
    ThreadPool pool{4};

    // ── FedAvg state ─────────────────────────────────────────────────────────
    std::mutex                            locker;
    std::condition_variable               round_cv;
    std::vector<std::vector<Tensor_t<T>>> updates;
    std::vector<int>                      client_sockets;
    size_t                                min_clients = 2;

    // ── Parameter layout cache ────────────────────────────────────────────────
    std::vector<size_t> param_sizes;
    size_t              total_param_elems = 0;

    // ── Core methods called by FederatedServer ────────────────────────────────

    /// Cache the number of elements in each parameter tensor.
    /// Must be called once after the model is loaded (constructor).
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

    /// Compute element-wise average across all client updates.
    std::vector<Tensor_t<T>> federatedAverage() {
        if (updates.empty())
            throw std::runtime_error("federatedAverage: no updates received");

        const size_t n_clients = updates.size();
        const size_t n_layers  = updates[0].size();

        std::vector<Tensor_t<T>> avg(n_layers);
        for (size_t l = 0; l < n_layers; ++l) {
            avg[l] = make_tensor<T>(Matrix<T>::zeros(updates[0][l]->shape));
            for (size_t c = 0; c < n_clients; ++c)
                avg[l] = avg[l] + updates[c][l];
            avg[l] = avg[l] / static_cast<T>(n_clients);
        }
        return avg;
    }

    /// Write averaged tensors back into model parameters.
    void applyUpdate(const std::vector<Tensor_t<T>>& avg) {
        auto params = model.parameters();
        if (avg.size() != params.size())
            throw std::runtime_error("applyUpdate: param/avg size mismatch");
        for (size_t i = 0; i < params.size(); ++i)
            params[i]->val = avg[i]->val;
    }

    /// Broadcast updated model weights to all connected clients.
    /// Wire: [uint64 total_bytes][T × N]
    /// Returns elapsed time in milliseconds.
    double broadcast() {
        std::vector<T> flat;
        flat.reserve(total_param_elems);
        for (auto& p : model.parameters())
            flat.insert(flat.end(), p->val.data.begin(), p->val.data.end());

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

    /// Receive one client's gradient update, reconstruct per-layer tensors,
    /// push onto updates[], and notify the round condition variable.
    /// Wire (receive): [uint64 total_bytes][T × N]
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

    /// Close and clear all open client sockets; clear pending updates.
    void closeClientSocks() {
        std::lock_guard<std::mutex> lk(locker);
        for (int fd : client_sockets) ::close(fd);
        client_sockets.clear();
        updates.clear();
    }

public:
    // Model is public so FederatedServer (and tests) can call model.generate() etc.
    Llama<T> model;

    // ── Constructor ──────────────────────────────────────────────────────────

    /// Load model from GGUF and tokenizer from vocab/merges files.
    /// min_clients_per_round controls how many updates are awaited before aggregation.
    Server(const std::string& model_path,
           const std::string& vocab_path,
           const std::string& merges_path,
           const LlamaHyperParams& hyper,
           size_t min_clients_per_round = 2)
        : hp(hyper),
          model(loader.load_model(model_path, hp)),
          min_clients(min_clients_per_round)
    {
        tokenizer.load(vocab_path, merges_path);
        build_param_layout();
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    void stop() {
        closeClientSocks();
        if (server_fd >= 0) { ::close(server_fd); server_fd = -1; }
    }
};