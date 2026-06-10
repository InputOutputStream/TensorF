#pragma once

#include "ThreadPool.hpp"

#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <mutex>
#include <condition_variable>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <random>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

#include "../Types/types.hpp"
#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"
#include "../Modules/Transformer/Llama/Llama.hpp
#include "../ModelLoader/GGUF.hpp"
#include "../DataLoader/DataLoading.hpp"
#include "../Tokenizer/GPT2Tokenizer.hpp"

// ── Hyperparams (GPT-2 small) ────────────────────────────────────────────────
struct GPT2HyperParams {
    size_t vocab_size;
    size_t d_model;
    size_t block_size;
    size_t n_layer;
    size_t n_head;
};

static GPT2HyperParams default_hp {
    .vocab_size = 50257,
    .d_model    = 768,
    .block_size = 1024,
    .n_layer    = 12,
    .n_head     = 12
};

inline bool read_exact(int fd, void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = ::read(fd, static_cast<char*>(buf) + done, len - done);
        if (r <= 0) return false;
        done += r;
    }
    return true;
}

inline bool write_exact(int fd, const void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t w = ::write(fd, static_cast<const char*>(buf) + done, len - done);
        if (w <= 0) return false;
        done += w;
    }
    return true;
}

// ── Server ───────────────────────────────────────────────────────────────────
template<typename T>
class Server {

    int server_fd = -1;
    sockaddr_in addr{};

    ThreadPool pool{4};
    std::mutex locker;

    std::condition_variable round_cv;
    size_t min_clients_per_round = 2;

    // updates[client_index][layer_index]
    std::vector<std::vector<Tensor_t<T>>> updates;
    std::vector<int> client_sockets;

    std::vector<size_t> param_sizes;
    size_t total_param_elems = 0;

    void build_param_layout() {
        param_sizes.clear();
        total_param_elems = 0;
        for (auto& p : model.parameters()) {
            size_t n = p->val.get_size();
            param_sizes.push_back(n);
            total_param_elems += n;
        }
    }

public:
    GPT2Tokenizer tokenizer;
    GGUFLoader<float> loader;
    GPT<T> model;

    Server(const std::string& modelPath  = "SLM/gpt2-small-f32.gguf",
           const std::string& vocabPath  = "SLM/gpt2-tokenizer/vocab.json",
           const std::string& mergesPath = "SLM/gpt2-tokenizer/merges.txt",
           const GPT2HyperParams& hp     = default_hp)
        : model(loader.load_model(modelPath, hp))
    {
        tokenizer.load(vocabPath, mergesPath);
        build_param_layout();
    }

    // ── Networking ───────────────────────────────────────────────────────────

    void start(uint16_t port = 8080) {
        server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
        if (server_fd < 0)
            throw std::runtime_error("Server::start: socket() failed");

        int opt = 1;
        ::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        addr.sin_family      = AF_INET;
        addr.sin_port        = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (::bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0)
            throw std::runtime_error("Server::start: bind() failed");

        ::listen(server_fd, 16);
        std::cout << "Server listening on port " << port << "...\n";

        while (true) {
            int clientSock = ::accept(server_fd, nullptr, nullptr);
            if (clientSock < 0) continue;
            pool.enqueue([this, clientSock]() { this->handleClient(clientSock); });
        }
    }

    // ── Federated logic ──────────────────────────────────────────────────────

    std::vector<Tensor_t<T>> federatedAverage() {
        // Called only from federated_round() which already holds locker
        if (updates.empty())
            throw std::runtime_error("federatedAverage: no updates received");

        size_t n_clients = updates.size();
        size_t n_layers  = updates[0].size();

        std::vector<Tensor_t<T>> avg(n_layers);
        for (size_t layer = 0; layer < n_layers; ++layer) {
            avg[layer] = make_tensor<T>(Matrix<T>::zeros(updates[0][layer]->shape));
            for (size_t client = 0; client < n_clients; ++client)
                avg[layer] = avg[layer] + updates[client][layer];

            avg[layer] = avg[layer] / (T)n_clients;
        }
        return avg;
    }

    // Apply averaged layer weights directly into the global model.
    void applyUpdate(const std::vector<Tensor_t<T>>& avg) {
        auto params = model.parameters();
        if (avg.size() != params.size())
            throw std::runtime_error("applyUpdate: param/avg size mismatch");
        for (size_t i = 0; i < params.size(); ++i)
            params[i]->val = avg[i]->val;
    }

    // Push current global weights to every connected client.
    // Wire: [uint64 total_bytes][T * N flat]
    
    void broadcast() {
        std::vector<T> flat;
        flat.reserve(total_param_elems);
        for (auto& p : model.parameters())
            flat.insert(flat.end(), p->val.data.begin(), p->val.data.end());

        uint64_t total_bytes = flat.size() * sizeof(T);

        std::lock_guard<std::mutex> lock(locker);
        for (int fd : client_sockets) {
            write_exact(fd, &total_bytes, sizeof(uint64_t));
            write_exact(fd, flat.data(), total_bytes);
        }
    }

    std::vector<float> decompress_deltas(const std::vector<fp8_e4m3>& compressed) {
        std::vector<float> out;
        for (auto& v : compressed)
            out.push_back(float(v));  // decode FP8 → float
        return out;
    }

    // Full federated round: wait for enough clients → average → apply → broadcast → reset.
    void federated_round() {
        {
            std::unique_lock<std::mutex> lock(locker);
            // Block until at least min_clients_per_round have submitted
            round_cv.wait(lock, [this] {
                return updates.size() >= min_clients_per_round;
            });

            auto avg = federatedAverage();
            applyUpdate(avg);
            updates.clear();
            // locker released here — broadcast locks it again separately
        }
        // Broadcast happens outside the aggregation lock so accept loop keeps running
        broadcast();
    }

    // ── Per-client handler (thread pool) ─────────────────────────────────────
    void handleClient(int clientSock) {
        uint64_t total_bytes = 0;
        if (!read_exact(clientSock, &total_bytes, sizeof(uint64_t))) {
            ::close(clientSock); return;
        }

        std::vector<T> flat(total_bytes / sizeof(T));
        if (!read_exact(clientSock, flat.data(), total_bytes)) {
            ::close(clientSock); return;
        }

        // Reconstruct per-layer tensors from the known model parameter layout
        auto params = model.parameters();
        std::vector<Tensor_t<T>> deltas;
        deltas.reserve(params.size());

        size_t offset = 0;
        for (size_t i = 0; i < params.size(); ++i) {
            size_t n = param_sizes[i];
            if (offset + n > flat.size()) {
                std::cerr << "handleClient: truncated data from fd=" << clientSock << "\n";
                ::close(clientSock); return;
            }
            std::vector<T> slice(flat.begin() + offset, flat.begin() + offset + n);
            deltas.push_back(make_tensor<T>(Matrix<T>(slice, params[i]->shape)));
            offset += n;
        }

        {
            std::lock_guard<std::mutex> lock(locker);
            updates.push_back(std::move(deltas));
            client_sockets.push_back(clientSock);
        }

        round_cv.notify_one();
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────

    void closeClientSocks() {
        std::lock_guard<std::mutex> lock(locker);
        for (int fd : client_sockets) ::close(fd);
        client_sockets.clear();
        updates.clear();
    }

    void stop() {
        closeClientSocks();
        if (server_fd >= 0) { ::close(server_fd); server_fd = -1; }
    }
};


#endif