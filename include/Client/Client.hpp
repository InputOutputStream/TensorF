#ifndef CLIENT_HPP__
#define CLIENT_HPP__

#include "../DataLoader/LazyLoader.hpp"
#include "../Modules/Transformer/Llama/Llama.hpp"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <optional>
#include <cstdint>
#include <stdexcept>

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

template <typename T>
class Client {

private:
    int sock = -1;
    size_t port;
    size_t iters;
    size_t eval_iters;
    sockaddr_in serv{};

public:
    LazyDataLoader<T> loader;
    std::vector<Tensor_t<T>> deltas;
    bool is_training = false;
    std::optional<Llama<T>> model;

    Client(const std::string& path,
           size_t port,
           size_t iters,
           size_t eval_iters,
           size_t vocab_size,
           size_t d_model,
           size_t block_size,
           size_t n_heads,
           size_t n_layer,
           size_t batch_size = 8)
        : loader(path, block_size, batch_size),
          port(port),
          iters(iters),
          eval_iters(eval_iters),
          model(std::in_place, vocab_size, d_model, block_size, n_heads, n_layer)
    {
        std::system("./setup_client.sh");
        // Populate deltas from model parameter shapes so receive() can unpack into them
        init_deltas();
    }

    // ── Initialization ───────────────────────────────────────────────────────

    void init_deltas() {
        deltas.clear();
        for (auto& p : model->parameters())
            deltas.push_back(make_tensor<T>(Matrix<T>::zeros(p->shape)));
    }

    // ── Training ─────────────────────────────────────────────────────────────

    void train(bool OnLocalData = true) {
        auto local_batch_fn = [this](const std::string& split) {
            return loader.getNextBatch(split);
        };

        if (OnLocalData) {
            while (is_training)
                model->train(local_batch_fn, iters, eval_iters);
            return;
        }

        // Remote-batch mode: server streams batches over the socket.
        while (is_training) {
            auto [inputs, targets] = receiveNextBatch();
            // Wrap the single received batch as a batch function for GPT::train
            auto remote_batch_fn = [&](const std::string&) {
                return std::make_pair(inputs, targets);
            };
            model->train(remote_batch_fn, iters, eval_iters);
        }
    }

    // ── Batch protocol (server → client) ────────────────────────────────────
    //   [uint64 batch_size][uint64 block_size]
    //   [T * batch*block  inputs]
    //   [T * batch*block  targets]
    std::pair<Tensor_t<T>, Tensor_t<T>> receiveNextBatch() {
        uint64_t batch_sz = 0, block_sz = 0;
        if (!read_exact(sock, &batch_sz, sizeof(uint64_t)) ||
            !read_exact(sock, &block_sz, sizeof(uint64_t)))
            throw std::runtime_error("Client::receiveNextBatch: connection dropped reading header");

        size_t total = batch_sz * block_sz;
        std::vector<T> ibuf(total), tbuf(total);

        if (!read_exact(sock, ibuf.data(), total * sizeof(T)) ||
            !read_exact(sock, tbuf.data(), total * sizeof(T)))
            throw std::runtime_error("Client::receiveNextBatch: connection dropped reading data");

        return { make_tensor<T>(Matrix<T>(ibuf, {batch_sz, block_sz})),
                 make_tensor<T>(Matrix<T>(tbuf, {batch_sz, block_sz})) };
    }

    // ── Networking ───────────────────────────────────────────────────────────

    void connect_to_server(const std::string& ip = "127.0.0.1") {
        sock = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock < 0)
            throw std::runtime_error("Client::connect_to_server: socket() failed");

        serv.sin_family = AF_INET;
        serv.sin_port   = htons(port);
        ::inet_pton(AF_INET, ip.c_str(), &serv.sin_addr);

        if (::connect(sock, reinterpret_cast<sockaddr*>(&serv), sizeof(serv)) < 0)
            throw std::runtime_error("Client::connect_to_server: connect() failed");
    }

    // Receive updated global weights from server after FedAvg.
    // Wire: [uint64 total_bytes][T * N]
    bool receive() {
        uint64_t total_bytes = 0;
        if (!read_exact(sock, &total_bytes, sizeof(uint64_t))) return false;

        std::vector<T> flat(total_bytes / sizeof(T));
        if (!read_exact(sock, flat.data(), total_bytes)) return false;

        size_t offset = 0;
        for (auto& g : deltas) {
            size_t n = g->val.get_size();
            if (offset + n > flat.size())
                throw std::runtime_error("Client::receive: size mismatch unpacking tensors");
            std::copy(flat.begin() + offset, flat.begin() + offset + n,
                      g->val.data.begin());
            offset += n;
        }
        return true;
    }

    // Send locally-computed gradient deltas to server.
    // Wire: [uint64 total_bytes][T * N]
    bool send() {
        std::vector<T> flat;
        for (auto& g : deltas)
            flat.insert(flat.end(), g->val.data.begin(), g->val.data.end());

        uint64_t total_bytes = flat.size() * sizeof(T);
        if (!write_exact(sock, &total_bytes, sizeof(uint64_t))) return false;
        return write_exact(sock, flat.data(), total_bytes);
    }

    // ── LoRA delta helpers ───────────────────────────────────────────────────

    std::vector<float> extract_lora_deltas(GPT<float>& model) {
        std::vector<float> deltas;
        for (auto& layer : model.blocks) {
            auto& A = layer.attn.q_proj.A->val.data;  
            auto& B = layer.attn.q_proj.B->val.data;
            deltas.insert(deltas.end(), A.begin(), A.end());
            deltas.insert(deltas.end(), B.begin(), B.end());
        }
        return deltas; 
    }

    std::vector<fp8_e4m3> compress_deltas(const std::vector<float>& deltas) {
        std::vector<fp8_e4m3> out;
        out.reserve(deltas.size());
        for (float v : deltas)
            out.emplace_back(v);  // encode float → FP8
        return out;
    }

    void disconnect() {
        if (sock >= 0) {
            ::close(sock);
            sock = -1;
        }
    }
};

#endif