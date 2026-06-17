#ifndef CLIENT_HPP__
#define CLIENT_HPP__

// Provides read_exact / write_exact — do NOT redefine them here.
#include "../Network/io_utils.hpp"

#include "../DataStructures/Matrix.hpp"
#include "../DataStructures/Tensor.hpp"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdio>

// ── Client<T> ─────────────────────────────────────────────────────────────────
//
// Lightweight networking base used by FederatedClient (client.cpp).
// Owns the socket and implements the two wire protocols:
//
//   Flat-tensor protocol  (params / deltas)
//     send   → [uint64 total_bytes][T × N]
//     receive← [uint64 total_bytes][T × N]
//
//   Batch protocol  (server → client data streaming)
//     receive← [uint64 batch_size][uint64 block_size]
//               [T × batch*block  inputs]
//               [T × batch*block  targets]
//
// FederatedClient builds the flat vectors from its own model.parameters()
// and passes them directly to send() / receive(), keeping model ownership
// in the subclass.

template<typename T>
class Client {
    int         sock_ = -1;
    sockaddr_in serv_{};

public:
    // ── Connection ────────────────────────────────────────────────────────────

    /// Open a TCP connection to ip:port.
    /// Returns true on success; on failure the socket is cleaned up.
    bool connect_to_server(const std::string& ip, uint16_t port) {
        sock_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (sock_ < 0) { perror("socket"); return false; }

        serv_.sin_family = AF_INET;
        serv_.sin_port   = htons(port);
        ::inet_pton(AF_INET, ip.c_str(), &serv_.sin_addr);

        if (::connect(sock_, reinterpret_cast<sockaddr*>(&serv_), sizeof(serv_)) < 0) {
            perror("connect"); ::close(sock_); sock_ = -1; return false;
        }
        return true;
    }

    void disconnect() {
        if (sock_ >= 0) { ::close(sock_); sock_ = -1; }
    }

    /// Raw file descriptor — used by FederatedClient to guard send/receive.
    int fd() const { return sock_; }

    // ── Flat-tensor wire protocol ────────────────────────────────────────────

    /// Send:  [uint64 total_bytes][T × flat.size()]
    /// Called by FederatedClient::send_deltas() after building flat from model params.
    bool send(const std::vector<T>& flat) {
        uint64_t total_bytes = flat.size() * sizeof(T);
        return write_exact(sock_, &total_bytes, sizeof(uint64_t)) &&
               write_exact(sock_, flat.data(), total_bytes);
    }

    /// Receive: [uint64 total_bytes] → resizes flat and fills it.
    /// Called by FederatedClient::receive_weights(); caller unpacks flat → model params.
    bool receive(std::vector<T>& flat) {
        uint64_t total_bytes = 0;
        if (!read_exact(sock_, &total_bytes, sizeof(uint64_t))) return false;
        flat.resize(total_bytes / sizeof(T));
        return read_exact(sock_, flat.data(), total_bytes);
    }

    // ── Batch protocol (server → client) ────────────────────────────────────

    /// Receive a training batch streamed from the server.
    /// Returns {inputs, targets} as Tensor_t<T> of shape [batch_size, block_size].
    std::pair<Tensor_t<T>, Tensor_t<T>> receiveNextBatch() {
        uint64_t batch_sz = 0, block_sz = 0;
        if (!read_exact(sock_, &batch_sz, sizeof(uint64_t)) ||
            !read_exact(sock_, &block_sz, sizeof(uint64_t)))
            throw std::runtime_error("Client::receiveNextBatch: connection dropped reading header");

        size_t total = static_cast<size_t>(batch_sz * block_sz);
        std::vector<T> ibuf(total), tbuf(total);

        if (!read_exact(sock_, ibuf.data(), total * sizeof(T)) ||
            !read_exact(sock_, tbuf.data(), total * sizeof(T)))
            throw std::runtime_error("Client::receiveNextBatch: connection dropped reading data");

        return { make_tensor<T>(Matrix<T>(ibuf, {batch_sz, block_sz})),
                 make_tensor<T>(Matrix<T>(tbuf, {batch_sz, block_sz})) };
    }
};

#endif // CLIENT_HPP__