#ifndef CLIENT_HPP__
#define CLIENT_HPP__

// Provides read_exact / write_exact — do NOT redefine them here.
#include "io_utils.hpp"

#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor.hpp"

#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <cstdio>
#include <functional>

// ── Client<T> ─────────────────────────────────────────────────────────────────
//
// Lightweight networking base used by FederatedClient (client.cpp).
// Owns the socket and implements three wire protocols:
//
//   Flat-tensor protocol  (params / deltas) — kept for reference/round-trip
//   tests; FederatedClient's actual weight exchange uses the chunked
//   protocol below instead.
//     send   → [uint64 total_bytes][T × N]
//     receive← [uint64 total_bytes][T × N]
//
//   Chunked flat-tensor protocol (params / deltas) — what send_deltas()/
//   receive_weights() in client.cpp actually use. Same logical payload as
//   above, streamed in caller-chosen pieces instead of one big message, so
//   peak extra memory is ~chunk_elems*sizeof(T) instead of the whole
//   model's size. See io_utils.hpp's send_chunked()/recv_chunked().
//     sendChunked    → [uint64 total_elems] then [uint64 chunk_len][T×chunk_len]…
//     receiveChunked ← same
//
//   Batch protocol  (server → client data streaming)
//     receive← [uint64 batch_size][uint64 block_size]
//               [T × batch*block  inputs]
//               [T × batch*block  targets]
//
// FederatedClient builds the flat vectors (or chunk callbacks) from its own
// model.parameters() and passes them directly to these methods, keeping
// model ownership in the subclass.

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

    // ── LoRA handshake ────────────────────────────────────────────────────────
    //
    // Sent once right after connect_to_server(), before any round traffic,
    // so a client/server LoRA-config mismatch (dense vs LoRA, or mismatched
    // rank/alpha) is caught with a clear error instead of surfacing later as
    // a confusing element-count mismatch deep in recv_chunked(). Generic
    // wire-level protocol (bool + uint32 + float) — Client<T> stays
    // model-agnostic per the header comment above; FederatedClient/
    // FederatedServer own deciding what config to send/expect.
    //
    //   send    → [uint8 lora_enabled][uint32 rank][float alpha]
    //   receive ← [uint8 accepted][uint32 reason_len][char × reason_len]

    bool sendLoraConfig(bool lora_enabled, uint32_t rank, float alpha) {
        uint8_t enabled = lora_enabled ? 1 : 0;
        return write_exact(sock_, &enabled, sizeof(uint8_t)) &&
               write_exact(sock_, &rank,    sizeof(uint32_t)) &&
               write_exact(sock_, &alpha,   sizeof(float));
    }

    /// Blocks for the server's accept/reject reply. Returns false on a
    /// connection-level failure (caller should treat this like a dropped
    /// connection); `accepted` distinguishes an actual config mismatch
    /// (accepted=false, `reason` explains why) from success.
    bool receiveLoraAck(bool& accepted, std::string& reason) {
        uint8_t ok = 0;
        uint32_t len = 0;
        if (!read_exact(sock_, &ok, sizeof(uint8_t))) return false;
        if (!read_exact(sock_, &len, sizeof(uint32_t))) return false;
        reason.clear();
        if (len > 0) {
            reason.resize(len);
            if (!read_exact(sock_, reason.data(), len)) return false;
        }
        accepted = (ok != 0);
        return true;
    }

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

    // ── Chunked flat-tensor wire protocol ───────────────────────────────────
    //
    // Same logical payload as send()/receive() above (a flattened model's
    // worth of T), but streamed in caller-chosen chunk_elems-sized pieces
    // instead of one [total_bytes][T×N] message — so the peak extra memory
    // for the transfer is ~chunk_elems*sizeof(T), not the whole model's size.
    // This is what makes federated rounds viable on very low-RAM machines;
    // see Modes.md's --chunk-mb for how to size it.
    //
    // Client<T> itself stays model-agnostic here too (consistent with
    // send()/receive()): FederatedClient (client.cpp) supplies get_chunk/
    // on_chunk callbacks that read from / write into student.parameters()
    // directly — Client<T> just forwards to io_utils.hpp's generic
    // send_chunked()/recv_chunked() over this object's own socket.

    bool sendChunked(uint64_t total_elems, size_t chunk_elems,
                     const std::function<void(uint64_t offset, size_t len, T* out)>& get_chunk) {
        return send_chunked<T>(sock_, total_elems, chunk_elems, get_chunk);
    }

    bool receiveChunked(uint64_t expected_total,
                        const std::function<void(uint64_t offset, const T* data, size_t len)>& on_chunk) {
        return recv_chunked<T>(sock_, expected_total, on_chunk);
    }

    // ── Logits wire protocol (federated distillation) ───────────────────────
    //
    // Used instead of send()/receive() when the round exchanges soft
    // predictions on a shared "proxy" batch rather than full weight
    // deltas (see Trainer::compute_logits / Trainer::distill_logits).
    // It's the same flat-float framing as the weight protocol above, with
    // two extra header fields so the receiver can reshape the buffer back
    // into [n_examples, vocab_size] without an out-of-band shape agreement.
    // This is also what lets clients run different model architectures —
    // unlike weight averaging, only the proxy batch + vocab size need to
    // line up, not the parameter layout.
    //
    //   send/receive → [uint64 n_examples][uint64 vocab_size]
    //                  [uint64 total_bytes][T × n_examples*vocab_size]

    bool sendLogits(const std::vector<T>& flat_logits,
                    uint64_t n_examples, uint64_t vocab_size) {
        uint64_t total_bytes = flat_logits.size() * sizeof(T);
        return write_exact(sock_, &n_examples, sizeof(uint64_t)) &&
               write_exact(sock_, &vocab_size, sizeof(uint64_t)) &&
               write_exact(sock_, &total_bytes, sizeof(uint64_t)) &&
               write_exact(sock_, flat_logits.data(), total_bytes);
    }

    bool receiveLogits(std::vector<T>& flat_logits,
                       uint64_t& n_examples, uint64_t& vocab_size) {
        uint64_t total_bytes = 0;
        if (!read_exact(sock_, &n_examples, sizeof(uint64_t)) ||
            !read_exact(sock_, &vocab_size,  sizeof(uint64_t)) ||
            !read_exact(sock_, &total_bytes, sizeof(uint64_t)))
            return false;
        flat_logits.resize(total_bytes / sizeof(T));
        return read_exact(sock_, flat_logits.data(), total_bytes);
    }
};

#endif // CLIENT_HPP__