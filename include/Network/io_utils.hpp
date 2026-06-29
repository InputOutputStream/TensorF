#pragma once

#include <unistd.h>
#include <cstddef>
#include <sys/types.h>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <functional>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <string>

// ── Network bandwidth counters ──────────────────────────────────────────────
// Cumulative bytes moved through read_exact()/write_exact() — every wire call
// in Client.hpp AND Server.hpp funnels through these two functions (flat-
// tensor protocol, batch protocol, logits protocol alike), so instrumenting
// them here covers the whole project for free instead of touching every call
// site separately.
//
// Two views are kept side by side, both updated on every call, so you get
// totals AND breakdown without picking one:
//   - g_net_bytes_sent / g_net_bytes_received: process-wide totals (lock-free
//     atomics) — cheap to read every round, this is "is THIS process
//     network-bound."
//   - g_net_per_fd: same byte counts, broken out by socket fd. On the server,
//     each connected client keeps the SAME fd for its entire session (see
//     Server.hpp's persistent handleClient loop), so this doubles as a
//     per-client breakdown for free — no protocol change, no client-ID
//     handshake needed. On the client there's normally only one fd, so this
//     map just has one entry; the per-fd view is mainly a server-side tool.
//
// `inline` (C++17) so this header can be included from both client.cpp and
// server.cpp without violating ODR — each binary gets its own instance.
//
// Usage pattern (snapshot-diff around a network phase):
//   uint64_t before = net_bytes_sent();
//   auto     before_per_fd = net_per_fd_snapshot();
//   ... do some write_exact() calls on various fds ...
//   uint64_t total_this_phase = net_bytes_sent() - before;
//   for (auto& [fd, now] : net_per_fd_snapshot()) {
//       uint64_t this_fd_this_phase = now.sent - before_per_fd[fd].sent;
//   }
struct NetIoCounters {
    uint64_t sent     = 0;
    uint64_t received = 0;
};

inline std::atomic<uint64_t> g_net_bytes_sent{0};
inline std::atomic<uint64_t> g_net_bytes_received{0};

inline std::mutex                              g_net_per_fd_mtx;
inline std::unordered_map<int, NetIoCounters>  g_net_per_fd;

inline uint64_t net_bytes_sent()      { return g_net_bytes_sent.load(); }
inline uint64_t net_bytes_received()  { return g_net_bytes_received.load(); }
inline void     net_reset_counters()  { g_net_bytes_sent = 0; g_net_bytes_received = 0; }

/// Per-fd counters for one socket (e.g. one client's connection on the server).
inline NetIoCounters net_per_fd(int fd) {
    std::lock_guard<std::mutex> lk(g_net_per_fd_mtx);
    auto it = g_net_per_fd.find(fd);
    return it != g_net_per_fd.end() ? it->second : NetIoCounters{};
}

/// Snapshot of every fd seen so far. Take one before a phase and one after,
/// diff matching fds, for a per-client breakdown of that phase.
inline std::unordered_map<int, NetIoCounters> net_per_fd_snapshot() {
    std::lock_guard<std::mutex> lk(g_net_per_fd_mtx);
    return g_net_per_fd;   // copy — caller diffs against this later
}

/// Call when a connection closes so the map doesn't grow forever across a
/// long-running server that sees many short-lived connections.
inline void net_forget_fd(int fd) {
    std::lock_guard<std::mutex> lk(g_net_per_fd_mtx);
    g_net_per_fd.erase(fd);
}

/// Read exactly `len` bytes from `fd`, retrying on partial reads.
/// Returns false if the connection drops before all bytes are received.
inline bool read_exact(int fd, void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t r = ::read(fd, static_cast<char*>(buf) + done, len - done);
        if (r <= 0) return false;
        done += static_cast<size_t>(r);
    }
    g_net_bytes_received += len;
    {
        std::lock_guard<std::mutex> lk(g_net_per_fd_mtx);
        g_net_per_fd[fd].received += len;
    }
    return true;
}

/// Write exactly `len` bytes to `fd`, retrying on partial writes.
/// Returns false if the connection drops before all bytes are sent.
inline bool write_exact(int fd, const void* buf, size_t len) {
    size_t done = 0;
    while (done < len) {
        ssize_t w = ::write(fd, static_cast<const char*>(buf) + done, len - done);
        if (w <= 0) return false;
        done += static_cast<size_t>(w);
    }
    g_net_bytes_sent += len;
    {
        std::lock_guard<std::mutex> lk(g_net_per_fd_mtx);
        g_net_per_fd[fd].sent += len;
    }
    return true;
}

// ── Chunked parameter streaming ─────────────────────────────────────────────
// Generic, Tensor-agnostic building block for moving a large logical array of
// T (e.g. a flattened model's worth of weights, or one client's delta) over
// the wire WITHOUT ever materializing the whole thing as one contiguous
// buffer on either side — this is what makes low-RAM machines viable: peak
// extra memory for the transfer is ~chunk_elems * sizeof(T), not
// total_elems * sizeof(T).
//
// The caller supplies where the data actually lives (which may be several
// separate tensors, not one array) via callbacks:
//   - send_chunked():  get_chunk(offset, len, T* out) fills out[0..len) with
//                       the source elements starting at flat offset `offset`.
//   - recv_chunked():  on_chunk(offset, data, len) is handed each chunk as it
//                       arrives; do whatever you need with it (copy it
//                       somewhere, add it into a running sum, …).
//
// Wire format: [uint64 total_elems] then repeated
//              [uint64 chunk_len][T × chunk_len]   (last chunk may be smaller)
//
// Sender and receiver choose their OWN chunk_elems independently — the
// receiver just reads whatever chunk_len the sender announces each time, so
// a weak client can send in small chunks without requiring the server to
// receive in small chunks, and vice versa for the broadcast direction.

template<typename T>
bool send_chunked(int fd, uint64_t total_elems, size_t chunk_elems,
                  const std::function<void(uint64_t offset, size_t len, T* out)>& get_chunk) {
    if (!write_exact(fd, &total_elems, sizeof(uint64_t))) return false;
    if (chunk_elems == 0) chunk_elems = 1;   // guard against a degenerate 0-size chunk

    std::vector<T> staging(chunk_elems);
    uint64_t off = 0;
    while (off < total_elems) {
        size_t take = static_cast<size_t>(
            std::min<uint64_t>(chunk_elems, total_elems - off));
        get_chunk(off, take, staging.data());

        uint64_t chunk_len = take;
        if (!write_exact(fd, &chunk_len, sizeof(uint64_t)))               return false;
        if (!write_exact(fd, staging.data(), take * sizeof(T)))           return false;
        off += take;
    }
    return true;
}

/// Receive a chunked stream sent by send_chunked(). Verifies the advertised
/// total against `expected_total` (the receiver already knows how many
/// elements it SHOULD get — e.g. its own model's parameter count — so a
/// mismatched model/checkpoint is caught immediately instead of silently
/// misreading the rest of the stream).
template<typename T>
bool recv_chunked(int fd, uint64_t expected_total,
                  const std::function<void(uint64_t offset, const T* data, size_t len)>& on_chunk) {
    uint64_t total = 0;
    if (!read_exact(fd, &total, sizeof(uint64_t))) return false;
    if (total != expected_total)
        throw std::runtime_error(
            "recv_chunked: size mismatch — stream advertises " + std::to_string(total) +
            " elements, receiver expected " + std::to_string(expected_total) +
            " (mismatched model architecture or checkpoint?)");

    std::vector<T> staging;   // grows to the largest chunk_len actually seen
    uint64_t consumed = 0;
    while (consumed < total) {
        uint64_t chunk_len = 0;
        if (!read_exact(fd, &chunk_len, sizeof(uint64_t))) return false;
        if (staging.size() < chunk_len) staging.resize(chunk_len);
        if (!read_exact(fd, staging.data(), chunk_len * sizeof(T))) return false;
        on_chunk(consumed, staging.data(), static_cast<size_t>(chunk_len));
        consumed += chunk_len;
    }
    return true;
}