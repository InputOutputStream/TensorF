#pragma once

/*
 * ProfilerTransport.hpp
 * =====================
 * Three transports, one interface.
 *
 * All transports expose:
 *
 *   bool send(const WireBuffer&)
 *   bool recv(WireBuffer&)        — server side
 *
 * Transport selection:
 *
 *   TCP   — reliable, ordered, connection-oriented
 *           use when: server is remote, internet, different machine
 *           checksum: XXHASH recommended
 *
 *   UNIX  — Unix domain socket, same machine only
 *           use when: server and client on same host
 *           checksum: CRC32 or NONE (kernel guarantees integrity)
 *
 *   UDP   — fire-and-forget, no ack, no ordering
 *           use when: LAN, federated broadcast, latency > reliability
 *           checksum: CRC32 mandatory (UDP has no delivery guarantee)
 *
 * The server side uses ProfilerServer which listens on all 3 simultaneously.
 */

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cerrno>
#include <string>
#include <functional>

#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <poll.h>
#include <sys/types.h>

#include "MessageProtocol.hpp"

// ─── Default ports / paths ───────────────────────────────────────────────────

static constexpr uint16_t DEFAULT_TCP_PORT  = 9731;
static constexpr uint16_t DEFAULT_UDP_PORT  = 9732;
static constexpr const char* DEFAULT_UNIX_PATH = "/tmp/tensorf_profiler.sock";

// ─── Send/recv helpers ───────────────────────────────────────────────────────

// Fully write `len` bytes to fd (handles partial writes)
static bool send_all(int fd, const void* buf, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(buf);
    size_t sent = 0;
    while (sent < len) {
        ssize_t n = ::write(fd, p + sent, len - sent);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        sent += (size_t)n;
    }
    return true;
}

// Fully read `len` bytes from fd (handles partial reads)
static bool recv_all(int fd, void* buf, size_t len, int timeout_ms = 5000) {
    uint8_t* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < len) {
        struct pollfd pfd{ fd, POLLIN, 0 };
        int ready = poll(&pfd, 1, timeout_ms);
        if (ready <= 0) return false;  // timeout or error

        ssize_t n = ::read(fd, p + got, len - got);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        got += (size_t)n;
    }
    return true;
}

// ─── TCP Transport ───────────────────────────────────────────────────────────

class TCPTransport {
public:
    // ── Client side ──────────────────────────────────────────────────────────

    bool connect(const std::string& host, uint16_t port = DEFAULT_TCP_PORT) {
        fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (fd_ < 0) { perror("socket"); return false; }

        // Disable Nagle — we send one big burst, not many small writes
        int one = 1;
        setsockopt(fd_, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        struct sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port   = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
            fprintf(stderr, "[TCP] Invalid address: %s\n", host.c_str());
            close_fd(); return false;
        }
        if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("[TCP] connect"); close_fd(); return false;
        }
        connected_ = true;
        return true;
    }

    bool send(const WireBuffer& wb) {
        if (!connected_) return false;
        // Prefix with 4-byte length so receiver knows how many bytes to read
        uint32_t len = htonl((uint32_t)wb.size);
        return send_all(fd_, &len, 4) && send_all(fd_, wb.data, wb.size);
    }

    // ── Server side ──────────────────────────────────────────────────────────

    bool listen_once(uint16_t port, WireBuffer& out,
                     std::function<void(const ParseResult&, const std::string&)> on_recv) {
        int srv = ::socket(AF_INET, SOCK_STREAM, 0);
        if (srv < 0) return false;

        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port);

        if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0 ||
            ::listen(srv, 8) < 0) {
            perror("[TCP] bind/listen"); ::close(srv); return false;
        }

        struct sockaddr_in cli{}; socklen_t cli_len = sizeof(cli);
        int cfd = ::accept(srv, (sockaddr*)&cli, &cli_len);
        ::close(srv);
        if (cfd < 0) return false;

        std::string peer = inet_ntoa(cli.sin_addr);

        uint32_t net_len;
        if (!recv_all(cfd, &net_len, 4)) { ::close(cfd); return false; }
        uint32_t msg_len = ntohl(net_len);

        if (msg_len > MAX_MSG_SIZE) { ::close(cfd); return false; }
        if (!recv_all(cfd, out.data, msg_len)) { ::close(cfd); return false; }
        out.size = msg_len;
        ::close(cfd);

        auto result = deserialize(out.data, out.size);
        on_recv(result, peer);
        return result.ok;
    }

    void close() { close_fd(); }
    ~TCPTransport() { close_fd(); }

private:
    int  fd_         = -1;
    bool connected_  = false;

    void close_fd() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; connected_ = false; }
    }
};

// ─── Unix Domain Socket Transport ────────────────────────────────────────────

class UnixTransport {
public:
    // ── Client side ──────────────────────────────────────────────────────────

    bool connect(const std::string& path = DEFAULT_UNIX_PATH) {
        fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd_ < 0) { perror("socket"); return false; }

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(fd_, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("[Unix] connect"); close_fd(); return false;
        }
        connected_ = true;
        return true;
    }

    bool send(const WireBuffer& wb) {
        if (!connected_) return false;
        uint32_t len = (uint32_t)wb.size;  // no htonl needed (same machine)
        return send_all(fd_, &len, 4) && send_all(fd_, wb.data, wb.size);
    }

    // ── Server side ──────────────────────────────────────────────────────────

    bool listen_once(const std::string& path, WireBuffer& out,
                     std::function<void(const ParseResult&, const std::string&)> on_recv) {
        ::unlink(path.c_str());  // remove stale socket

        int srv = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (srv < 0) return false;

        struct sockaddr_un addr{};
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, path.c_str(), sizeof(addr.sun_path) - 1);

        if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0 ||
            ::listen(srv, 8) < 0) {
            perror("[Unix] bind/listen"); ::close(srv); return false;
        }

        int cfd = ::accept(srv, nullptr, nullptr);
        ::close(srv);
        if (cfd < 0) return false;

        uint32_t msg_len;
        if (!recv_all(cfd, &msg_len, 4)) { ::close(cfd); return false; }
        if (msg_len > MAX_MSG_SIZE)       { ::close(cfd); return false; }
        if (!recv_all(cfd, out.data, msg_len)) { ::close(cfd); return false; }
        out.size = msg_len;
        ::close(cfd);

        auto result = deserialize(out.data, out.size);
        on_recv(result, "unix:" + path);
        return result.ok;
    }

    void close() { close_fd(); }
    ~UnixTransport() { close_fd(); }

private:
    int  fd_        = -1;
    bool connected_ = false;

    void close_fd() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; connected_ = false; }
    }
};

// ─── UDP Transport ───────────────────────────────────────────────────────────

class UDPTransport {
public:
    // ── Client side ──────────────────────────────────────────────────────────

    bool setup_client(const std::string& host, uint16_t port = DEFAULT_UDP_PORT) {
        fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (fd_ < 0) { perror("socket"); return false; }

        srv_addr_.sin_family = AF_INET;
        srv_addr_.sin_port   = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &srv_addr_.sin_addr) <= 0)
            return false;

        ready_ = true;
        return true;
    }

    // UDP: single sendto — no length prefix (receiver reads full datagram)
    bool send(const WireBuffer& wb) {
        if (!ready_) return false;
        ssize_t n = ::sendto(fd_, wb.data, wb.size, 0,
                             (sockaddr*)&srv_addr_, sizeof(srv_addr_));
        return n == (ssize_t)wb.size;
    }

    // ── Server side ──────────────────────────────────────────────────────────

    bool listen_once(uint16_t port, WireBuffer& out,
                     std::function<void(const ParseResult&, const std::string&)> on_recv) {
        int srv = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (srv < 0) return false;

        int opt = 1;
        setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port        = htons(port);

        if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0) {
            perror("[UDP] bind"); ::close(srv); return false;
        }

        struct sockaddr_in cli{}; socklen_t cli_len = sizeof(cli);
        ssize_t n = ::recvfrom(srv, out.data, MAX_MSG_SIZE, 0,
                               (sockaddr*)&cli, &cli_len);
        ::close(srv);
        if (n <= 0) return false;
        out.size = (size_t)n;

        std::string peer = inet_ntoa(cli.sin_addr);
        auto result = deserialize(out.data, out.size);
        on_recv(result, peer);
        return result.ok;
    }

    void close() {
        if (fd_ >= 0) { ::close(fd_); fd_ = -1; ready_ = false; }
    }
    ~UDPTransport() { close(); }

private:
    int              fd_      = -1;
    bool             ready_   = false;
    struct sockaddr_in srv_addr_{};
};

// ─── Unified client sender ────────────────────────────────────────────────────
// Tries all 3 transports in order: Unix (fastest) → TCP → UDP

struct TransportConfig {
    std::string server_host  = "127.0.0.1";
    uint16_t    tcp_port     = DEFAULT_TCP_PORT;
    uint16_t    udp_port     = DEFAULT_UDP_PORT;
    std::string unix_path    = DEFAULT_UNIX_PATH;
    bool        try_unix     = true;
    bool        try_tcp      = true;
    bool        try_udp      = true;
    bool        internet_mode = false;  // if true: use XXHASH checksum
};

static bool send_profile(const WireBuffer& wb, const TransportConfig& cfg) {
    // 1. Try Unix socket first (same machine — zero network overhead)
    if (cfg.try_unix) {
        UnixTransport ut;
        if (ut.connect(cfg.unix_path)) {
            if (ut.send(wb)) {
                printf("[Transport] Sent via Unix socket (%zu bytes)\n", wb.size);
                return true;
            }
        }
    }

    // 2. TCP (reliable, ordered, remote)
    if (cfg.try_tcp) {
        TCPTransport tt;
        if (tt.connect(cfg.server_host, cfg.tcp_port)) {
            if (tt.send(wb)) {
                printf("[Transport] Sent via TCP to %s:%u (%zu bytes)\n",
                       cfg.server_host.c_str(), cfg.tcp_port, wb.size);
                return true;
            }
        }
    }

    // 3. UDP (fire-and-forget fallback)
    if (cfg.try_udp) {
        UDPTransport ut;
        if (ut.setup_client(cfg.server_host, cfg.udp_port)) {
            if (ut.send(wb)) {
                printf("[Transport] Sent via UDP to %s:%u (%zu bytes)\n",
                       cfg.server_host.c_str(), cfg.udp_port, wb.size);
                return true;
            }
        }
    }

    fprintf(stderr, "[Transport] All transports failed\n");
    return false;
}
        