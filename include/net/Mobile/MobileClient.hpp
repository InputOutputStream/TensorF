#pragma once
// MobileClient.hpp — federated client designed for phones: flaky links, battery
// and thermal limits, small RAM. Model-agnostic: it talks to FedCoordinator
// through three callbacks (train / get_params / set_params), so wiring it to a
// TensorF model is a few lines (see docs/MOBILE_FEDERATED.md).
//
// What it does for you
//   * connects with a timeout (getaddrinfo, so hostnames work), reconnects with
//     exponential backoff + jitter, and re-identifies by client id;
//   * on first connect it SYNCs with the global model, so every client starts
//     from the same weights (fixes "late joiner starts from the GGUF");
//   * if the link dies after local training but before the update was delivered,
//     the trained update is KEPT and re-offered after reconnect instead of
//     being thrown away and recomputed (battery!);
//   * `can_participate()` is polled before each round: return false while the
//     phone is on battery < x %, hot, on a metered network, in use, ...;
//   * uploads fp16 when the server allows it (half the bytes on a cellular link);
//   * streams chunk-by-chunk straight from/to the model tensors: peak extra
//     memory is one chunk (default 256 KB).

#include "FedWire.hpp"
#include "FedShard.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <mutex>
#include <random>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

namespace tf { namespace fed {

struct ClientConfig {
    std::string host = "127.0.0.1";
    uint16_t    port = 9740;
    std::string client_id;                 // stable per device/install, [A-Za-z0-9._-], no spaces
    std::string algo;                      // task name sent in HELLO (server rejects a mismatch)
    std::string auth_token;
    uint64_t    param_count = 0;           // length of the flat vector exchanged
    uint32_t    model_fingerprint = 0;     // must equal the server's
    Platform    platform =
#ifdef __ANDROID__
        Platform::Android;
#else
        Platform::Linux;
#endif
    uint32_t ram_mb = 0;                   // informational (server dashboard)
    size_t   chunk_kb = 256;               // outgoing chunk size; lower on very low-RAM devices
    bool     metered = false;              // informational flag (cellular)
    float    score = 50.f;                 // device capability 0..100 (profiler); drives how much data the server assigns
    uint32_t mem_budget_mb = 0;            // memory this device lets the shard use (hard cap on its partition; 0 = unlimited)
    uint8_t  shard_pref = 0;               // what data to receive: 0 auto (server decides), 1 partition, 2 partition+halo, 3 whole dataset
    uint8_t  shard_halo = 2;               // halo hops for shard_pref = 2
    bool     prefer_fp16 = true;
    bool     sync_on_connect = true;
    int      connect_timeout_ms = 5000;
    int      io_timeout_ms = 30000;
    int      backoff_min_ms = 500;
    int      backoff_max_ms = 30000;
    int      gate_poll_ms = 2000;
    uint64_t max_rounds = 0;               // stop after N completed rounds (0 = forever)
};

struct ClientHooks {
    /// Run local training starting from the current parameters. Return the number
    /// of samples used (it becomes the averaging weight); 0 = nothing trained, skip.
    std::function<uint64_t(uint64_t base_round)> train;
    /// Copy `len` parameters starting at flat offset `off` into `out`.
    std::function<void(uint64_t off, size_t len, float* out)> get_params;
    /// Write `len` parameters starting at flat offset `off`.
    std::function<void(uint64_t off, const float* data, size_t len)> set_params;
    /// Optional gate (battery / thermal / network policy). Default: always true.
    std::function<bool()> can_participate;
    /// Called after a complete global model has been written through set_params (round result or SYNC).
    std::function<void(uint64_t round)> on_global;
    /// Optional data shard: if wants_shard() is true after connecting, the client enrols, waits for the
    /// server's partitioning and hands the shard to on_shard() before any training.
    std::function<bool()> wants_shard;
    std::function<void(Shard&&)> on_shard;
    std::function<void(const std::string&)> on_log;
};

struct ClientStats {
    std::atomic<uint64_t> rounds_done{0}, connects{0}, reconnects{0}, skips{0}, syncs{0}, bytes_up{0}, bytes_down{0};
};

class MobileClient {
    ClientConfig cfg_; ClientHooks h_;
    int fd_ = -1;
    std::mutex fd_mu_;   // fd_ is closed by run() but shut down by stop() (another thread)
    std::atomic<bool> stop_{false}, fatal_{false}, finished_{false};
    uint64_t base_round_ = 0;
    bool have_unsent_ = false; uint64_t unsent_samples_ = 0;
    bool need_sync_ = false;   // the local params were partially overwritten by an interrupted download
    Dtype up_ = Dtype::F32, down_ = Dtype::F32;
    int wait_ms_ = 120000;
    ClientStats st_;
    std::mt19937 rng_{std::random_device{}()};

public:
    MobileClient(ClientConfig c, ClientHooks h) : cfg_(std::move(c)), h_(std::move(h)) {}
    ~MobileClient() { disconnect(); }
    void stop() { 
        stop_ = true; 
        std::lock_guard<std::mutex> l(fd_mu_); 
        if (fd_ >= 0) ::shutdown(fd_, SHUT_RDWR); }
    const ClientStats& stats() const { return st_; }
    uint64_t base_round() const { return base_round_; }
    bool failed() const { return fatal_; }          // true if the server rejected this client permanently
    bool finished() const { return finished_; }     // true if the server said training is over

    /// Blocking main loop. Returns when stop() is called, max_rounds is reached,
    /// the server reports training finished, or the server permanently rejected us.
    void run() {
        int backoff = cfg_.backoff_min_ms;
        while (!stop_ && !fatal_ && !finished_ && (cfg_.max_rounds == 0 || st_.rounds_done < cfg_.max_rounds)) {
            if (fd_ < 0) {
                if (!connect_and_hello()) 
                {
                    if (fatal_) break;
                    st_.reconnects++;
                    int jitter = int(rng_() % (backoff / 4 + 1));
                    sleep_ms(backoff + jitter);
                    backoff = std::min(backoff * 2, cfg_.backoff_max_ms);
                    continue;
                }
                backoff = cfg_.backoff_min_ms;
                if (h_.wants_shard && h_.wants_shard() && !acquire_shard()) { if (fatal_) break; disconnect(); continue; }
                if ((need_sync_ || (cfg_.sync_on_connect && !have_unsent_)) && !do_sync()) { disconnect(); continue; }
            }
            if (h_.can_participate && !h_.can_participate()) { sleep_ms(cfg_.gate_poll_ms); continue; }
            if (!have_unsent_) {
                uint64_t n = h_.train ? h_.train(base_round_) : 0;
                if (n == 0) { sleep_ms(1000); continue; }
                unsent_samples_ = n; have_unsent_ = true;
            }
            if (!do_round()) disconnect();
        }
        if (fd_ >= 0) { Wr b; send_msg(fd_, Msg::Bye, b); }
        disconnect();
    }

private:
    void log(const std::string& s) { if (h_.on_log) h_.on_log(s); }
    void sleep_ms(int ms) {
        for (int t = 0; t < ms && !stop_; t += 50) std::this_thread::sleep_for(std::chrono::milliseconds(std::min(50, ms - t)));
    }
    void disconnect() { 
        std::lock_guard<std::mutex> l(fd_mu_); 
        if (fd_ >= 0) 
        { 
            net_forget_fd(fd_); 
            ::close(fd_); 
            fd_ = -1; 
        } 
    }

    bool tcp_connect() {
        addrinfo hints{}; hints.ai_family = AF_UNSPEC; hints.ai_socktype = SOCK_STREAM;
        addrinfo* res = nullptr;
        if (::getaddrinfo(cfg_.host.c_str(), std::to_string(cfg_.port).c_str(), &hints, &res) != 0) return false;
        int fd = -1;
        for (addrinfo* a = res; a && fd < 0; a = a->ai_next) 
        {
            int s = ::socket(a->ai_family, a->ai_socktype, a->ai_protocol);
            if (s < 0) continue;
            int fl = ::fcntl(s, F_GETFL, 0); ::fcntl(s, F_SETFL, fl | O_NONBLOCK);
            int rc = ::connect(s, a->ai_addr, a->ai_addrlen);
            if (rc < 0 && errno == EINPROGRESS) {
                pollfd p{s, POLLOUT, 0};
                if (::poll(&p, 1, cfg_.connect_timeout_ms) == 1) {
                    int e = 0; socklen_t el = sizeof e; ::getsockopt(s, SOL_SOCKET, SO_ERROR, &e, &el);
                    rc = e == 0 ? 0 : -1;
                } else rc = -1;
            }
            if (rc == 0) { ::fcntl(s, F_SETFL, fl); fd = s; } else ::close(s);
        }
        ::freeaddrinfo(res);
        if (fd < 0) return false;
        int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
        set_socket_keepalive(fd);
        { std::lock_guard<std::mutex> l(fd_mu_); fd_ = fd; }
        return true;
    }

    bool connect_and_hello() {
        if (!tcp_connect()) { log("connect failed"); return false; }
        set_socket_timeouts(fd_, cfg_.io_timeout_ms, cfg_.io_timeout_ms);
        Wr b; b.str(cfg_.client_id); b.str(cfg_.auth_token); b.str(cfg_.algo); b.u64(cfg_.param_count); b.u32(cfg_.model_fingerprint);
        b.u8(cfg_.prefer_fp16 ? 3 : 1); b.u32(uint32_t(cfg_.chunk_kb)); b.u32(cfg_.ram_mb);
        b.u8(uint8_t(cfg_.platform)); b.u8(cfg_.metered ? 1 : 0);
        b.u16(uint16_t(std::max(0.f, std::min(cfg_.score, 100.f)) * 10.f)); b.u32(cfg_.mem_budget_mb);
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!send_msg(fd_, Msg::Hello, b) || !recv_msg(fd_, t, body, err) || t != Msg::Welcome) { log("handshake failed: " + err); disconnect(); return false; }
        Rd r(body); uint8_t ok = r.u8(); std::string reason = r.str();
        if (!ok) {
            log("rejected by server: " + reason);
            if (reason != "server full") fatal_ = true;
            disconnect(); return false;
        }
        uint64_t gr = r.u64(); up_ = Dtype(r.u8()); down_ = Dtype(r.u8()); r.u32();
        uint32_t io = r.u32(), rdy = r.u32(), upl = r.u32();
        if (!r.done()) { disconnect(); return false; }
        (void)gr; (void)io;
        wait_ms_ = int(std::max<uint32_t>(rdy, upl)) + cfg_.io_timeout_ms;
        set_socket_timeouts(fd_, wait_ms_, cfg_.io_timeout_ms);   // GO / RESULT may legitimately take a while
        st_.connects++;
        return true;
    }

    /// Enrol + wait for the server to partition the data, then receive this device's shard.
    bool acquire_shard() {
        while (!stop_) {
            Wr b; b.u8(cfg_.shard_pref); b.u8(cfg_.shard_halo); Msg t; std::vector<uint8_t> body; std::string err;
            if (!send_msg(fd_, Msg::ShardReq, b) || !recv_msg(fd_, t, body, err)) return false;
            if (t == Msg::ShardWait) {
                Rd r(body); uint32_t retry = r.u32(); uint8_t fl = r.u8(); std::string why = r.str();
                if (fl & 3) { log("shard refused: " + why); fatal_ = true; return false; }
                sleep_ms(int(std::max<uint32_t>(retry, 100)));
                continue;
            }
            if (t != Msg::ShardHdr) return false;
            Shard sh; uint64_t before = net_bytes_received();
            if (!recv_shard(fd_, body, sh, err)) { log("shard transfer failed: " + err); return false; }
            st_.bytes_down += net_bytes_received() - before;
            log("received shard: " + std::to_string(sh.bytes() / 1024) + " KB");
            if (h_.on_shard) h_.on_shard(std::move(sh));
            return true;
        }
        return false;
    }

    bool recv_global(const std::string& what) {
        std::string err; uint64_t off_seen = 0;
        uint64_t before = net_bytes_received();
        bool ok = recv_stream(fd_, down_, cfg_.param_count, [&](uint64_t off, const float* d, size_t n) {
            if (h_.set_params) h_.set_params(off, d, n);
            off_seen = off + n; return true; }, err);
        st_.bytes_down += net_bytes_received() - before;
        if (!ok) log(what + ": " + err + " after " + std::to_string(off_seen) + " elems");
        return ok;
    }

    bool do_sync() {
        Wr b; Msg t; std::vector<uint8_t> body; std::string err;
        if (!send_msg(fd_, Msg::SyncReq, b) || !recv_msg(fd_, t, body, err) || t != Msg::Result) return false;
        Rd r(body); uint64_t round = r.u64(); r.u32(); uint8_t status = r.u8(); down_ = Dtype(r.u8());
        if (!r.done() || status != 0) return false;
        need_sync_ = true;                       // cleared only once the whole model arrived
        if (!recv_global("sync")) return false;
        need_sync_ = false; base_round_ = round; have_unsent_ = false; st_.syncs++;
        if (h_.on_global) h_.on_global(round);
        return true;
    }

    bool do_round() {
        Wr b; b.u64(unsent_samples_); b.u64(base_round_); b.u8(0);
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!send_msg(fd_, Msg::Ready, b) || !recv_msg(fd_, t, body, err)) return false;

        if (t == Msg::Skip) {
            Rd r(body); r.u64(); uint32_t retry = r.u32(); uint8_t fl = r.u8(); std::string why = r.str();
            st_.skips++;
            if (fl & 2) { finished_ = true; return true; }
            if (fl & 1) { have_unsent_ = false; return do_sync(); }   // stale: our update is useless, re-sync and retrain
            log("skipped: " + why);
            sleep_ms(int(std::max<uint32_t>(retry, 200)));
            return true;                                              // keep the trained update, re-offer it
        }
        if (t != Msg::Go) return false;
        Rd g(body); uint64_t round = g.u64(); up_ = Dtype(g.u8()); (void)round;

        const size_t chunk = std::max<size_t>(1, cfg_.chunk_kb * 1024 / 4);
        uint64_t before = net_bytes_sent();
        bool ok = send_stream(fd_, up_, cfg_.param_count, chunk, [&](uint64_t off, size_t n, float* out) {
            if (h_.get_params) h_.get_params(off, n, out); });
        st_.bytes_up += net_bytes_sent() - before;
        if (!ok) return false;                                        // partial upload: server dropped it, keep it in have_unsent_
        // Fully sent: the server may or may not have counted it. Re-offering it
        // later would double-count, so from here on we never re-send it; if the
        // result is lost we re-sync with the global model instead.
        have_unsent_ = false; need_sync_ = true;

        if (!recv_msg(fd_, t, body, err) || t != Msg::Result) return false;
        Rd r(body); uint64_t rid = r.u64(); r.u32(); uint8_t status = r.u8(); down_ = Dtype(r.u8());
        if (!r.done()) return false;
        if (status != 0) { need_sync_ = false; have_unsent_ = true; return true; }   // aborted round: model untouched, offer again
        if (!recv_global("global")) return false; // model is half-overwritten: must re-sync after reconnect
        need_sync_ = false;
        base_round_ = rid;
        if (h_.on_global) h_.on_global(rid); have_unsent_ = false; st_.rounds_done++;
        return true;
    }
};

}} // namespace tf::fed
