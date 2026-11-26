#pragma once
// FedCoordinator.hpp — model-agnostic federated-averaging server built for
// unreliable clients (phones). It aggregates flat float vectors; what those
// floats mean (full weights, LoRA adapters, ...) is the caller's business, so
// the same coordinator serves GPT, Llama, a GCN or the synthetic test below.
//
// Round lifecycle (all decisions under one mutex, driven by a 20 ms ticker):
//
//   clients send READY(n_samples, base_round)
//     -> they wait in `pending`; the COHORT closes when
//          pending >= max_clients, or pending >= min_clients for `grace_ms`.
//        Stale clients (base_round too old) get SKIP(needs_sync) instead.
//     -> every cohort member gets GO and streams its update. Only cohort members
//        may write into the accumulator, so a late client can never corrupt a
//        round in progress (the race the legacy Server.hpp had).
//   round ends when all members are done/dropped, or at upload_deadline_ms
//     (members still streaming are cut off and counted as dropped).
//
// Dropout handling is exact, not approximate: chunks arrive in order, so a
// client that disconnects after `k` elements has contributed to the prefix
// [0,k). Per element i the divisor is
//        W_completed + sum(weight of dropped clients whose prefix > i)
// which is what a client that had *finished* would have produced for those
// elements. Memory stays O(model) (accumulator) + O(model) (immutable result).
//
// Results are published as shared_ptr<const vector<float>> snapshots, so a slow
// phone can finish downloading round N while round N+1 is already aggregating.

#include "FedWire.hpp"
#include "FedShard.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <thread>

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>

namespace tf { namespace fed {

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::milliseconds;

struct RoundReport {
    uint64_t round = 0;
    bool     aborted = false;
    uint32_t participants = 0;   // clients that completed their upload
    uint32_t dropped = 0;        // clients lost mid-round
    double   weight_sum = 0;
    double   ms = 0;
};

struct CoordinatorConfig {
    std::string bind_addr = "0.0.0.0";
    uint16_t    port = 9740;               // 0 = pick a free port (see port())
    std::string algo;                      // algorithm/task name; clients must send the same ("" = accept any)
    uint64_t    param_count = 0;           // length of the flat vector being averaged
    uint32_t    model_fingerprint = 0;     // opaque; client must send the same value
    std::string auth_token;                // empty = no auth. NOT encryption: use a tunnel/VPN off-LAN.

    enum class Weighting { Samples, Uniform };
    Weighting weighting = Weighting::Samples;   // Samples: weight = n_samples (FedAvg); Uniform: every client counts 1
    // Server-side optimizer hook (FedAvgM / FedAdam / ...): called with the previous global vector and the
    // freshly averaged one; rewrite `averaged` in place into the new global vector. Runs under the round
    // lock, so keep it O(P). See FedStrategies.hpp.
    std::function<void(uint64_t round, const std::vector<float>& old_global, std::vector<float>& averaged)> post_aggregate;

    // Data shards (optional). When set, devices send ShardReq and receive only their own part of the data.
    std::shared_ptr<ShardProvider> shards;
    size_t   enroll_min = 0;               // devices needed before partitioning (0 = min_clients)
    int      enroll_grace_ms = 1000;       // wait this long for more devices once enroll_min is reached

    size_t   min_clients = 2;
    size_t   max_clients = 64;
    int      grace_ms = 1000;              // wait this long for more clients once min_clients is reached
    int      ready_timeout_ms = 60000;     // a READY that finds no cohort in time is answered SKIP
    int      upload_deadline_ms = 120000;  // per-round cut-off for updates
    int      hello_timeout_ms = 5000;
    int      io_timeout_ms = 30000;        // stall limit while streaming
    int      session_idle_ms = 30 * 60 * 1000;
    uint64_t max_staleness = 2;            // rounds behind the global model before a client must sync
    double   staleness_decay = 1.0;        // weight *= decay^staleness (1.0 = off)
    bool     allow_fp16_upload = true;     // halves upload bytes when the client supports it
    bool     fp16_download = false;        // lossy for the client's copy; off by default
    size_t   chunk_elems = 1u << 18;
    size_t   max_sessions = 256;
    uint64_t max_rounds = 0;               // 0 = unlimited

    std::function<void(const std::string& json_line)> on_event;               // default: "@@TFEVT {json}" on stdout
    std::function<void(const RoundReport&, const std::vector<float>&)> on_round;
};

// tiny JSON object builder (no external dependency)
class J {
    std::string s_ = "{"; bool first_ = true;
    void key(const char* k) { if (!first_) s_ += ','; first_ = false; s_ += '"'; s_ += k; s_ += "\":"; }
public:
    J& kv(const char* k, const std::string& v) {
        key(k); s_ += '"';
        for (char c : v) { if (c == '"' || c == '\\') { s_ += '\\'; s_ += c; } else if (static_cast<unsigned char>(c) < 0x20) s_ += ' '; else s_ += c; }
        s_ += '"'; return *this; }
    J& kv(const char* k, const char* v) { return kv(k, std::string(v)); }
    J& kv(const char* k, uint64_t v) { key(k); s_ += std::to_string(v); return *this; }
    J& kv(const char* k, int v) { key(k); s_ += std::to_string(v); return *this; }
    J& kv(const char* k, double v) { key(k); char b[40]; std::snprintf(b, sizeof b, "%.3f", v); s_ += b; return *this; }
    J& kv(const char* k, bool v) { key(k); s_ += v ? "true" : "false"; return *this; }
    std::string str() const { return s_ + "}"; }
};

class FedCoordinator {
    struct Sock {
        std::mutex mu; int fd;
        explicit Sock(int f) : fd(f) {}
        void shut() { std::lock_guard<std::mutex> l(mu); if (fd >= 0) ::shutdown(fd, SHUT_RDWR); }
        void close() { std::lock_guard<std::mutex> l(mu); if (fd >= 0) { net_forget_fd(fd); ::close(fd); fd = -1; } }
    };
    struct RoundCtx;
    enum class Dec { Waiting, Go, Skip };
    struct Member {
        std::string id; std::shared_ptr<Sock> sock;
        double weight = 1, weight_eff = 1; uint64_t base = 0;
        Dec dec = Dec::Waiting; std::string skip_reason; uint32_t retry_ms = 0; bool needs_sync = false;
        std::shared_ptr<RoundCtx> ctx;
        bool active = false, done = false, dropped = false; uint64_t applied = 0;
    };
    struct RoundCtx {
        uint64_t id = 0; std::vector<std::shared_ptr<Member>> members;
        Clock::time_point started, deadline;
        bool finalized = false, aborted = false; uint32_t n_participants = 0;
        std::shared_ptr<const std::vector<float>> result;
    };
    struct Session {
        std::shared_ptr<Sock> sock; std::string id; EnrollInfo caps; std::thread th; std::atomic<bool> finished{false};
    };

    CoordinatorConfig cfg_;
    std::mutex m_; std::condition_variable cv_;
    std::shared_ptr<const std::vector<float>> global_;
    uint64_t global_round_ = 0;
    std::vector<float> accum_;
    std::vector<std::shared_ptr<Member>> pending_;
    std::shared_ptr<RoundCtx> cur_;
    bool have_min_ = false; Clock::time_point min_since_;
    bool finished_ = false;
    std::vector<std::shared_ptr<Session>> sessions_;
    std::map<std::string, std::shared_ptr<Session>> by_id_;
    std::vector<std::string> evq_;
    std::vector<std::pair<RoundReport, std::shared_ptr<const std::vector<float>>>> repq_;
    std::mutex emit_mu_;

    enum class Enroll { Open, Preparing, Ready, Failed };
    Enroll enroll_ = Enroll::Open; std::vector<EnrollInfo> enrolled_; Clock::time_point enroll_last_; std::string enroll_err_;
    std::thread prep_th_;

    std::atomic<bool> stopping_{false}; std::atomic<bool> started_{false};
    int listen_fd_ = -1; uint16_t bound_port_ = 0;
    std::thread accept_th_, tick_th_;

public:
    explicit FedCoordinator(CoordinatorConfig c) : cfg_(std::move(c)) {
        if (cfg_.min_clients == 0) cfg_.min_clients = 1;
        if (cfg_.max_clients < cfg_.min_clients) cfg_.max_clients = cfg_.min_clients;
        accum_.assign(cfg_.param_count, 0.f);
        if (cfg_.shards) cfg_.shards->set_event_sink([this](const std::string& j) { emit(j); });
        global_ = std::make_shared<std::vector<float>>(cfg_.param_count, 0.f);
    }
    ~FedCoordinator() { stop(); }
    FedCoordinator(const FedCoordinator&) = delete;
    FedCoordinator& operator=(const FedCoordinator&) = delete;

    /// Must be called before start() (or while no round is running).
    bool set_initial_global(const std::vector<float>& g) {
        if (g.size() != cfg_.param_count) return false;
        std::lock_guard<std::mutex> l(m_);
        global_ = std::make_shared<std::vector<float>>(g);
        return true;
    }
    std::shared_ptr<const std::vector<float>> global() { std::lock_guard<std::mutex> l(m_); return global_; }
    uint64_t round() { std::lock_guard<std::mutex> l(m_); return global_round_; }
    bool finished() { std::lock_guard<std::mutex> l(m_); return finished_; }
    uint16_t port() const { return bound_port_; }

    bool wait_for_round(uint64_t n, int timeout_ms) {
        std::unique_lock<std::mutex> l(m_);
        return cv_.wait_for(l, Ms(timeout_ms), [&] { return global_round_ >= n || stopping_.load(); });
    }
    /// Drop a client's connection (it is expected to reconnect). Returns true if it was connected.
    bool kick(const std::string& id) {
        std::shared_ptr<Session> s;
        { std::lock_guard<std::mutex> l(m_); auto it = by_id_.find(id); if (it == by_id_.end()) return false; s = it->second; }
        s->sock->shut(); return true;
    }
    size_t connected() { std::lock_guard<std::mutex> l(m_); return by_id_.size(); }
    /// Thread-safe: emit an application event (JSON object string) through the same channel as coordinator events.
    void publish_event(const std::string& json) { emit(json); }

    bool start() {
        if (started_.exchange(true)) return false;
        listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
        if (listen_fd_ < 0) { perror("[fed] socket"); return false; }
        int one = 1; ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(cfg_.port);
        if (::inet_pton(AF_INET, cfg_.bind_addr.c_str(), &a.sin_addr) != 1) { std::fprintf(stderr, "[fed] bad bind address\n"); return false; }
        if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&a), sizeof a) < 0 || ::listen(listen_fd_, 128) < 0) {
            perror("[fed] bind/listen"); ::close(listen_fd_); listen_fd_ = -1; return false; }
        socklen_t al = sizeof a; ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&a), &al); bound_port_ = ntohs(a.sin_port);
        accept_th_ = std::thread([this] { accept_loop(); });
        tick_th_   = std::thread([this] { tick_loop(); });
        emit(J().kv("type", "fed_server_started").kv("port", int(bound_port_)).kv("algo", cfg_.algo).kv("params", cfg_.param_count)
                 .kv("min_clients", uint64_t(cfg_.min_clients)).str());
        return true;
    }

    void stop() {
        if (!started_.load() || stopping_.exchange(true)) return;
        cv_.notify_all();
        if (accept_th_.joinable()) accept_th_.join();
        if (tick_th_.joinable()) tick_th_.join();
        if (prep_th_.joinable()) prep_th_.join();
        std::vector<std::shared_ptr<Session>> ss;
        { std::lock_guard<std::mutex> l(m_); ss = sessions_; }
        for (auto& s : ss) s->sock->shut();
        cv_.notify_all();
        for (auto& s : ss) if (s->th.joinable()) s->th.join();
        if (listen_fd_ >= 0) { ::close(listen_fd_); listen_fd_ = -1; }
    }

private:
    // ── events ---------------------------------------------------------------
    void emit(const std::string& line) {
        std::lock_guard<std::mutex> l(emit_mu_);
        if (cfg_.on_event) cfg_.on_event(line);
        else { std::printf("@@TFEVT %s\n", line.c_str()); std::fflush(stdout); }
    }
    void flush() {   // call WITHOUT m_ held
        std::vector<std::string> ev; decltype(repq_) rp;
        { std::lock_guard<std::mutex> l(m_); ev.swap(evq_); rp.swap(repq_); }
        for (auto& e : ev) emit(e);
        if (cfg_.on_round) for (auto& r : rp) cfg_.on_round(r.first, *r.second);
    }

    // ── threads ---------------------------------------------------------------
    void accept_loop() {
        while (!stopping_) {
            pollfd p{listen_fd_, POLLIN, 0};
            int r = ::poll(&p, 1, 100);
            reap();
            if (r <= 0) continue;
            int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) continue;
            int one = 1; ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
            set_socket_keepalive(fd);
            auto s = std::make_shared<Session>(); s->sock = std::make_shared<Sock>(fd);
            bool full;
            { std::lock_guard<std::mutex> l(m_); full = by_id_.size() + 0 >= cfg_.max_sessions || sessions_.size() >= cfg_.max_sessions * 2;
              if (!full) sessions_.push_back(s); }
            if (full) { Wr b; b.u8(0); b.str("server full"); send_msg(fd, Msg::Welcome, b); s->sock->close(); continue; }
            s->th = std::thread([this, s] { session_main(s); s->finished = true; });
        }
    }
    void reap() {
        std::vector<std::shared_ptr<Session>> dead;
        { std::lock_guard<std::mutex> l(m_);
          for (auto it = sessions_.begin(); it != sessions_.end();) {
              if ((*it)->finished) { dead.push_back(*it); it = sessions_.erase(it); } else ++it; } }
        for (auto& s : dead) if (s->th.joinable()) s->th.join();
    }
    void tick_loop() {
        while (!stopping_) {
            { std::unique_lock<std::mutex> l(m_); tick_locked(Clock::now()); }
            flush();
            std::this_thread::sleep_for(Ms(20));
        }
    }

    // ── round state machine (m_ held) -----------------------------------------
    void tick_locked(Clock::time_point now) {
        if (cur_) {
            bool all = true;
            for (auto& m : cur_->members) if (!m->done && !m->dropped) { all = false; break; }
            if (!all && now >= cur_->deadline) {
                for (auto& m : cur_->members) if (m->active) {
                    m->active = false; m->dropped = true; m->sock->shut();
                    evq_.push_back(J().kv("type", "fed_client_dropped").kv("client", m->id).kv("round", cur_->id)
                                      .kv("reason", "upload deadline").kv("applied", m->applied).str());
                }
                all = true;
            }
            if (all) finalize_locked(now);
        }
        maybe_prepare_locked(now);
        if (!cur_) close_cohort_locked(now);
    }

    void maybe_prepare_locked(Clock::time_point now) {
        if (!cfg_.shards || enroll_ != Enroll::Open || enrolled_.empty()) return;
        const size_t need = cfg_.enroll_min ? cfg_.enroll_min : cfg_.min_clients;
        if (!(enrolled_.size() >= cfg_.max_clients || (enrolled_.size() >= need && now - enroll_last_ >= Ms(cfg_.enroll_grace_ms)))) return;
        enroll_ = Enroll::Preparing;
        std::vector<EnrollInfo> snap = enrolled_;
        std::sort(snap.begin(), snap.end(), [](const EnrollInfo& a, const EnrollInfo& b) { return a.id < b.id; });   // deterministic order
        prep_th_ = std::thread([this, snap] {
            auto t0 = Clock::now(); std::string err; bool ok = true;
            try { cfg_.shards->prepare(snap); } catch (const std::exception& e) { ok = false; err = e.what(); }
            double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            std::lock_guard<std::mutex> l(m_);
            if (ok) { enroll_ = Enroll::Ready; evq_.push_back(J().kv("type", "fed_shards_ready").kv("devices", uint64_t(snap.size())).kv("ms", ms).str()); }
            else { enroll_ = Enroll::Failed; enroll_err_ = err; evq_.push_back(J().kv("type", "fed_shards_failed").kv("reason", err).str()); }
            cv_.notify_all();
        });
    }

    void close_cohort_locked(Clock::time_point now) {
        if (cur_ || finished_) return;
        if (pending_.size() < cfg_.min_clients) { have_min_ = false; return; }
        if (!have_min_) { have_min_ = true; min_since_ = now; }
        if (pending_.size() < cfg_.max_clients && now - min_since_ < Ms(cfg_.grace_ms)) return;

        std::vector<std::shared_ptr<Member>> fresh;
        for (auto& p : pending_) {
            uint64_t stale = p->base > global_round_ ? UINT64_MAX : global_round_ - p->base;
            if (stale > cfg_.max_staleness) {
                p->dec = Dec::Skip; p->skip_reason = "stale model, sync first"; p->needs_sync = true; p->retry_ms = 0;
            } else { p->weight_eff = p->weight * std::pow(cfg_.staleness_decay, double(stale)); fresh.push_back(p); }
        }
        pending_.clear();
        have_min_ = false;
        if (fresh.size() < cfg_.min_clients) { pending_ = fresh; cv_.notify_all(); return; }

        auto ctx = std::make_shared<RoundCtx>();
        ctx->id = global_round_ + 1; ctx->started = now; ctx->deadline = now + Ms(cfg_.upload_deadline_ms);
        ctx->members = fresh;
        for (auto& m : fresh) { m->dec = Dec::Go; m->ctx = ctx; m->active = true; }
        cur_ = ctx;
        evq_.push_back(J().kv("type", "fed_round_started").kv("round", ctx->id).kv("clients", uint64_t(fresh.size())).str());
        cv_.notify_all();
    }

    void finalize_locked(Clock::time_point now) {
        auto ctx = cur_;
        RoundReport rep; rep.round = ctx->id;
        double W = 0; std::vector<std::pair<uint64_t, double>> partial;   // (prefix, weight)
        for (auto& m : ctx->members) {
            if (m->done) { W += m->weight_eff; rep.participants++; }
            else if (m->dropped) { rep.dropped++; if (m->applied > 0) partial.emplace_back(m->applied, m->weight_eff); }
        }
        rep.weight_sum = W;
        rep.ms = std::chrono::duration<double, std::milli>(now - ctx->started).count();
        if (W <= 0) {                           // nobody finished: keep the old global model
            ctx->aborted = true; rep.aborted = true;
            std::fill(accum_.begin(), accum_.end(), 0.f);
        } else {
            std::sort(partial.begin(), partial.end());
            double extra = 0; for (auto& p : partial) extra += p.second;
            auto nv = std::make_shared<std::vector<float>>(cfg_.param_count);
            uint64_t pos = 0;
            auto fill = [&](uint64_t end, double denom) {
                float inv = static_cast<float>(1.0 / denom);
                for (uint64_t i = pos; i < end; i++) (*nv)[i] = accum_[i] * inv;
                pos = end;
            };
            for (auto& p : partial) { fill(std::min<uint64_t>(p.first, cfg_.param_count), W + extra); extra -= p.second; }
            fill(cfg_.param_count, W);
            if (cfg_.post_aggregate) cfg_.post_aggregate(ctx->id, *global_, *nv);
            std::fill(accum_.begin(), accum_.end(), 0.f);
            global_ = nv; global_round_ = ctx->id; ctx->result = nv;
        }
        ctx->n_participants = rep.participants; ctx->finalized = true;
        for (auto& m : ctx->members) m->active = false;
        cur_.reset();
        if (cfg_.max_rounds && global_round_ >= cfg_.max_rounds) finished_ = true;
        evq_.push_back(J().kv("type", rep.aborted ? "fed_round_aborted" : "fed_round_done").kv("round", rep.round)
                          .kv("participants", uint64_t(rep.participants)).kv("dropped", uint64_t(rep.dropped))
                          .kv("weight", rep.weight_sum).kv("ms", rep.ms).str());
        if (!rep.aborted) repq_.emplace_back(rep, global_);
        cv_.notify_all();
    }

    // ── per-connection logic ---------------------------------------------------
    bool send_skip(int fd, uint64_t round, uint32_t retry_ms, bool needs_sync, bool fin, const std::string& why) {
        Wr b; b.u64(round); b.u32(retry_ms); b.u8(uint8_t((needs_sync ? 1 : 0) | (fin ? 2 : 0))); b.str(why);
        return send_msg(fd, Msg::Skip, b);
    }

    void session_main(const std::shared_ptr<Session>& s) {
        const int fd = s->sock->fd;
        set_socket_timeouts(fd, cfg_.hello_timeout_ms, cfg_.hello_timeout_ms);
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!recv_msg(fd, t, body, err) || t != Msg::Hello) { s->sock->close(); return; }
        Rd r(body);
        std::string id = r.str(), token = r.str(), algo = r.str();
        uint64_t pc = r.u64(); uint32_t fp = r.u32(); uint8_t mask = r.u8(); uint32_t chunk_kb = r.u32();
        uint32_t ram_mb = r.u32(); uint8_t plat = r.u8(); uint8_t flags = r.u8(); uint16_t score10 = r.u16(); uint32_t mem_mb = r.u32();
        std::string reject;
        bool id_ok = !id.empty() && std::all_of(id.begin(), id.end(), [](unsigned char c) { return c >= 0x21 && c < 0x7F; });
        if (!r.done() || !id_ok) reject = "malformed HELLO";
        else if (!cfg_.auth_token.empty() && !ct_equal(token, cfg_.auth_token)) reject = "authentication failed";
        else if (!cfg_.algo.empty() && algo != cfg_.algo) reject = "algorithm mismatch: server runs '" + cfg_.algo + "', client sent '" + algo + "'";
        else if (pc != cfg_.param_count) reject = "parameter count mismatch: server " + std::to_string(cfg_.param_count) + ", client " + std::to_string(pc) + " (different model / LoRA config?)";
        else if (fp != cfg_.model_fingerprint) reject = "model fingerprint mismatch";
        if (!reject.empty()) {
            Wr b; b.u8(0); b.str(reject); send_msg(fd, Msg::Welcome, b);
            emit(J().kv("type", "fed_client_rejected").kv("client", id).kv("reason", reject).str());
            s->sock->close(); return;
        }
        Dtype up = (mask & 2) && cfg_.allow_fp16_upload ? Dtype::F16 : Dtype::F32;
        Dtype down = (mask & 2) && cfg_.fp16_download ? Dtype::F16 : Dtype::F32;
        size_t chunk = cfg_.chunk_elems;
        if (chunk_kb) chunk = std::max<size_t>(256, std::min<size_t>(chunk, size_t(chunk_kb) * 1024 / 4));
        uint64_t gr;
        {   std::lock_guard<std::mutex> l(m_);
            auto it = by_id_.find(id);
            if (it != by_id_.end()) it->second->sock->shut();     // reconnect: evict the stale half-dead session
            s->id = id; by_id_[id] = s; gr = global_round_;
            s->caps.id = id; s->caps.score = float(score10) / 10.f; s->caps.mem_budget_mb = mem_mb; s->caps.platform = plat;
        }
        {   Wr b; b.u8(1); b.str(""); b.u64(gr); b.u8(uint8_t(up)); b.u8(uint8_t(down));
            b.u32(uint32_t(chunk * 4 / 1024)); b.u32(cfg_.io_timeout_ms); b.u32(cfg_.ready_timeout_ms); b.u32(cfg_.upload_deadline_ms);
            if (!send_msg(fd, Msg::Welcome, b)) { leave(s); return; } }
        emit(J().kv("type", "fed_client_joined").kv("client", id).kv("platform", int(plat)).kv("ram_mb", uint64_t(ram_mb))
                 .kv("metered", bool(flags & 1)).kv("score", double(score10) / 10.0).kv("mem_mb", uint64_t(mem_mb)).kv("up_dtype", up == Dtype::F16 ? "fp16" : "fp32").str());

        std::string why = "closed";
        while (!stopping_) {
            set_socket_timeouts(fd, cfg_.session_idle_ms, cfg_.io_timeout_ms);
            if (!recv_msg(fd, t, body, err)) { why = err; break; }
            bool ok = true;
            if (t == Msg::Ready) ok = handle_ready(s, body, up, down, chunk);
            else if (t == Msg::SyncReq) ok = handle_sync(fd, down, chunk);
            else if (t == Msg::ShardReq) ok = handle_shard_req(s, body, chunk);
            else if (t == Msg::Bye) { why = "bye"; break; }
            else { why = "unexpected message"; break; }
            if (!ok) { why = "session error"; break; }
        }
        leave(s, why);
    }

    void leave(const std::shared_ptr<Session>& s, const std::string& why = "closed") {
        NetIoCounters c = net_per_fd(s->sock->fd);
        { std::lock_guard<std::mutex> l(m_);
          auto it = by_id_.find(s->id); if (it != by_id_.end() && it->second == s) by_id_.erase(it); }
        if (!s->id.empty())
            emit(J().kv("type", "fed_client_left").kv("client", s->id).kv("reason", why)
                     .kv("recv_bytes", c.received).kv("sent_bytes", c.sent).str());
        s->sock->close();
    }

    bool send_shard_wait(int fd, uint32_t retry_ms, uint8_t flags, const std::string& why) {
        Wr b; b.u32(retry_ms); b.u8(flags); b.str(why); return send_msg(fd, Msg::ShardWait, b);
    }

    bool handle_shard_req(const std::shared_ptr<Session>& s, const std::vector<uint8_t>& body, size_t chunk) {
        const int fd = s->sock->fd;
        { Rd r(body); uint8_t pref = r.u8(), hops = r.u8(); if (!r.done() || pref > 3) return false; s->caps.shard_pref = pref; s->caps.halo_hops = hops; }
        if (!cfg_.shards) return send_shard_wait(fd, 0, 2, "this server does not distribute data shards");
        uint32_t retry = 500; std::string why; uint8_t flags = 0; bool ready = false;
        {   std::lock_guard<std::mutex> l(m_);
            switch (enroll_) {
            case Enroll::Open: {
                bool known = std::any_of(enrolled_.begin(), enrolled_.end(), [&](const EnrollInfo& e) { return e.id == s->id; });
                if (!known) {
                    enrolled_.push_back(s->caps); enroll_last_ = Clock::now();
                    evq_.push_back(J().kv("type", "fed_enrolled").kv("client", s->id).kv("score", double(s->caps.score))
                                      .kv("mem_mb", uint64_t(s->caps.mem_budget_mb)).kv("shard_pref", int(s->caps.shard_pref)).kv("enrolled", uint64_t(enrolled_.size())).str());
                }
                why = "enrolling (" + std::to_string(enrolled_.size()) + " devices)"; break; }
            case Enroll::Preparing: retry = 300; why = "partitioning"; break;
            case Enroll::Failed: flags = 1; retry = 0; why = "partitioning failed: " + enroll_err_; break;
            case Enroll::Ready: {
                bool known = std::any_of(enrolled_.begin(), enrolled_.end(), [&](const EnrollInfo& e) { return e.id == s->id; });
                if (known) ready = true; else { flags = 1; retry = 0; why = "enrollment closed before this device joined"; }
                break; }
            }
        }
        flush();
        if (ready) {
            auto sh = cfg_.shards->shard_for(s->id);
            if (!sh) return send_shard_wait(fd, 0, 1, "no shard for this device");
            bool ok = send_shard(fd, *sh, chunk);
            if (ok) emit(J().kv("type", "fed_shard_sent").kv("client", s->id).kv("bytes", sh->bytes()).kv("nodes", uint64_t(sh->i("nodes", 0))).str());
            return ok;
        }
        return send_shard_wait(fd, retry, flags, why);
    }

    bool handle_sync(int fd, Dtype down, size_t chunk) {
        std::shared_ptr<const std::vector<float>> g; uint64_t gr;
        { std::lock_guard<std::mutex> l(m_); g = global_; gr = global_round_; }
        return send_result(fd, gr, 0, 0, down, chunk, g);
    }

    bool send_result(int fd, uint64_t round, uint32_t parts, uint8_t status, Dtype down, size_t chunk,
                     const std::shared_ptr<const std::vector<float>>& g) {
        Wr b; b.u64(round); b.u32(parts); b.u8(status); b.u8(uint8_t(down));
        if (!send_msg(fd, Msg::Result, b)) return false;
        if (status != 0) return true;
        const std::vector<float>& v = *g;
        return send_stream(fd, down, v.size(), chunk, [&v](uint64_t off, size_t n, float* out) {
            std::memcpy(out, v.data() + off, n * sizeof(float)); });
    }

    bool handle_ready(const std::shared_ptr<Session>& s, const std::vector<uint8_t>& body, Dtype up, Dtype down, size_t chunk) {
        Rd r(body); uint64_t n = r.u64(), base = r.u64(); r.u8();
        if (!r.done()) return false;
        const int fd = s->sock->fd;
        auto m = std::make_shared<Member>();
        m->id = s->id; m->sock = s->sock; m->weight = cfg_.weighting == CoordinatorConfig::Weighting::Uniform ? 1.0 : std::max<double>(1.0, double(n)); m->base = base;
        {
            std::unique_lock<std::mutex> l(m_);
            if (finished_) { l.unlock(); return send_skip(fd, 0, 0, false, true, "training finished"); }
            pending_.push_back(m);
            tick_locked(Clock::now());
            l.unlock(); flush(); l.lock();
            bool got = cv_.wait_for(l, Ms(cfg_.ready_timeout_ms), [&] { return m->dec != Dec::Waiting || stopping_.load(); });
            if (!got || m->dec == Dec::Waiting) {
                pending_.erase(std::remove(pending_.begin(), pending_.end(), m), pending_.end());
                if (m->dec == Dec::Waiting) { m->dec = Dec::Skip; m->skip_reason = "not enough clients"; m->retry_ms = 2000; }
            }
        }
        flush();
        if (stopping_) return false;
        if (m->dec == Dec::Skip) {
            uint64_t gr = round();
            return send_skip(fd, gr, m->retry_ms, m->needs_sync, false, m->skip_reason);
        }

        // ── GO: stream the update straight into the shared accumulator ─────────
        { Wr b; b.u64(m->ctx->id); b.u8(uint8_t(up)); if (!send_msg(fd, Msg::Go, b)) { drop(m, "send GO failed"); return false; } }
        std::string err;
        bool ok = recv_stream(fd, up, cfg_.param_count,
            [&](uint64_t off, const float* d, size_t k) {
                std::lock_guard<std::mutex> l(m_);
                if (!m->active || cur_ != m->ctx) return false;        // cut off by deadline
                const float w = static_cast<float>(m->weight_eff);
                for (size_t i = 0; i < k; i++) accum_[off + i] += w * d[i];
                m->applied = off + k;
                return true;
            }, err);
        {
            std::unique_lock<std::mutex> l(m_);
            if (ok && m->active) { m->done = true; m->active = false; }
            else if (m->active) {
                m->dropped = true; m->active = false;
                evq_.push_back(J().kv("type", "fed_client_dropped").kv("client", m->id).kv("round", m->ctx->id)
                                  .kv("reason", err).kv("applied", m->applied).str());
            }
            tick_locked(Clock::now());
        }
        flush();
        if (!ok || m->dropped) return false;

        std::shared_ptr<const std::vector<float>> g; bool aborted; uint32_t parts; uint64_t rid;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [&] { return m->ctx->finalized || stopping_.load(); });
            if (!m->ctx->finalized) return false;
            g = m->ctx->result; aborted = m->ctx->aborted; parts = m->ctx->n_participants; rid = m->ctx->id;
        }
        return send_result(fd, rid, parts, aborted ? 1 : 0, down, chunk, g);
    }

    void drop(const std::shared_ptr<Member>& m, const char* why) {
        { std::unique_lock<std::mutex> l(m_);
          if (m->active) { m->active = false; m->dropped = true;
              evq_.push_back(J().kv("type", "fed_client_dropped").kv("client", m->id).kv("round", m->ctx->id).kv("reason", why).kv("applied", m->applied).str()); }
          tick_locked(Clock::now()); }
        flush();
    }

    static bool ct_equal(const std::string& a, const std::string& b) {   // constant-time-ish compare
        unsigned char d = a.size() != b.size();
        for (size_t i = 0; i < std::min(a.size(), b.size()); i++) d |= (unsigned char)(a[i] ^ b[i]);
        return d == 0;
    }
};

}} // namespace tf::fed
