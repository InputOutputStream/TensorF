// fed_mobile_sim.cpp — loopback test-bench for the mobile federated layer.
// No model, no BLAS, no network: threads stand in for phones. Covers fp16 codec,
// weighted FedAvg, EXACT dropout handling, upload deadline, SYNC for late
// joiners, reconnect without losing the trained update, auth / model mismatch
// rejection, garbage input, "not enough clients" SKIP.
//
//   g++ -std=c++20 -O2 -pthread -Iinclude tests/fed_mobile_sim.cpp -o bin/fed_mobile_sim && ./bin/fed_mobile_sim
//   (add -fsanitize=thread  or  -fsanitize=address,undefined  for the sanitizer runs)
#include "net/Mobile/FedCoordinator.hpp"
#include "net/Mobile/MobileClient.hpp"

#include <cstdlib>
#include <random>

using namespace tf::fed;

static std::atomic<int> g_fail{0}, g_pass{0};
#define CHECK(c, ...) do { if (c) { g_pass++; } else { g_fail++; std::printf("  [FAIL] %s:%d  %s  ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

struct EventLog {
    std::mutex m; std::vector<std::string> v;
    void add(const std::string& s) { std::lock_guard<std::mutex> l(m); v.push_back(s); }
    int count(const std::string& needle) { std::lock_guard<std::mutex> l(m); int n = 0; for (auto& s : v) if (s.find(needle) != std::string::npos) n++; return n; }
};

static CoordinatorConfig base_cfg(uint64_t P, size_t minc, EventLog& ev) {
    CoordinatorConfig c; c.bind_addr = "127.0.0.1"; c.port = 0; c.param_count = P; c.model_fingerprint = 0xC0FFEE;
    c.min_clients = minc; c.max_clients = minc; c.grace_ms = 50; c.chunk_elems = 1024;
    c.ready_timeout_ms = 5000; c.upload_deadline_ms = 5000; c.io_timeout_ms = 3000; c.hello_timeout_ms = 1000;
    c.on_event = [&ev](const std::string& s) { ev.add(s); };
    return c;
}
static void wait_until(const std::function<bool()>& f, int ms) {
    for (int t = 0; t < ms && !f(); t += 10) std::this_thread::sleep_for(Ms(10));
}

// ── raw protocol client (for abnormal behaviours) ──────────────────────────
struct Raw {
    int fd = -1; Dtype up = Dtype::F32, down = Dtype::F32; std::string reject;
    bool connect_hello(uint16_t port, const std::string& id, uint64_t P, uint32_t fp, const std::string& tok = "", uint8_t mask = 1, const std::string& algo = "") {
        fd = ::socket(AF_INET, SOCK_STREAM, 0);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port); ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        if (::connect(fd, (sockaddr*)&a, sizeof a) < 0) return false;
        set_socket_timeouts(fd, 5000, 5000);
        Wr b; b.str(id); b.str(tok); b.str(algo); b.u64(P); b.u32(fp); b.u8(mask); b.u32(4); b.u32(0); b.u8(1); b.u8(0); b.u16(500); b.u32(0);
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!send_msg(fd, Msg::Hello, b) || !recv_msg(fd, t, body, err) || t != Msg::Welcome) return false;
        Rd r(body); uint8_t ok = r.u8(); reject = r.str();
        if (!ok) return false;
        r.u64(); up = Dtype(r.u8()); down = Dtype(r.u8());
        return true;
    }
    void close() { if (fd >= 0) { ::close(fd); fd = -1; } }
    // READY -> expects GO; returns false on SKIP
    bool ready(uint64_t n, uint64_t base, std::string* skip_reason = nullptr) {
        Wr b; b.u64(n); b.u64(base); b.u8(0); send_msg(fd, Msg::Ready, b);
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!recv_msg(fd, t, body, err)) { if (skip_reason) *skip_reason = "io:" + err; return false; }
        if (t == Msg::Skip) { Rd r(body); r.u64(); r.u32(); r.u8(); if (skip_reason) *skip_reason = r.str(); return false; }
        if (t != Msg::Go) return false;
        Rd g(body); g.u64(); up = Dtype(g.u8()); return true;
    }
    // send `elems` of constant `v` (of P total, advertised) in 1000-element chunks, then maybe stop
    bool upload(uint64_t P, float v, uint64_t stop_after = UINT64_MAX) {
        uint64_t sent = 0; bool ok = true;
        // hand-rolled so we can cut the stream mid-way
        uint8_t tb[8]; for (int k = 0; k < 8; k++) tb[k] = uint8_t(P >> (8 * k));
        ok = write_exact(fd, tb, 8);
        std::vector<float> chunk(1000, v);
        while (ok && sent < P && sent < stop_after) {
            uint32_t n = uint32_t(std::min<uint64_t>(1000, P - sent));
            uint8_t lb[4]; for (int k = 0; k < 4; k++) lb[k] = uint8_t(n >> (8 * k));
            ok = write_exact(fd, lb, 4) && write_exact(fd, chunk.data(), n * 4);
            sent += n;
        }
        return ok;
    }
    bool result(uint64_t P, std::vector<float>& out, uint8_t* status = nullptr) {
        Msg t; std::vector<uint8_t> body; std::string err;
        if (!recv_msg(fd, t, body, err) || t != Msg::Result) return false;
        Rd r(body); r.u64(); r.u32(); uint8_t st = r.u8(); Dtype d = Dtype(r.u8()); if (status) *status = st;
        if (st != 0) return true;
        out.assign(P, 0.f);
        return recv_stream(fd, d, P, [&](uint64_t o, const float* x, size_t n) { std::memcpy(&out[o], x, n * 4); return true; }, err);
    }
};

// ── tests ------------------------------------------------------------------
static void test_fp16() {
    std::printf("[fp16 codec]\n");
    int bad = 0;
    for (uint32_t h = 0; h < 65536; h++) {
        uint16_t hh = uint16_t(h); float f = half_to_float(hh);
        if (std::isnan(f)) { if (!std::isnan(half_to_float(float_to_half(f)))) bad++; continue; }
        if (float_to_half(f) != hh) bad++;
    }
    CHECK(bad == 0, "%d half values do not round-trip", bad);
#if defined(__FLT16_MAX__)
    std::mt19937 g(1); std::uniform_real_distribution<float> d(-70000, 70000); int diff = 0;
    for (int i = 0; i < 2000000; i++) {
        float f = (i & 3) == 0 ? d(g) : d(g) * std::pow(2.f, float(int(g() % 40) - 30));
        _Float16 ref = static_cast<_Float16>(f); uint16_t rb; std::memcpy(&rb, &ref, 2);
        uint16_t mine = float_to_half(f);
        if (mine != rb && !(std::isnan(f))) diff++;
    }
    CHECK(diff == 0, "%d of 2M random floats differ from the compiler's _Float16", diff);
#endif
}

static void test_weighted_avg(bool fp16) {
    std::printf("[weighted FedAvg, 3 clients, %s upload, small chunks]\n", fp16 ? "fp16" : "fp32");
    EventLog ev; const uint64_t P = 50000;
    auto cc = base_cfg(P, 3, ev); cc.allow_fp16_upload = fp16;
    FedCoordinator srv(cc); CHECK(srv.start(), "start");
    std::vector<std::vector<float>> params(3, std::vector<float>(P, 0.f));
    std::vector<std::unique_ptr<MobileClient>> cl; std::vector<std::thread> th;
    for (int k = 0; k < 3; k++) {
        ClientConfig c; c.port = srv.port(); c.client_id = "phone-" + std::to_string(k); c.param_count = P;
        c.model_fingerprint = 0xC0FFEE; c.chunk_kb = 4; c.max_rounds = 2; c.io_timeout_ms = 3000;
        ClientHooks h;
        h.train = [&params, k](uint64_t) { for (auto& x : params[k]) x += float(k + 1); return uint64_t(k + 1); };
        h.get_params = [&params, k](uint64_t o, size_t n, float* out) { std::memcpy(out, &params[k][o], n * 4); };
        h.set_params = [&params, k](uint64_t o, const float* d, size_t n) { std::memcpy(&params[k][o], d, n * 4); };
        cl.push_back(std::make_unique<MobileClient>(c, h));
    }
    for (auto& c : cl) th.emplace_back([&c] { c->run(); });
    for (auto& t : th) t.join();
    auto g = srv.global();
    const float exp = 2.f * 14.f / 6.f;      // each round adds (1*1+2*2+3*3)/(1+2+3)
    float maxerr = 0; for (float x : *g) maxerr = std::max(maxerr, std::fabs(x - exp));
    CHECK(srv.round() == 2, "round=%llu", (unsigned long long)srv.round());
    CHECK(maxerr < (fp16 ? 5e-3f : 1e-5f), "max error %g", maxerr);
    float lerr = 0; for (int k = 0; k < 3; k++) for (float x : params[k]) lerr = std::max(lerr, std::fabs(x - exp));
    CHECK(lerr < (fp16 ? 5e-3f : 1e-5f), "clients did not receive the global model (err %g)", lerr);
    CHECK(ev.count("fed_round_done") == 2, "events");
    uint64_t up = 0; for (auto& c : cl) up += c->stats().bytes_up;
    std::printf("    (process-wide counters, include the server side) %.2f MB total for 2 rounds x 3 clients x %.0f KB model\n", up / 1e6, P * 4 / 1e3);
}

static void test_exact_dropout() {
    std::printf("[client drops mid-stream: exact per-element divisor]\n");
    EventLog ev; const uint64_t P = 10000;
    FedCoordinator srv(base_cfg(P, 3, ev)); srv.start();
    std::vector<float> ra, rb; uint8_t sa = 9, sb = 9;
    std::thread A([&] { Raw r; r.connect_hello(srv.port(), "A", P, 0xC0FFEE); if (r.ready(1, 0)) { r.upload(P, 2.f); r.result(P, ra, &sa); } r.close(); });
    std::thread B([&] { Raw r; r.connect_hello(srv.port(), "B", P, 0xC0FFEE); if (r.ready(3, 0)) { r.upload(P, 4.f); r.result(P, rb, &sb); } r.close(); });
    std::thread C([&] { Raw r; r.connect_hello(srv.port(), "C", P, 0xC0FFEE); if (r.ready(2, 0)) { r.upload(P, 10.f, 4000); } r.close(); });
    A.join(); B.join(); C.join();
    wait_until([&] { return srv.round() >= 1; }, 3000);
    auto g = srv.global();
    CHECK(srv.round() == 1, "round %llu", (unsigned long long)srv.round());
    double e1 = (1 * 2 + 3 * 4 + 2 * 10) / 6.0, e2 = (1 * 2 + 3 * 4) / 4.0; float m1 = 0, m2 = 0;
    for (uint64_t i = 0; i < P; i++) { float e = i < 4000 ? float(e1) : float(e2); (i < 4000 ? m1 : m2) = std::max(i < 4000 ? m1 : m2, std::fabs((*g)[i] - e)); }
    CHECK(m1 < 1e-5f && m2 < 1e-5f, "prefix err %g, tail err %g (expected %.4f / %.4f)", m1, m2, e1, e2);
    CHECK(sa == 0 && sb == 0 && ra.size() == P && std::fabs(ra[0] - float(e1)) < 1e-5f, "A/B did not get the result");
    CHECK(ev.count("fed_client_dropped") == 1, "dropped events=%d", ev.count("fed_client_dropped"));
}

static void test_deadline() {
    std::printf("[stalled client is cut off at the upload deadline]\n");
    EventLog ev; const uint64_t P = 5000;
    auto cc = base_cfg(P, 3, ev); cc.upload_deadline_ms = 600;
    FedCoordinator srv(cc); srv.start();
    auto t0 = Clock::now(); std::vector<float> ra; uint8_t sa = 9;
    std::thread A([&] { Raw r; r.connect_hello(srv.port(), "A", P, 0xC0FFEE); if (r.ready(1, 0)) { r.upload(P, 2.f); r.result(P, ra, &sa); } r.close(); });
    std::thread B([&] { Raw r; r.connect_hello(srv.port(), "B", P, 0xC0FFEE); if (r.ready(1, 0)) { r.upload(P, 6.f); std::vector<float> o; r.result(P, o, &sa); } r.close(); });
    std::thread S([&] { Raw r; r.connect_hello(srv.port(), "S", P, 0xC0FFEE); if (r.ready(1, 0)) std::this_thread::sleep_for(Ms(2500)); r.close(); });
    A.join(); B.join();
    double el = std::chrono::duration<double>(Clock::now() - t0).count();
    CHECK(srv.round() == 1, "round %llu", (unsigned long long)srv.round());
    CHECK(el < 2.0, "round took %.2fs (deadline was 0.6s)", el);
    CHECK(ra.size() == P && std::fabs(ra[0] - 4.f) < 1e-5f, "average of the 2 finishers should be 4, got %g", ra.empty() ? -1.f : ra[0]);
    S.join();
}

static void test_reconnect_and_sync() {
    std::printf("[kick between rounds: reconnect keeps training; late joiner syncs]\n");
    EventLog ev; const uint64_t P = 3000;
    auto cc = base_cfg(P, 1, ev);
    FedCoordinator srv(cc); srv.start();
    std::vector<float> p(P, 0.f);
    ClientConfig c; c.port = srv.port(); c.client_id = "M"; c.param_count = P; c.model_fingerprint = 0xC0FFEE;
    c.chunk_kb = 1; c.backoff_min_ms = 50; c.io_timeout_ms = 3000;
    ClientHooks h;
    h.train = [&](uint64_t) { for (auto& x : p) x += 1.f; return uint64_t(1); };
    h.get_params = [&](uint64_t o, size_t n, float* out) { std::memcpy(out, &p[o], n * 4); };
    h.set_params = [&](uint64_t o, const float* d, size_t n) { std::memcpy(&p[o], d, n * 4); };
    MobileClient m(c, h);
    std::thread t([&] { m.run(); });
    wait_until([&] { return srv.round() >= 1; }, 3000);
    srv.kick("M"); std::this_thread::sleep_for(Ms(30)); srv.kick("M");
    wait_until([&] { return srv.round() >= 4; }, 6000);
    CHECK(srv.round() >= 4, "round %llu", (unsigned long long)srv.round());
    CHECK(m.stats().connects >= 2, "connects=%llu", (unsigned long long)m.stats().connects.load());

    // late joiner
    std::vector<float> q(P, 123.f);
    ClientConfig c2 = c; c2.client_id = "late"; c2.max_rounds = 0;
    ClientHooks h2; h2.train = [](uint64_t) { return uint64_t(0); };      // never trains: only syncs
    h2.get_params = [&](uint64_t o, size_t n, float* out) { std::memcpy(out, &q[o], n * 4); };
    h2.set_params = [&](uint64_t o, const float* d, size_t n) { std::memcpy(&q[o], d, n * 4); };
    MobileClient late(c2, h2); std::thread t2([&] { late.run(); });
    wait_until([&] { return late.stats().syncs >= 1; }, 3000);
    late.stop(); t2.join();
    m.stop(); t.join();
    auto g = srv.global();
    CHECK(late.stats().syncs >= 1 && late.base_round() >= 1, "late joiner did not sync");
    // q was synced to *some* global snapshot (round may have advanced in between): must be a whole number of rounds
    float v = q[0]; bool uniform = true; for (float x : q) if (x != v) uniform = false;
    CHECK(uniform && v == std::floor(v) && v >= 1.f && v <= float(srv.round()), "late joiner params not a clean snapshot: %g", v);
    CHECK((*g)[0] == float(srv.round()), "each completed round must add exactly +1 (global=%g rounds=%llu): no update lost or doubled",
          (*g)[0], (unsigned long long)srv.round());
}

static void test_rejections() {
    std::printf("[auth / model mismatch / garbage / not-enough-clients]\n");
    EventLog ev; const uint64_t P = 100;
    auto cc = base_cfg(P, 2, ev); cc.auth_token = "s3cret"; cc.ready_timeout_ms = 300;
    FedCoordinator srv(cc); srv.start();
    { Raw r; bool ok = r.connect_hello(srv.port(), "x", P, 0xC0FFEE, "wrong"); CHECK(!ok && r.reject.find("authentication") != std::string::npos, "auth: '%s'", r.reject.c_str()); r.close(); }
    { Raw r; bool ok = r.connect_hello(srv.port(), "x", P + 1, 0xC0FFEE, "s3cret"); CHECK(!ok && r.reject.find("parameter count") != std::string::npos, "pc: '%s'", r.reject.c_str()); r.close(); }
    { Raw r; bool ok = r.connect_hello(srv.port(), "x", P, 0xBAD, "s3cret"); CHECK(!ok && r.reject.find("fingerprint") != std::string::npos, "fp: '%s'", r.reject.c_str()); r.close(); }
    { Raw r; bool ok = r.connect_hello(srv.port(), "bad id", P, 0xC0FFEE, "s3cret"); CHECK(!ok, "id with space accepted"); r.close(); }
    {   // garbage: an HTTP request, then oversize body length
        int fd = ::socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(srv.port()); ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        ::connect(fd, (sockaddr*)&a, sizeof a); set_socket_timeouts(fd, 2000, 2000);
        const char* junk = "GET / HTTP/1.1\r\nHost: x\r\n\r\n"; write_exact(fd, junk, std::strlen(junk));
        char c; ssize_t n = ::read(fd, &c, 1); CHECK(n <= 0, "server answered garbage"); ::close(fd);
        Raw r2; r2.fd = ::socket(AF_INET, SOCK_STREAM, 0); ::connect(r2.fd, (sockaddr*)&a, sizeof a); set_socket_timeouts(r2.fd, 2000, 2000);
        uint8_t hdr[8] = {'T','F','F','D', VERSION, 1, 0xFF, 0xFF}; write_exact(r2.fd, hdr, 8);
        n = ::read(r2.fd, &c, 1); CHECK(n <= 0, "oversize body accepted"); r2.close();
    }
    {   Raw r; CHECK(r.connect_hello(srv.port(), "solo", P, 0xC0FFEE, "s3cret"), "valid hello");
        auto t0 = Clock::now(); std::string why; bool go = r.ready(1, 0, &why);
        double el = std::chrono::duration<double>(Clock::now() - t0).count();
        CHECK(!go && why == "not enough clients" && el < 2.0, "why='%s' after %.2fs", why.c_str(), el); r.close(); }
    {   // server still healthy after all that
        Raw r; CHECK(r.connect_hello(srv.port(), "after", P, 0xC0FFEE, "s3cret"), "server unhealthy after abuse"); r.close(); }
}

static void test_stale_skip() {
    std::printf("[stale client is told to sync instead of joining]\n");
    EventLog ev; const uint64_t P = 200;
    auto cc = base_cfg(P, 1, ev); cc.max_staleness = 0;
    FedCoordinator srv(cc); srv.start();
    for (int i = 0; i < 2; i++) { Raw r; r.connect_hello(srv.port(), "fresh", P, 0xC0FFEE); std::vector<float> o; if (r.ready(1, uint64_t(i))) { r.upload(P, 1.f); r.result(P, o); } r.close(); }
    CHECK(srv.round() == 2, "round %llu", (unsigned long long)srv.round());
    Raw r; r.connect_hello(srv.port(), "old", P, 0xC0FFEE); std::string why;
    CHECK(!r.ready(1, 0, &why) && why.find("stale") != std::string::npos, "why='%s'", why.c_str()); r.close();
}

// ── shards: enrollment, capacity-proportional split, late joiner, provider failure, wire roundtrip ──
struct FakeProvider : ShardProvider {
    bool fail = false; std::mutex m; std::map<std::string, std::shared_ptr<const Shard>> out; std::vector<EnrollInfo> seen;
    void prepare(const std::vector<EnrollInfo>& d) override {
        if (fail) throw std::runtime_error("boom");
        double tot = 0; for (auto& e : d) tot += e.score;
        const size_t N = 1000; size_t at = 0; std::lock_guard<std::mutex> l(m); seen = d;
        for (size_t i = 0; i < d.size(); i++) {
            size_t n = i + 1 == d.size() ? N - at : size_t(N * d[i].score / tot);
            auto sh = std::make_shared<Shard>(); sh->set("nodes", (long long)n);
            Blob v; v.name = "v"; for (size_t k = 0; k < n; k++) v.f.push_back(float(at + k)); Blob u; u.name = "u"; u.kind = Blob::Kind::U32; for (size_t k = 0; k < n; k++) u.u.push_back(uint32_t(at + k));
            sh->blobs = {v, u}; out[d[i].id] = sh; at += n;
        }
    }
    std::shared_ptr<const Shard> shard_for(const std::string& id) override { std::lock_guard<std::mutex> l(m); auto it = out.find(id); return it == out.end() ? nullptr : it->second; }
};

static void test_shards() {
    std::printf("[shards: enrollment, capacity-proportional split, late joiner, failure, wire]\n");
    EventLog ev; auto prov = std::make_shared<FakeProvider>();
    auto cc = base_cfg(10, 3, ev); cc.shards = prov; cc.enroll_grace_ms = 100;
    FedCoordinator srv(cc); srv.start();
    const float scores[3] = {90.f, 60.f, 30.f}; std::vector<std::unique_ptr<MobileClient>> cl; std::vector<std::thread> th;
    std::atomic<size_t> got[3] = {{0}, {0}, {0}}; std::atomic<int> okdata[3] = {{1}, {1}, {1}}, have[3] = {{0}, {0}, {0}};
    for (int k = 0; k < 3; k++) {
        ClientConfig c; c.port = srv.port(); c.client_id = "p" + std::to_string(k); c.param_count = 10; c.model_fingerprint = 0xC0FFEE; c.score = scores[k]; c.mem_budget_mb = 64; c.backoff_min_ms = 50;
        ClientHooks h; h.train = [](uint64_t) { return uint64_t(0); };
        h.get_params = [](uint64_t, size_t n, float* o) { std::memset(o, 0, n * 4); }; h.set_params = [](uint64_t, const float*, size_t) {};
        h.wants_shard = [&have, k] { return have[k].load() == 0; };
        h.on_shard = [&, k](Shard&& sh) { const Blob* v = sh.blob("v"); const Blob* u = sh.blob("u"); got[k] = v ? v->f.size() : 0;
            if (!v || !u || v->f.size() != u->u.size() || sh.i("nodes") != (long long)v->f.size()) okdata[k] = 0;
            for (size_t i = 0; v && i < v->f.size(); i++) if (v->f[i] != float(u->u[i])) okdata[k] = 0;
            have[k] = 1; };
        cl.push_back(std::make_unique<MobileClient>(c, h));
    }
    for (auto& c : cl) th.emplace_back([&c] { c->run(); });
    wait_until([&] { return have[0] && have[1] && have[2]; }, 5000);
    CHECK(have[0] && have[1] && have[2], "not every device received its shard");
    CHECK(got[0] + got[1] + got[2] == 1000, "shards must cover everything: %zu+%zu+%zu", got[0].load(), got[1].load(), got[2].load());
    CHECK(std::abs(int(got[0].load()) - 500) <= 2 && std::abs(int(got[1].load()) - 333) <= 2 && std::abs(int(got[2].load()) - 167) <= 3, "split not proportional to scores: %zu %zu %zu", got[0].load(), got[1].load(), got[2].load());
    CHECK(okdata[0] && okdata[1] && okdata[2], "shard payload corrupted");
    { std::lock_guard<std::mutex> l(prov->m); CHECK(prov->seen.size() == 3 && prov->seen[0].id == "p0" && prov->seen[0].score == 90.f && prov->seen[2].mem_budget_mb == 64, "provider got wrong capabilities"); }
    // late joiner: enrollment is closed
    ClientConfig lc; lc.port = srv.port(); lc.client_id = "late"; lc.param_count = 10; lc.model_fingerprint = 0xC0FFEE;
    ClientHooks lh; lh.wants_shard = [] { return true; }; lh.on_shard = [](Shard&&) {}; lh.train = [](uint64_t) { return uint64_t(0); };
    lh.get_params = [](uint64_t, size_t n, float* o) { std::memset(o, 0, n * 4); }; lh.set_params = [](uint64_t, const float*, size_t) {};
    MobileClient late(lc, lh); late.run();            // returns by itself because it is refused
    CHECK(late.failed(), "late joiner should be refused");
    for (auto& c : cl) c->stop(); for (auto& t : th) t.join();
    // provider failure is reported to devices
    EventLog ev2; auto bad = std::make_shared<FakeProvider>(); bad->fail = true;
    auto c2 = base_cfg(10, 1, ev2); c2.shards = bad; c2.enroll_grace_ms = 30; FedCoordinator s2(c2); s2.start();
    ClientConfig fc; fc.port = s2.port(); fc.client_id = "x"; fc.param_count = 10; fc.model_fingerprint = 0xC0FFEE;
    MobileClient f(fc, lh); f.run();
    CHECK(f.failed() && ev2.count("fed_shards_failed") == 1, "failure not propagated");
    // wire: malformed shard header rejected
    int sv[2]; ::socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    Wr b; b.u8(0); b.u8(200); std::vector<uint8_t> body = b.b; Shard o; std::string err;
    CHECK(!recv_shard(sv[0], body, o, err), "too many blobs must be rejected"); ::close(sv[0]); ::close(sv[1]);
    // wire: fp16 blob roundtrip over a socketpair
    ::socketpair(AF_UNIX, SOCK_STREAM, 0, sv);
    Shard in; in.set("k", "v"); Blob f16; f16.name = "x"; f16.kind = Blob::Kind::F16; for (int i = 0; i < 5000; i++) f16.f.push_back(float(i) * 0.37f - 900.f); in.blobs = {f16};
    std::thread w([&] { send_shard(sv[1], in, 777); });
    Msg t; std::vector<uint8_t> hb; Shard rcv; bool okrecv = recv_msg(sv[0], t, hb, err) && t == Msg::ShardHdr && recv_shard(sv[0], hb, rcv, err); w.join();
    float maxrel = 0; if (okrecv) for (int i = 0; i < 5000; i++) maxrel = std::max(maxrel, std::fabs(rcv.blobs[0].f[i] - in.blobs[0].f[i]) / (1.f + std::fabs(in.blobs[0].f[i])));
    CHECK(okrecv && rcv.str("k") == "v" && maxrel < 1e-3f, "fp16 shard roundtrip (%s, rel err %g)", err.c_str(), maxrel);
    ::close(sv[0]); ::close(sv[1]);
}

int main() {
    test_fp16();
    test_weighted_avg(false);
    test_weighted_avg(true);
    test_exact_dropout();
    test_deadline();
    test_reconnect_and_sync();
    test_rejections();
    test_stale_skip();
    test_shards();
    std::printf("\n%d checks passed, %d failed\n", g_pass.load(), g_fail.load());
    return g_fail ? 1 : 0;
}
