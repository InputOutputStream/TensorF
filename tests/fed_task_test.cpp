// fed_task_test — checks the algorithm-agnostic layer: strategies, tensor<->flat bridge,
// algorithm/fingerprint handshake, registry consistency server<->client, graph partitioning.
// Graph checks need Datasets/cora (./scripts/download_cora.sh); they are skipped otherwise.
#include "net/Mobile/BuiltinTasks.hpp"
#include "net/Mobile/FedStrategies.hpp"
#include "net/Mobile/graph/WeightedPartitioner.hpp"
#include "net/Mobile/graph/FedGraphData.hpp"
#include <cstdlib>
#include <cmath>
#include <fstream>
using namespace tf::fed;
static int g_fail = 0, g_pass = 0;
#define CHECK(c, ...) do { if (c) g_pass++; else { g_fail++; std::printf("  [FAIL] %s:%d  %s  ", __FILE__, __LINE__, #c); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)
static bool near(double a, double b, double eps = 1e-5) { return std::fabs(a - b) <= eps * (1 + std::fabs(b)); }

static void test_strategies() {
    std::printf("[strategies]\n");
    std::vector<float> old{1.f, 2.f}, avg{2.f, 1.f};
    auto m = fedavgm(0.5, 1.0); auto a1 = avg; m(1, old, a1);            // v=d=(1,-1) -> (2,1)
    CHECK(near(a1[0], 2) && near(a1[1], 1), "fedavgm r1 %g %g", a1[0], a1[1]);
    std::vector<float> old2 = a1, avg2{2.f, 1.f}; m(2, old2, avg2);      // d=0, v=0.5*(1,-1) -> (2.5,0.5)
    CHECK(near(avg2[0], 2.5) && near(avg2[1], 0.5), "fedavgm r2 %g %g", avg2[0], avg2[1]);
    auto ad = fedadam(0.1, 0.9, 0.99, 1e-3); std::vector<float> o{0.f}, v{1.f}; ad(1, o, v);
    double mm = 0.1, vv = 0.01, exp = 0.1 * mm / (std::sqrt(vv) + 1e-3);
    CHECK(near(v[0], exp), "fedadam %g vs %g", v[0], exp);
    CHECK(!make_strategy("fedavg") && make_strategy("fedavgm:0.9") && make_strategy("fedadam:0.01"), "make_strategy");
    bool threw = false; try { make_strategy("nope"); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw, "bad strategy must throw");
    float w[2] = {4.f, 0.f}, anchor[2] = {0.f, 0.f}; fedprox_pull(w, anchor, 2, 0.25f); CHECK(near(w[0], 3.f), "fedprox %g", w[0]);
}

static void test_param_view() {
    std::printf("[ParamView: tensors <-> flat vector]\n");
    auto a = make_tensor<float>(Matrix<float>(std::vector<float>{1, 2, 3, 4, 5, 6}, {2, 3}));
    auto b = make_tensor<float>(Matrix<float>(std::vector<float>{7, 8, 9, 10}, {4, 1}));
    ParamView<float> pv({a, b});
    CHECK(pv.total() == 10, "total %llu", (unsigned long long)pv.total());
    float out[5]; pv.get(4, 5, out);                                      // spans the tensor boundary
    CHECK(out[0] == 5 && out[1] == 6 && out[2] == 7 && out[4] == 9, "boundary read");
    float in[3] = {-1, -2, -3}; pv.set(5, in, 3);
    CHECK(a->val.data[5] == -1 && b->val.data[0] == -2 && b->val.data[1] == -3, "boundary write");
    auto f = pv.flatten(); CHECK(f.size() == 10 && f[9] == 10, "flatten");
    bool threw = false; try { pv.get(8, 5, out); } catch (const std::out_of_range&) { threw = true; } CHECK(threw, "range check");
    auto c = make_tensor<float>(Matrix<float>(std::vector<float>{1, 2, 3, 4, 5, 6}, {3, 2}));   // same count, other shape
    ParamView<float> pv2({c, b});
    CHECK(pv.fingerprint("gcn") != pv2.fingerprint("gcn"), "shape change must change fingerprint");
    CHECK(pv.fingerprint("gcn") != pv.fingerprint("sage"), "algo name must change fingerprint");
    CHECK(pv.fingerprint("gcn") == pv.fingerprint("gcn"), "stable");
}

static void test_handshake_and_hooks() {
    std::printf("[algo handshake, uniform weighting, post_aggregate hook]\n");
    CoordinatorConfig c; c.bind_addr = "127.0.0.1"; c.port = 0; c.param_count = 4; c.algo = "gcn"; c.model_fingerprint = 7;
    c.min_clients = 2; c.max_clients = 2; c.grace_ms = 20; c.weighting = CoordinatorConfig::Weighting::Uniform;
    c.post_aggregate = [](uint64_t, const std::vector<float>&, std::vector<float>& a) { for (auto& x : a) x *= 2; };
    c.on_event = [](const std::string&) {};
    FedCoordinator srv(c); srv.start();
    auto hello = [&](const std::string& id, const std::string& algo, uint32_t fp, std::string& why) {
        int fd = ::socket(AF_INET, SOCK_STREAM, 0); sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(srv.port()); ::inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
        ::connect(fd, (sockaddr*)&a, sizeof a); set_socket_timeouts(fd, 3000, 3000);
        Wr b; b.str(id); b.str(""); b.str(algo); b.u64(4); b.u32(fp); b.u8(1); b.u32(4); b.u32(0); b.u8(1); b.u8(0); b.u16(500); b.u32(0);
        Msg t; std::vector<uint8_t> body; std::string err;
        send_msg(fd, Msg::Hello, b); recv_msg(fd, t, body, err); Rd r(body); uint8_t ok = r.u8(); why = r.str();
        if (!ok) { ::close(fd); return -1; } return fd; };
    std::string why; int bad = hello("x", "sage", 7, why);
    CHECK(bad < 0 && why.find("algorithm mismatch") != std::string::npos && why.find("sage") != std::string::npos, "why='%s'", why.c_str());
    int f1 = -1, f2 = -1; std::vector<float> res1;
    std::thread t1([&] { f1 = hello("A", "gcn", 7, why); Wr b; b.u64(1); b.u64(0); b.u8(0); send_msg(f1, Msg::Ready, b);
        Msg t; std::vector<uint8_t> body; std::string err; recv_msg(f1, t, body, err);
        std::vector<float> d(4, 2.f); send_stream(f1, Dtype::F32, 4, 4, [&](uint64_t o, size_t n, float* out) { std::memcpy(out, &d[o], n * 4); });
        recv_msg(f1, t, body, err); recv_stream(f1, Dtype::F32, 4, [&](uint64_t o, const float* x, size_t n) { res1.insert(res1.end(), x, x + n); return true; }, err); });
    std::thread t2([&] { f2 = hello("B", "gcn", 7, why); Wr b; b.u64(3); b.u64(0); b.u8(0); send_msg(f2, Msg::Ready, b);   // n=3, but Uniform ignores it
        Msg t; std::vector<uint8_t> body; std::string err; recv_msg(f2, t, body, err);
        std::vector<float> d(4, 4.f); send_stream(f2, Dtype::F32, 4, 4, [&](uint64_t o, size_t n, float* out) { std::memcpy(out, &d[o], n * 4); });
        recv_msg(f2, t, body, err); recv_stream(f2, Dtype::F32, 4, [&](uint64_t, const float*, size_t) { return true; }, err); });
    t1.join(); t2.join();
    CHECK(res1.size() == 4 && near(res1[0], 6.0), "uniform mean (2+4)/2=3, hook x2 -> 6, got %g", res1.empty() ? -1.0 : res1[0]);
    if (f1 >= 0) ::close(f1); if (f2 >= 0) ::close(f2);
}

static bool have_cora() { return std::ifstream("Datasets/cora/cora.content").good() && std::ifstream("Datasets/cora/cora.cites").good(); }
static void test_graph() {
    std::printf("[registry consistency + graph partitioning]\n");
    register_builtin_tasks();
    bool threw = false; try { TaskRegistry::get().find("nope"); } catch (const std::runtime_error& e) { threw = std::string(e.what()).find("gcn") != std::string::npos; }
    CHECK(threw, "unknown algo error lists registered ones");
    if (!have_cora()) { std::printf("  (Datasets/cora missing: graph checks skipped)\n"); return; }
    TaskOptions o; o.parse("data=Datasets/cora"); o.parse("parts=3"); o.parse("part=1"); o.parse("verbose=0");
    auto& fac = TaskRegistry::get().find("gcn");
    ServerTaskInfo si = fac.server(o); auto ct = fac.client(o); TaskSpec cs = ct->spec();
    CHECK(si.spec.algo == cs.algo && si.spec.param_count == cs.param_count && si.spec.fingerprint == cs.fingerprint,
          "server/client specs disagree: %llu/%llu", (unsigned long long)si.spec.param_count, (unsigned long long)cs.param_count);
    CHECK(si.initial.size() == si.spec.param_count, "initial size");
    std::vector<float> chunk(1000); ct->get_params(0, 1000, chunk.data()); ct->set_params(0, si.initial.data(), 1000); ct->get_params(0, 1000, chunk.data());
    CHECK(chunk[17] == si.initial[17], "set/get roundtrip through the task");
    TaskOptions o2 = o; o2.kv["hidden"] = "32";
    CHECK(fac.server(o2).spec.fingerprint != si.spec.fingerprint, "hidden size must change the fingerprint");

    using namespace tf::fed::graph;
    TaskOptions d; d.parse("data=Datasets/cora"); std::string cp, ep; linqs_paths(d, cp, ep);
    auto ds = load_linqs<float>(cp, ep, true); const auto& g = ds.graph; const auto& adj = g.adjacency();
    auto bfs = partition_nodes(adj, 4, PartitionMode::Bfs, 42), rnd = partition_nodes(adj, 4, PartitionMode::Random, 42);
    std::vector<size_t> sz(4, 0); for (int p : bfs) { CHECK(p >= 0 && p < 4, "unassigned node"); if (p >= 0 && p < 4) sz[p]++; }
    CHECK(*std::max_element(sz.begin(), sz.end()) - *std::min_element(sz.begin(), sz.end()) <= 2, "bfs unbalanced %zu..%zu", *std::min_element(sz.begin(), sz.end()), *std::max_element(sz.begin(), sz.end()));
    double cb = cut_fraction(adj, bfs), cr = cut_fraction(adj, rnd);
    std::printf("    edge cut: bfs %.1f%%, random %.1f%%\n", 100 * cb, 100 * cr);
    CHECK(cb < cr * 0.5, "bfs should cut far fewer edges than random");
    size_t tot = 0; for (int p = 0; p < 4; p++) { auto lg = induced_subgraph(g, bfs, p); tot += lg.g.num_nodes(); CHECK(lg.g.num_nodes() == lg.global_id.size(), "ids"); }
    CHECK(tot == g.num_nodes(), "partitions must cover every node exactly once");
}

static void test_capacity_and_partitioners() {
    std::printf("[capacity policy, caps, partitioners]\n");
    using namespace tf::fed::graph;
    std::vector<DeviceCaps> d(4); float sc[4] = {95, 80, 40, 15}; for (int i = 0; i < 4; i++) { d[i].id = "d" + std::to_string(i); d[i].score = sc[i]; }
    CapacityPolicy pol; auto f = target_fractions(d, pol);
    double sum = 0; for (double x : f) sum += x;
    CHECK(near(sum, 1.0), "fractions sum %g", sum);
    CHECK(near(f[0], f[1]), "scores >= hi must saturate to the same share (%g vs %g)", f[0], f[1]);
    CHECK(near(f[2] / f[3], 40.0 / 15.0), "below hi the share is proportional to the score");
    pol.hi = 100; auto g = target_fractions(d, pol); CHECK(near(g[0] / g[1], 95.0 / 80.0), "hi=100 must be plain proportional");
    d[3].score = 0; pol.hi = 70; auto h = target_fractions(d, pol); CHECK(h[3] > 0.02, "floor keeps a weak device alive (%g)", h[3]);
    auto caps = node_caps({DeviceCaps{"a", 50, 1, 0}, DeviceCaps{"b", 50, 0, 0}}, 1000.0, 0.5, 1.0);
    CHECK(caps[0] == 524 && caps[1] == UINT64_MAX, "node caps %llu", (unsigned long long)caps[0]);
    if (!have_cora()) { std::printf("  (Datasets/cora missing: partitioner checks skipped)\n"); return; }
    TaskOptions dd; dd.parse("data=Datasets/cora"); std::string cp, ep; linqs_paths(dd, cp, ep);
    auto ds = load_linqs<float>(cp, ep, true); const auto& adj = ds.graph.adjacency(); const size_t N = ds.graph.num_nodes();
    std::vector<DeviceCaps> dv(4); for (int i = 0; i < 4; i++) { dv[i].id = "d" + std::to_string(i); dv[i].score = sc[i]; }
    pol = CapacityPolicy(); auto fr = target_fractions(dv, pol); auto vw = degree_weights(adj);
    auto pb = partition_builtin(adj, vw, fr, 0.03, 42); auto sb = evaluate(adj, pb, vw, fr);
    auto rnd = partition_nodes(adj, 4, PartitionMode::Random, 42);
    size_t tot = 0; for (auto n : sb.nodes) tot += n; CHECK(tot == N, "builtin must assign every node");
    CHECK(sb.max_imbalance < 1.10, "builtin imbalance %.3f", sb.max_imbalance);
    CHECK(sb.cut_fraction < cut_fraction(adj, rnd) * 0.4, "builtin cut %.3f not far below random", sb.cut_fraction);
    std::printf("    builtin: cut %.1f%%, imbalance %.2f\n", 100 * sb.cut_fraction, sb.max_imbalance);
    // every compiled-in backend: full coverage, targets respected from above AND below, far better than random
    size_t nback = 0; double best_cut = 1; Backend best = Backend::Builtin; std::vector<int> pm;
    for (Backend be : available_backends()) {
        auto pp = partition_with(be, adj, vw, fr, 0.03, 42); auto sp = evaluate(adj, pp, vw, fr); nback++;
        size_t tt = 0; for (auto n : sp.nodes) tt += n; double under = 1e9; for (size_t k = 0; k < 4; k++) under = std::min(under, sp.weight[k] / sp.target[k]);
        CHECK(tt == N, "%s: not every node assigned", backend_name(be));
        CHECK(sp.max_imbalance < 1.10 && under > 0.85, "%s: targets missed (max %.3f, min %.3f)", backend_name(be), sp.max_imbalance, under);
        CHECK(sp.cut_fraction < cut_fraction(adj, rnd) * 0.4, "%s: cut %.3f not far below random", backend_name(be), sp.cut_fraction);
        std::printf("    %-9s cut %5.1f%%  imbalance %.2f..%.2f\n", backend_name(be), 100 * sp.cut_fraction, under, sp.max_imbalance);
        if (be != Backend::Builtin && sp.cut_fraction < best_cut) { best_cut = sp.cut_fraction; best = be; }
        if (be == Backend::Metis) pm = pp;
    }
    CHECK(nback >= 1, "no backend");
    if (best != Backend::Builtin) CHECK(best_cut < sb.cut_fraction, "best external backend (%s %.3f) should beat the builtin fallback (%.3f)", backend_name(best), best_cut, sb.cut_fraction);
    if (pm.empty()) pm = pb;
    {   // hard caps: shrink device 0 far below its target, enforce, then everything must fit and still cover all nodes
        auto sm = evaluate(adj, pm, vw, fr);
        std::vector<uint64_t> cap(4, UINT64_MAX); cap[0] = sm.nodes[0] / 2; auto pc = pm; enforce_caps(adj, pc, cap);
        auto sc2 = evaluate(adj, pc, vw, fr); CHECK(sc2.nodes[0] <= cap[0], "cap violated: %zu > %llu", sc2.nodes[0], (unsigned long long)cap[0]);
        size_t t2 = 0; for (auto n : sc2.nodes) t2 += n; CHECK(t2 == N, "caps lost nodes");
        std::vector<uint64_t> tiny(4, 10); bool threw = false; auto pt = pm; try { enforce_caps(adj, pt, tiny); } catch (const std::runtime_error&) { threw = true; } CHECK(threw, "infeasible caps must throw");
    }
    bool bad = false; try { partition_with(parse_backend("scotch"), adj, vw, fr); } catch (const std::runtime_error& e) { bad = !has_scotch() && std::string(e.what()).find("not compiled") != std::string::npos; }
    CHECK(has_scotch() || bad, "an unavailable backend must fail with a clear message");
    bool pbad = false; try { parse_backend("nope"); } catch (const std::invalid_argument&) { pbad = true; } CHECK(pbad, "unknown backend name");
}

static void test_shard_provider() {
    std::printf("[GraphShardProvider -> client graph]\n");
    if (!have_cora()) { std::printf("  (Datasets/cora missing: skipped)\n"); return; }
    using namespace tf::fed::graph;
    TaskOptions o; o.parse("data=Datasets/cora"); o.parse("verbose=0");
    GraphShardProvider prov(o); prov.set_event_sink([](const std::string&) {});
    std::vector<EnrollInfo> en{{"a", 90.f, 0, 0}, {"b", 50.f, 0, 0}, {"c", 20.f, 0, 0}};
    prov.prepare(en);
    size_t nodes = 0, edges = 0, trainf = 0; const size_t N = prov.dataset().graph.num_nodes();
    TaskOptions co; co.parse("source=shard"); co.parse("features=1433"); co.parse("classes=7"); co.parse("verbose=0");
    for (auto& e : en) {
        auto sh = prov.shard_for(e.id); CHECK(sh != nullptr, "no shard for %s", e.id.c_str()); if (!sh) continue;
        nodes += size_t(sh->i("nodes")); edges += size_t(sh->i("edges"));
        for (auto v : sh->blob("flags")->u) trainf += v & 1u;
        NodeClassifierTask<GCN2> t(co); CHECK(t.wants_shard(), "shard mode must ask for a shard");
        Shard copy = *sh; t.on_shard(std::move(copy));
        CHECK(!t.wants_shard() && t.n_nodes() == size_t(sh->i("nodes")), "client graph size");
        CHECK(t.n_edges_local() == 2 * size_t(sh->i("edges")) + t.n_nodes(), "adjacency must be 2E + self loops: %zu vs %zu", t.n_edges_local(), 2 * size_t(sh->i("edges")) + t.n_nodes());
        CHECK(t.spec().param_count == 23040, "param layout independent of the data");
    }
    CHECK(nodes == N, "every node in exactly one shard: %zu vs %zu", nodes, N);
    CHECK(trainf == 140, "train labels must all be distributed (%zu)", trainf);
    CHECK(prov.shard_for("zzz") == nullptr, "unknown device");
    TaskOptions bad = co; bad.kv["classes"] = "5"; NodeClassifierTask<GCN2> t2(bad); bool threw = false; Shard c2 = *prov.shard_for("a");
    try { t2.on_shard(std::move(c2)); } catch (const std::runtime_error&) { threw = true; } CHECK(threw, "mismatching model config must be refused");
    std::printf("    cross-shard edges dropped: %zu of %zu undirected edges kept\n", edges, prov.dataset().graph.adjacency().col_idx.size() / 2 - N / 2);
}

static void test_ogb_pipeline() {
    std::printf("[OGB layout -> TFG1 -> FedGraphData]\n");
    if (std::system("python3 scripts/make_fake_ogb.py /tmp/tf_fake_ogb.zip 1500 24 5 6 >/dev/null && rm -rf /tmp/tf_ds && DATASETS_DIR=/tmp/tf_ds ./scripts/download_datasets.sh --from /tmp/tf_fake_ogb.zip ogbn-arxiv >/dev/null 2>&1") != 0) {
        std::printf("  (python3/gzip/unzip/g++ unavailable: skipped)\n"); return; }
    using namespace tf::fed::graph; TaskOptions o; o.parse("data=/tmp/tf_ds/ogbn-arxiv");
    auto h = tfg_read_header("/tmp/tf_ds/ogbn-arxiv/graph.tfg");
    CHECK(h.N == 1500 && h.F == 24 && h.C == 5, "header %llu/%llu/%u", (unsigned long long)h.N, (unsigned long long)h.F, h.C);
    size_t F = 0, C = 0, N = 0; dataset_dims(o, F, C, &N); CHECK(F == 24 && C == 5 && N == 1500, "dataset_dims");
    FedGraphData d = load_fed_graph(o);
    CHECK(d.graph.num_nodes() == 1500 && d.graph.feature_dim() == 24 && d.num_classes == 5 && d.official_split, "loaded sizes");
    CHECK(d.split.n_train == 750 && d.split.n_val == 375 && d.split.n_test == 375, "official split sizes %zu/%zu/%zu", d.split.n_train, d.split.n_val, d.split.n_test);
    size_t overlap = 0; for (size_t i = 0; i < 1500; i++) overlap += (d.split.train[i] + d.split.val[i] + d.split.test[i]) > 1; CHECK(overlap == 0, "splits overlap");
    const auto& adj = d.graph.adjacency(); bool sym = true; size_t selfl = 0;
    for (size_t u = 0; u < adj.n && sym; u++) for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; selfl += v == u; bool back = false; for (size_t k = adj.row_ptr[v]; k < adj.row_ptr[v + 1]; k++) if (size_t(adj.col_idx[k]) == u) { back = true; break; } if (!back) { sym = false; break; } }
    CHECK(sym && selfl == 1500, "adjacency must be symmetric with one self loop per node (selfloops %zu)", selfl);
    // a planted community structure must be learnable: class-dependent features => labels recoverable
    double same = 0, tot = 0; for (size_t u = 0; u < adj.n; u++) for (size_t e = adj.row_ptr[u]; e < adj.row_ptr[u + 1]; e++) { size_t v = adj.col_idx[e]; if (v != u) { tot++; same += d.graph.labels()[u] == d.graph.labels()[v]; } }
    CHECK(same / tot > 0.6, "planted homophily lost in conversion (%.2f)", same / tot);
    TaskOptions bad; bad.parse("data=/tmp/tf_ds/ogbn-arxiv/graph.tfg"); bad.parse("split=planetoid"); auto d2 = load_fed_graph(bad); CHECK(!d2.official_split && d2.split.n_train == 100, "planetoid override");
    std::system("head -c 3000 /tmp/tf_ds/ogbn-arxiv/graph.tfg > /tmp/tf_ds/trunc.tfg"); bool threw = false;
    try { TaskOptions t; t.parse("data=/tmp/tf_ds/trunc.tfg"); load_fed_graph(t); } catch (const std::runtime_error&) { threw = true; } CHECK(threw, "truncated file must be rejected");
}

static void test_shard_modes() {
    std::printf("[shard modes: partition | halo | full | memory downgrade | client request vs server override]\n");
    if (!have_cora()) { std::printf("  (Datasets/cora missing: skipped)\n"); return; }
    using namespace tf::fed::graph;
    auto run = [&](const std::string& server_mode, std::vector<EnrollInfo> en) {
        TaskOptions o; o.parse("data=Datasets/cora"); o.parse("shard_mode=" + server_mode);
        auto p = std::make_shared<GraphShardProvider>(o); p->set_event_sink([](const std::string&) {}); p->prepare(en); return p; };
    std::vector<EnrollInfo> en{{"a", 90.f, 0, 0, 1, 2}, {"b", 60.f, 0, 0, 2, 2}, {"c", 30.f, 0, 0, 3, 2}};   // a: partition, b: halo:2, c: full
    auto p = run("auto", en);
    const size_t N = p->dataset().graph.num_nodes();
    auto sa = p->shard_for("a"), sb = p->shard_for("b"), sc = p->shard_for("c");
    CHECK(sa->str("mode") == "partition" && sb->str("mode") == "halo:2" && sc->str("mode") == "full", "modes %s %s %s", sa->str("mode").c_str(), sb->str("mode").c_str(), sc->str("mode").c_str());
    CHECK(size_t(sc->i("nodes")) == N, "full shard must hold every node (%lld)", (long long)sc->i("nodes"));
    CHECK(sa->i("nodes") == sa->i("owned"), "partition shard holds only owned nodes");
    CHECK(sb->i("nodes") > sb->i("owned") && sb->i("edges") > sa->i("edges") * 0 && sb->i("nodes") < (long long)N, "halo must add context nodes but not everything");
    size_t owned_total = size_t(sa->i("owned") + sb->i("owned") + sc->i("owned")); CHECK(owned_total == N, "ownership must partition the graph exactly: %zu vs %zu", owned_total, N);
    auto trainbits = [](const Shard& s) { size_t t = 0, own = 0; for (auto v : s.blob("flags")->u) { t += v & 1u; own += (v & 8u) != 0; if ((v & 8u) == 0) { if (v != 0) return size_t(-1); } } (void)own; return t; };
    CHECK(trainbits(*sa) != size_t(-1) && trainbits(*sb) != size_t(-1) && trainbits(*sc) != size_t(-1), "non-owned nodes must carry no train/val/test flag");
    CHECK(trainbits(*sa) + trainbits(*sb) + trainbits(*sc) == 140, "every training label is trained by exactly one device");
    // full graph keeps ALL edges, partition drops cut edges: edges(full) = all undirected edges
    CHECK(sc->i("edges") >= sa->i("edges") && sc->i("edges") == (long long)((p->dataset().graph.adjacency().col_idx.size() - N) / 2), "full edge count %lld", (long long)sc->i("edges"));
    // client side: a full shard builds the whole graph, but only owned train nodes are trained
    { TaskOptions co; co.parse("source=shard"); co.parse("features=1433"); co.parse("classes=7"); co.parse("verbose=0");
      NodeClassifierTask<GCN2> t(co); Shard cp = *sc; t.on_shard(std::move(cp)); CHECK(t.n_nodes() == N && t.n_train() == trainbits(*sc), "full-graph client trains only its owned train nodes (%zu)", t.n_train()); }
    // server override beats client requests
    auto q = run("full", en); CHECK(q->shard_for("a")->str("mode") == "full" && q->shard_for("a")->str("requested") == "full", "server override");
    // memory downgrade: tiny budget on the device that asked for full
    std::vector<EnrollInfo> en2{{"a", 90.f, 0, 0, 3, 2}, {"b", 60.f, 1, 0, 3, 2}, {"c", 30.f, 0, 0, 1, 2}};
    auto r = run("auto", en2); auto sb2 = r->shard_for("b");
    CHECK(sb2->str("mode") != "full" && sb2->str("reason") == "memory budget" && sb2->str("requested") == "full", "downgrade: mode=%s reason=%s", sb2->str("mode").c_str(), sb2->str("reason").c_str());
    CHECK(r->shard_for("a")->str("mode") == "full" && r->shard_for("a")->str("reason").empty(), "unconstrained device keeps its request");
    bool threw = false; try { run("bogus", en); } catch (const std::invalid_argument&) { threw = true; } CHECK(threw, "bad shard_mode rejected");
}

int main() {
    test_shard_modes(); test_ogb_pipeline(); test_capacity_and_partitioners(); test_shard_provider(); test_strategies(); test_param_view(); test_handshake_and_hooks(); test_graph();
    std::printf("\n%d checks passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
