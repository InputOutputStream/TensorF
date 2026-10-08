// Correctness + speed checks for SpMMOperation and the sparse GCNLayer path.

#include "core/DataStructures/Graph.hpp"
#include "nn/Modules/GCNLayer.hpp"
#include "nn/Modules/Optimizer.hpp"
#include "core/Types/types.hpp"
#include <iostream>
#include <random>
#include <chrono>
#include <iostream>
#include <random>

static int failures = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::cout << "FAIL: " << msg << "\n"; failures++; } else std::cout << "ok:   " << msg << "\n"; } while (0)

static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
    double m = 0; for (size_t i = 0; i < a.size(); i++) m = std::max(m, std::abs(a[i] - b[i])); return m;
}

// ---- 1. finite-difference gradient check on a NON-symmetric CSR (scatter path) and symmetric flag path
static void gradcheck(bool use_symmetric_flag, bool symmetric_matrix) {
    const size_t N = 7, F = 3;
    std::mt19937 rng(1); 
    std::uniform_real_distribution<double> u(-1, 1);
    auto csr = std::make_shared<CSR<double>>();
    csr->n = N; 
    csr->row_ptr.assign(N + 1, 0);
    std::vector<std::vector<std::pair<node_t, double>>> rows(N);
    for (size_t i = 0; i < N; i++) for (size_t j = 0; j < N; j++)
        if (symmetric_matrix ? (i <= j && (i + j) % 3 == 0) : ((i * 3 + j) % 4 == 0)) {
            double v = u(rng); rows[i].push_back({node_t(j), v});
            if (symmetric_matrix && i != j) rows[j].push_back({node_t(i), v});
        }
    for (size_t i = 0; i < N; i++) {
        std::sort(rows[i].begin(), rows[i].end());
        for (auto& [c, v] : rows[i]) { csr->col_idx.push_back(c); csr->values.push_back(v); }
        csr->row_ptr[i + 1] = csr->col_idx.size();
    }
    csr->symmetric = use_symmetric_flag;

    std::vector<double> x0(N * F); for (auto& v : x0) v = u(rng);
    auto loss_of = [&](const std::vector<double>& xv, std::vector<double>* grad) {
        auto X = make_tensor<double>(Matrix<double>(xv, shape_t{N, F}));
        auto Y = spmm<double>(csr, X);
        auto L = (Y * Y)->sum();
        if (grad) { L->backward(make_tensor<double>(1.0)); *grad = X->grad.data; }
        double l = L->val.data[0]; L->reset_graph(); return l;
    };
    std::vector<double> analytic; loss_of(x0, &analytic);
    std::vector<double> numeric(N * F); const double h = 1e-6;
    for (size_t i = 0; i < N * F; i++) {
        auto xp = x0, xm = x0; xp[i] += h; xm[i] -= h;
        numeric[i] = (loss_of(xp, nullptr) - loss_of(xm, nullptr)) / (2 * h);
    }
    double d = max_abs_diff(analytic, numeric);
    CHECK(d < 1e-6, std::string("gradient check, ") + (symmetric_matrix ? "symmetric" : "non-symmetric") +
                    " matrix, symmetric flag=" + (use_symmetric_flag ? "true" : "false") + " (max diff " + std::to_string(d) + ")");
}

// ---- random sparse undirected graph
static Graph<float> random_graph(size_t N, size_t F, size_t avg_deg, unsigned seed) {
    std::mt19937 rng(seed); std::normal_distribution<float> nz(0.f, 1.f);
    Graph<float> g(F, N);
    for (size_t i = 0; i < N; i++) { std::vector<float> x(F); for (auto& v : x) v = nz(rng); g.add_node(x, int(i % 4)); }
    std::uniform_int_distribution<size_t> pick(0, N - 1);
    for (size_t e = 0; e < N * avg_deg / 2; e++) g.add_edge(node_t(pick(rng)), node_t(pick(rng)));
    g.build();
    return g;
}

int main() {
    gradcheck(false, false);
    gradcheck(false, true);
    gradcheck(true,  true);

    // ---- 2. sparse GCN layer == dense GCN layer (same weights): outputs and weight grads
    {
        auto g = random_graph(200, 16, 6, 3);
        auto X = features_tensor(g);
        auto A = dense_adjacency_tensor(g);
        GCNLayer<float> L(16, 5);
        auto run = [&](bool sparse, size_t threads) {
            L.zero_grad();
            auto out = sparse ? L.forward(g.adjacency_shared(), X, threads) : L.forward(A, X);
            auto loss = (out * out)->sum();
            loss->backward(make_tensor<float>(1.0f));
            std::vector<double> r(out->val.data.begin(), out->val.data.end());
            for (auto p : L.parameters()) r.insert(r.end(), p->grad.data.begin(), p->grad.data.end());
            loss->reset_graph();
            return r;
        };
        auto dense = run(false, 1), sp1 = run(true, 1), sp4 = run(true, 4);
        CHECK(max_abs_diff(dense, sp1) < 1e-3, "sparse GCN layer matches dense (output + weight/bias grads), diff " + std::to_string(max_abs_diff(dense, sp1)));
        CHECK(max_abs_diff(sp1, sp4) == 0.0, "4 threads give bit-identical results to 1 thread");
    }

    // ---- 3. training with the sparse path
    {
        const size_t PER = 30, F = 8, C = 2, N = PER * C;
        std::mt19937 rng(7); std::normal_distribution<float> noise(0.f, 0.35f);
        Graph<float> g(F, N);
        for (size_t c = 0; c < C; c++) for (size_t i = 0; i < PER; i++) {
            std::vector<float> x(F); for (size_t k = 0; k < F; k++) x[k] = noise(rng) + ((k % C == c) ? 1.5f : 0.f);
            g.add_node(x, int(c)); }
        g.connect_by_similarity(0.80f); g.build();
        std::vector<char> train(N, 0); for (size_t c = 0; c < C; c++) for (size_t i = 0; i < 3; i++) train[c * PER + i] = 1;
        auto X = features_tensor(g); auto Y = onehot_targets(g, C, train);
        GCNLayer<float> l1(F, 8), l2(8, C);
        std::vector<Tensor_t<float>> ps; for (auto p : l1.parameters()) ps.push_back(p); for (auto p : l2.parameters()) ps.push_back(p);
        Optimizer<float> opt(ps, 0.05f, ADAM, true);
        auto adj = g.adjacency_shared(); float last = 0; size_t ok = 0;
        for (int e = 0; e < 100; e++) {
            opt.zero_grad();
            auto out = l2.forward(adj, l1.forward(adj, X)->relu())->softmax();
            auto loss = Tensor<float>::cross_entropy(Y, out);
            loss->backward(make_tensor<float>(1.0f)); opt.step();
            last = loss->val.data[0];
            if (e == 99) for (size_t i = 0; i < N; i++) ok += ((out->val.data[i*C+1] > out->val.data[i*C]) ? 1 : 0) == (size_t)g.labels()[i];
            loss->reset_graph();
        }
        CHECK(last < 0.01f && ok == N, "sparse GCN trains: loss " + std::to_string(last) + ", accuracy " + std::to_string(100.0 * ok / N) + "%");
    }

    // ---- 4. speed: dense vs sparse, forward+backward of one layer, N=3000 (Cora-sized), ~4 neighbours/node
    {
        const size_t N = 3000, F = 64, H = 16;
        auto g = random_graph(N, F, 4, 5);
        auto X = features_tensor(g);
        GCNLayer<float> L(F, H);
        auto time_it = [&](auto&& fwd, int reps) {
            auto t0 = std::chrono::steady_clock::now();
            for (int r = 0; r < reps; r++) {
                L.zero_grad(); auto out = fwd(); auto loss = (out * out)->sum();
                loss->backward(make_tensor<float>(1.0f)); loss->reset_graph();
            }
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / reps;
        };
        auto A = dense_adjacency_tensor(g);
        double td = time_it([&] { return L.forward(A, X); }, 5);
        double ts = time_it([&] { return L.forward(g.adjacency_shared(), X); }, 20);
        std::cout << "N=" << N << " nnz=" << g.adjacency().nnz() << "  dense: " << td << " ms/iter ("
                  << (N * N * sizeof(float)) / 1048576.0 << " MB for A)   sparse: " << ts << " ms/iter ("
                  << (g.adjacency().nnz() * (sizeof(float) + sizeof(node_t))) / 1024.0 << " KB for A)\n";
    }
    std::cout << (failures ? "FAILURES: " : "all passed, failures=") << failures << "\n";
    return failures != 0;
}