#ifndef __TENSORF_GRAPHLOADER_HPP_
#define __TENSORF_GRAPHLOADER_HPP_

// Loader for citation-network datasets in the LINQS format (Cora, Citeseer):
//
//   <name>.content   one node per line, whitespace separated:
//                    <paper_id> <feature_1> ... <feature_F> <class_label>
//   <name>.cites     one edge per line:  <cited_paper_id> <citing_paper_id>
//
// The loader maps paper ids (any string) to node indices 0..N-1, maps class names to
// integers 0..C-1 (alphabetical, so the mapping is deterministic), builds the Graph
// and calls Graph::build() (symmetrise + self loops + D^-1/2 (A+I) D^-1/2).
// Unknown ids in the .cites file are skipped and counted (Citeseer has a few).

#include "DataStructures/Graph.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T>
struct GraphDataset {
    Graph<T> graph;
    std::vector<std::string> class_names;      // class_names[label] -> original text
    size_t skipped_edges = 0;                  // edges whose endpoint id was not in .content
    size_t num_classes() const { return class_names.size(); }
    explicit GraphDataset(Graph<T>&& g) : graph(std::move(g)) {}
};

// row_normalize: divide every feature row by its sum 
template <typename T>
GraphDataset<T> load_linqs(const std::string& content_path, const std::string& cites_path,
                           bool row_normalize = true)
{
    // ---- 1. .content --------------------------------------------------------
    std::ifstream fc(content_path);
    if (!fc) throw std::runtime_error("load_linqs: cannot open " + content_path);

    std::vector<std::string> ids, labels_txt;
    std::vector<T> flat;
    size_t F = 0;
    std::string line;
    size_t line_no = 0;
    while (std::getline(fc, line)) {
        line_no++;
        while (!line.empty() && std::isspace((unsigned char)line.back())) line.pop_back();
        if (line.empty()) continue;
        const char* s = line.c_str();
        const char* e = s + line.size();

        const char* p = s;                                   // token 0: paper id
        while (p < e && !std::isspace((unsigned char)*p)) p++;
        const char* q = e;                                   // last token: class label
        while (q > p && !std::isspace((unsigned char)q[-1])) q--;
        if (q <= p) throw std::runtime_error(content_path + ":" + std::to_string(line_no) + ": expected id, features, label");

        size_t before = flat.size();
        const char* c = p;
        while (c < q) {                                      // numeric features between id and label
            while (c < q && std::isspace((unsigned char)*c)) c++;
            if (c >= q) break;
            char* endp = nullptr;
            T v = static_cast<T>(std::strtof(c, &endp));
            if (endp == c) throw std::runtime_error(content_path + ":" + std::to_string(line_no) + ": bad feature value");
            flat.push_back(v);
            c = endp;
        }
        size_t nf = flat.size() - before;
        if (F == 0) F = nf;
        if (nf != F || F == 0)
            throw std::runtime_error(content_path + ":" + std::to_string(line_no) + ": expected " + std::to_string(F) +
                                     " features, found " + std::to_string(nf));
        ids.emplace_back(s, p);
        labels_txt.emplace_back(q, e);
    }
    const size_t N = ids.size();
    if (N == 0) throw std::runtime_error("load_linqs: " + content_path + " is empty");

    // ---- 2. class names -> 0..C-1 (alphabetical) -----------------------------
    std::map<std::string, int> cls;
    for (auto& l : labels_txt) cls.emplace(l, 0);
    std::vector<std::string> class_names;
    for (auto& kv : cls) { kv.second = int(class_names.size()); class_names.push_back(kv.first); }

    // ---- 3. nodes -------------------------------------------------------------
    Graph<T> g(F, N);
    std::unordered_map<std::string, node_t> index;
    index.reserve(N * 2);
    for (size_t i = 0; i < N; i++) {
        T* row = &flat[i * F];
        if (row_normalize) {
            double sum = 0;
            for (size_t k = 0; k < F; k++) sum += row[k];
            if (sum > 0) for (size_t k = 0; k < F; k++) row[k] = static_cast<T>(row[k] / sum);
        }
        node_t id = g.add_node(row, F, cls[labels_txt[i]]);
        if (!index.emplace(ids[i], id).second)
            throw std::runtime_error("load_linqs: duplicate paper id " + ids[i]);
    }

    GraphDataset<T> ds(std::move(g));
    ds.class_names = std::move(class_names);

    // ---- 4. .cites -------------------------------------------------------------
    std::ifstream fe(cites_path);
    if (!fe) throw std::runtime_error("load_linqs: cannot open " + cites_path);
    std::string a, b;
    while (fe >> a >> b) {
        auto ia = index.find(a), ib = index.find(b);
        if (ia == index.end() || ib == index.end()) { ds.skipped_edges++; continue; }
        ds.graph.add_edge(ia->second, ib->second);
    }
    ds.graph.build();
    return ds;
}

// Planetoid-style split: `per_class` training nodes per class, then n_val validation and
// n_test test nodes drawn from the remaining labelled nodes. Deterministic for a given seed.
struct Split {
    std::vector<char> train, val, test;           // masks of size N
    size_t n_train = 0, n_val = 0, n_test = 0;
};

inline Split make_split(const std::vector<int>& labels, size_t num_classes, size_t per_class = 20,
                        size_t n_val = 500, size_t n_test = 1000, unsigned seed = 42)
{
    const size_t N = labels.size();
    Split s;
    s.train.assign(N, 0); s.val.assign(N, 0); s.test.assign(N, 0);
    std::mt19937 rng(seed);

    std::vector<std::vector<size_t>> by_class(num_classes);
    for (size_t i = 0; i < N; i++) if (labels[i] >= 0) by_class[labels[i]].push_back(i);

    std::vector<size_t> rest;
    for (auto& v : by_class) {
        std::shuffle(v.begin(), v.end(), rng);
        for (size_t k = 0; k < v.size(); k++) {
            if (k < per_class) { s.train[v[k]] = 1; s.n_train++; }
            else rest.push_back(v[k]);
        }
    }
    std::shuffle(rest.begin(), rest.end(), rng);
    for (size_t k = 0; k < rest.size(); k++) {
        if (k < n_val)                { s.val[rest[k]] = 1;  s.n_val++; }
        else if (k < n_val + n_test)  { s.test[rest[k]] = 1; s.n_test++; }
    }
    return s;
}

#endif