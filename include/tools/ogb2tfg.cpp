// ogb2tfg — convert an extracted OGB node-property dataset (raw/*.csv.gz + split/<scheme>/*.csv.gz) to TFG1.
//   ogb2tfg --dir Datasets/.tmp/arxiv --out Datasets/ogbn-arxiv/graph.tfg [--split time]
// Streams everything (gzip -dc | parse), so ogbn-products converts in a few hundred MB of RAM.
// Needs only `gzip` on the PATH. Unsupported on purpose: multi-label (ogbn-proteins), heterogeneous
// (ogbn-mag), papers100M (needs out-of-core).
//
// Build:  g++ -std=c++17 -O2 -Iinclude/data tools/ogb2tfg.cpp -o bin/ogb2tfg
// Usage:  ogb2tfg --dir <extracted ogb dir containing raw/ and split/> --out graph.tfg [--split scheme]

#include "net/Mobile/graph/TfgFormat.hpp"
#include <cstdlib>
#include <cstdint>
#include <dirent.h>
#include <string>
#include <vector>
using namespace tf::fed::graph;

struct Lines {
    FILE* p = nullptr; char* buf = nullptr; size_t cap = 0;
    explicit Lines(const std::string& base) {                       // base without extension: tries .csv.gz then .csv
        std::string gz = base + ".csv.gz", plain = base + ".csv";
        std::string cmd;
        if (FILE* t = std::fopen(gz.c_str(), "rb")) { std::fclose(t); cmd = "gzip -dc '" + gz + "'"; }
        else if (FILE* t2 = std::fopen(plain.c_str(), "rb")) { std::fclose(t2); cmd = "cat '" + plain + "'"; }
        else throw std::runtime_error("missing " + gz);
        if (cmd.find('"') != std::string::npos) throw std::runtime_error("unsupported characters in path");
        p = popen(cmd.c_str(), "r"); if (!p) throw std::runtime_error("popen failed: " + cmd);
    }
    ~Lines() { if (p) pclose(p); std::free(buf); }
    bool next(const char*& s, ssize_t& n) { n = getline(&buf, &cap, p); s = buf; return n >= 0; }
};
static long long to_ll(const char* s, bool& ok) { char* e; while (*s == ' ') s++; long long v = std::strtoll(s, &e, 10); ok = e != s; return v; }

int main(int argc, char** argv) {
    std::string dir, out, scheme;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--dir" && i + 1 < argc) dir = argv[++i]; else if (a == "--out" && i + 1 < argc) out = argv[++i]; else if (a == "--split" && i + 1 < argc) scheme = argv[++i];
        else { std::fprintf(stderr, "usage: %s --dir <extracted ogb dir containing raw/ and split/> --out graph.tfg [--split scheme]\n", argv[0]); return 2; }
    }
    if (dir.empty() || out.empty()) { std::fprintf(stderr, "--dir and --out are required\n"); return 2; }
    try {
        if (scheme.empty()) {                                        // first sub-directory of split/ that has train.csv.gz
            std::string sd = dir + "/split"; DIR* d = opendir(sd.c_str()); if (!d) throw std::runtime_error("no split/ directory in " + dir);
            while (dirent* e = readdir(d)) { std::string n = e->d_name; if (n == "." || n == "..") continue;
                if (FILE* t = std::fopen((sd + "/" + n + "/train.csv.gz").c_str(), "rb")) { std::fclose(t); scheme = n; break; } }
            closedir(d); if (scheme.empty()) throw std::runtime_error("no split scheme found under " + sd);
        }
        TfgHeader h; std::vector<int32_t> lab; int32_t maxlab = -1;
        { Lines L(dir + "/raw/node-label"); const char* s; ssize_t n;
          while (L.next(s, n)) { bool ok; long long v = to_ll(s, ok); if (!ok || v < 0) v = -1; lab.push_back(int32_t(v)); maxlab = std::max<int32_t>(maxlab, int32_t(v)); } }
        if (lab.empty()) throw std::runtime_error("empty node-label file");
        if (maxlab < 0) throw std::runtime_error("no labelled nodes (multi-label datasets such as ogbn-proteins are not supported)");
        h.N = lab.size(); h.C = uint32_t(maxlab + 1);
        auto read_idx = [&](const std::string& name) { std::vector<uint32_t> v; Lines L(dir + "/split/" + scheme + "/" + name); const char* s; ssize_t n;
            while (L.next(s, n)) { bool ok; long long x = to_ll(s, ok); if (ok && x >= 0 && uint64_t(x) < h.N) v.push_back(uint32_t(x)); } return v; };
        auto tr = read_idx("train"), va = read_idx("valid"), te = read_idx("test");
        h.n_train = tr.size(); h.n_val = va.size(); h.n_test = te.size();

        FILE* f = std::fopen(out.c_str(), "wb"); if (!f) throw std::runtime_error("cannot write " + out);
        uint8_t hb[TfgHeader::BYTES]; h.pack(hb); std::fwrite(hb, 1, sizeof hb, f);          // placeholder, rewritten at the end
        std::fwrite(lab.data(), 4, lab.size(), f);
        std::fwrite(tr.data(), 4, tr.size(), f); std::fwrite(va.data(), 4, va.size(), f); std::fwrite(te.data(), 4, te.size(), f);
        { Lines L(dir + "/raw/node-feat"); const char* s; ssize_t n; size_t rows = 0; std::vector<float> row;
          while (L.next(s, n)) {
              row.clear(); const char* c = s;
              while (*c && *c != '\n' && *c != '\r') { char* e; float v = std::strtof(c, &e); if (e == c) throw std::runtime_error("bad feature value at row " + std::to_string(rows)); row.push_back(v); c = e; if (*c == ',') c++; }
              if (rows == 0) h.F = row.size(); else if (row.size() != h.F) throw std::runtime_error("ragged feature row " + std::to_string(rows));
              std::fwrite(row.data(), 4, row.size(), f); rows++;
          }
          if (rows != h.N) throw std::runtime_error("node-feat has " + std::to_string(rows) + " rows but node-label has " + std::to_string(h.N)); }
        { Lines L(dir + "/raw/edge"); const char* s; ssize_t n; std::vector<uint32_t> buf; buf.reserve(2 << 20);
          while (L.next(s, n)) { char* e; long long a = std::strtoll(s, &e, 10); if (e == s) continue; long long b = std::strtoll(e + 1, nullptr, 10);
              if (a < 0 || b < 0 || uint64_t(a) >= h.N || uint64_t(b) >= h.N) throw std::runtime_error("edge endpoint out of range");
              buf.push_back(uint32_t(a)); buf.push_back(uint32_t(b)); h.E++;
              if (buf.size() >= (2u << 20)) { std::fwrite(buf.data(), 4, buf.size(), f); buf.clear(); } }
          std::fwrite(buf.data(), 4, buf.size(), f); }
        std::fwrite("END1", 1, 4, f);
        h.pack(hb); std::fseek(f, 0, SEEK_SET); std::fwrite(hb, 1, sizeof hb, f);            // final header with F and E
        if (std::fclose(f) != 0) throw std::runtime_error("write error on " + out);
        std::printf("ogb2tfg: N=%llu F=%llu C=%u E=%llu train/val/test=%llu/%llu/%llu (split '%s') -> %s\n", (unsigned long long)h.N, (unsigned long long)h.F, h.C,
                    (unsigned long long)h.E, (unsigned long long)h.n_train, (unsigned long long)h.n_val, (unsigned long long)h.n_test, scheme.c_str(), out.c_str());
    } catch (const std::exception& e) { std::fprintf(stderr, "ogb2tfg: %s\n", e.what()); return 1; }
    return 0;
}
