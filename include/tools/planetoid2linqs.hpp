#ifndef __TENSORF_PLANETOID2LINQS_HPP_
#define __TENSORF_PLANETOID2LINQS_HPP_

// Convert a Planetoid-format dataset (Citeseer, Pubmed, Cora) into the LINQS text files that
// load_linqs() (GraphLoader.hpp) reads:
//
//   <out>/<name>.content   <id> <f_0> ... <f_{F-1}> <label>      (one node per line)
//   <out>/<name>.cites     <id_a> <id_b>                         (each undirected edge once)
//
//   std::string summary = planetoid_to_linqs("Datasets/citeseer/raw", "citeseer", "Datasets/citeseer");
//
// raw_dir holds ind.<name>.{allx,ally,tx,ty,graph,test.index}.  The out directory must exist.
// Throws std::runtime_error on any malformed or missing file.
//
// The Planetoid files are Python-2 pickles (protocol 2) of numpy arrays, scipy CSR matrices
// and a defaultdict(list).  This header contains a minimal pickle reader for exactly those
// opcodes; it never imports or executes anything from the file, it only builds plain values.
//
// Node layout (same as Kipf & Welling's loader): rows 0..n_allx-1 come from allx/ally, and
// row k of tx/ty belongs to the node whose id is on line k of test.index.  Nodes that appear
// in neither (Citeseer has 15) have no features and no label; they are dropped together
// with their edges.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace planetoid_detail {

// ───────────────────────── minimal pickle reader ─────────────────────────

struct Val;
using VP = std::shared_ptr<Val>;
struct Val {
    enum Kind { NONE, INT, STR, LIST, TUPLE, DICT, OBJ, GLOBAL, MARK } kind = NONE;
    long long i = 0;
    std::string s;                              // STR bytes, or "module.name" for GLOBAL / OBJ
    std::vector<VP> items;                      // LIST / TUPLE items, OBJ constructor args
    std::vector<std::pair<VP, VP>> dict;        // DICT entries, or entries set on an OBJ (defaultdict)
    VP state;                                   // OBJ state set by BUILD
};
inline VP mk(Val::Kind k) { auto v = std::make_shared<Val>(); v->kind = k; return v; }

class Unpickler {
    const std::string& b;
    size_t p = 0;
    std::vector<VP> st;
    std::unordered_map<uint32_t, VP> memo;

    [[noreturn]] void fail(const std::string& m) const {
        throw std::runtime_error("pickle: " + m + " at byte " + std::to_string(p));
    }
    uint8_t u8() { if (p >= b.size()) fail("unexpected end"); return (uint8_t)b[p++]; }
    uint32_t u32() { uint32_t v = 0; for (int k = 0; k < 4; k++) v |= (uint32_t)u8() << (8 * k); return v; }
    std::string bytes(size_t n) { if (p + n > b.size()) fail("string past end"); std::string r = b.substr(p, n); p += n; return r; }
    std::string line() { size_t e = b.find('\n', p); if (e == std::string::npos) fail("unterminated line"); std::string r = b.substr(p, e - p); p = e + 1; return r; }
    VP pop() { if (st.empty()) fail("stack underflow"); VP v = st.back(); st.pop_back(); return v; }
    std::vector<VP> pop_n(size_t n) {
        if (st.size() < n) fail("stack underflow");
        std::vector<VP> r(st.end() - n, st.end()); st.resize(st.size() - n); return r;
    }
    std::vector<VP> pop_mark() {
        size_t m = st.size();
        while (m > 0 && st[m - 1]->kind != Val::MARK) m--;
        if (m == 0) fail("no mark");
        std::vector<VP> r(st.begin() + m, st.end()); st.resize(m - 1); return r;
    }
    VP integer(long long v) { auto x = mk(Val::INT); x->i = v; return x; }
    VP str(std::string s) { auto x = mk(Val::STR); x->s = std::move(s); return x; }
    VP tuple(std::vector<VP> it) { auto x = mk(Val::TUPLE); x->items = std::move(it); return x; }
    VP long_n(size_t n) {                                   // little-endian two's complement
        std::string d = bytes(n);
        long long v = 0;
        for (size_t k = 0; k < n && k < 8; k++) v |= (long long)(uint8_t)d[k] << (8 * k);
        if (n > 0 && n < 8 && ((uint8_t)d[n - 1] & 0x80)) v -= (1LL << (8 * n));
        return integer(v);
    }
public:
    explicit Unpickler(const std::string& data) : b(data) {}
    VP load() {
        for (;;) {
            uint8_t op = u8();
            switch (op) {
                case 0x80: u8(); break;                                         // PROTO
                case '.': return pop();                                         // STOP
                case 'c': { auto g = mk(Val::GLOBAL); std::string m = line(); g->s = m + "." + line(); st.push_back(g); break; }
                case 'q': memo[u8()] = st.back(); break;                        // BINPUT
                case 'r': memo[u32()] = st.back(); break;                       // LONG_BINPUT
                case 'h': { auto it = memo.find(u8()); if (it == memo.end()) fail("bad BINGET"); st.push_back(it->second); break; }
                case 'j': { auto it = memo.find(u32()); if (it == memo.end()) fail("bad LONG_BINGET"); st.push_back(it->second); break; }
                case '(': st.push_back(mk(Val::MARK)); break;
                case ')': st.push_back(tuple({})); break;
                case 0x85: st.push_back(tuple(pop_n(1))); break;
                case 0x86: st.push_back(tuple(pop_n(2))); break;
                case 0x87: st.push_back(tuple(pop_n(3))); break;
                case 't': st.push_back(tuple(pop_mark())); break;
                case '}': st.push_back(mk(Val::DICT)); break;
                case ']': st.push_back(mk(Val::LIST)); break;
                case 'a': { VP v = pop(); st.back()->items.push_back(v); break; }                 // APPEND
                case 'e': { auto v = pop_mark(); auto& t = st.back()->items; t.insert(t.end(), v.begin(), v.end()); break; }
                case 's': { VP v = pop(); VP k = pop(); st.back()->dict.emplace_back(k, v); break; }  // SETITEM
                case 'u': { auto kv = pop_mark(); for (size_t k = 0; k + 1 < kv.size(); k += 2) st.back()->dict.emplace_back(kv[k], kv[k + 1]); break; }
                case 'K': st.push_back(integer(u8())); break;                    // BININT1
                case 'M': { unsigned lo = u8(); st.push_back(integer(lo | (unsigned)u8() << 8)); break; }  // BININT2
                case 'J': st.push_back(integer((int32_t)u32())); break;          // BININT
                case 0x8a: { size_t n = u8(); st.push_back(long_n(n)); break; }  // LONG1
                case 0x8b: { size_t n = u32(); st.push_back(long_n(n)); break; } // LONG4
                case 'I': { st.push_back(integer(std::stoll(line()))); break; }  // INT (text; also True/False)
                case 'L': { std::string l = line(); if (!l.empty() && l.back() == 'L') l.pop_back(); st.push_back(integer(std::stoll(l))); break; }
                case 'U': { size_t n = u8(); st.push_back(str(bytes(n))); break; }    // SHORT_BINSTRING
                case 'T': { size_t n = u32(); st.push_back(str(bytes(n))); break; }   // BINSTRING
                case 'X': { size_t n = u32(); st.push_back(str(bytes(n))); break; }   // BINUNICODE
                case 'N': st.push_back(mk(Val::NONE)); break;
                case 0x88: st.push_back(integer(1)); break;                      // NEWTRUE
                case 0x89: st.push_back(integer(0)); break;                      // NEWFALSE
                case '0': pop(); break;                                          // POP
                case '2': st.push_back(st.back()); break;                        // DUP
                case 0x81: {                                                     // NEWOBJ
                    VP args = pop(), cls = pop();
                    auto o = mk(Val::OBJ); o->s = cls->s; o->items = args->items; st.push_back(o); break;
                }
                case 'R': {                                                      // REDUCE
                    VP args = pop(), fn = pop();
                    auto o = mk(Val::OBJ); o->s = fn->s; o->items = args->items; st.push_back(o); break;
                }
                case 'b': { VP s = pop(); st.back()->state = s; break; }         // BUILD
                default: { char buf[64]; std::snprintf(buf, sizeof buf, "unsupported opcode 0x%02x", op); fail(buf); }
            }
        }
    }
};

// ───────────────────────── numpy / scipy extraction ─────────────────────────

struct Dense { size_t rows = 0, cols = 0; std::vector<double> v; double at(size_t r, size_t c) const { return v[r * cols + c]; } };
struct Csr { size_t rows = 0, cols = 0; std::vector<long long> indptr, indices; std::vector<double> data; };

inline VP dict_get(const VP& d, const std::string& key) {
    for (auto& kv : d->dict) if (kv.first->kind == Val::STR && kv.first->s == key) return kv.second;
    return nullptr;
}

// ndarray = _reconstruct(ndarray, (0,), 'b') + BUILD (version, shape, dtype, fortran, rawbytes)
inline std::vector<double> array_values(const VP& o, std::vector<long long>& shape) {
    if (!o || o->kind != Val::OBJ || !o->state || o->state->items.size() < 5) throw std::runtime_error("not an ndarray");
    auto& st = o->state->items;
    shape.clear();
    for (auto& d : st[1]->items) shape.push_back(d->i);
    if (st[3]->i) throw std::runtime_error("fortran-ordered arrays are not supported");
    const VP& dt = st[2];
    if (dt->items.empty() || dt->items[0]->kind != Val::STR) throw std::runtime_error("unsupported dtype (not a plain scalar type)");
    std::string code = dt->items[0]->s;                       // e.g. "i4", "f8", "b1"
    if (dt->state && dt->state->items.size() > 1 && dt->state->items[1]->s == ">") throw std::runtime_error("big-endian arrays are not supported");
    char kind = code[0];
    size_t sz = (size_t)std::stoi(code.substr(1));
    const std::string& raw = st[4]->s;
    size_t n = raw.size() / sz;
    std::vector<double> out(n);
    for (size_t k = 0; k < n; k++) {
        const char* q = raw.data() + k * sz;
        if (kind == 'f' && sz == 4) { float x; std::memcpy(&x, q, 4); out[k] = x; }
        else if (kind == 'f' && sz == 8) { double x; std::memcpy(&x, q, 8); out[k] = x; }
        else if (kind == 'i' && sz == 4) { int32_t x; std::memcpy(&x, q, 4); out[k] = x; }
        else if (kind == 'i' && sz == 8) { int64_t x; std::memcpy(&x, q, 8); out[k] = (double)x; }
        else if (kind == 'i' && sz == 2) { int16_t x; std::memcpy(&x, q, 2); out[k] = x; }
        else if ((kind == 'u' || kind == 'b') && sz == 1) out[k] = (uint8_t)*q;
        else if (kind == 'u' && sz == 4) { uint32_t x; std::memcpy(&x, q, 4); out[k] = x; }
        else if (kind == 'u' && sz == 8) { uint64_t x; std::memcpy(&x, q, 8); out[k] = (double)x; }
        else throw std::runtime_error("unsupported dtype " + code);
    }
    return out;
}

inline Dense as_dense(const VP& o) {
    std::vector<long long> shape;
    Dense d; d.v = array_values(o, shape);
    d.rows = (size_t)shape.at(0); d.cols = shape.size() > 1 ? (size_t)shape[1] : 1;
    if (d.rows * d.cols != d.v.size()) throw std::runtime_error("array shape does not match its data");
    return d;
}

inline Csr as_csr(const VP& o) {
    if (!o || o->kind != Val::OBJ || !o->state || o->state->kind != Val::DICT) throw std::runtime_error("not a scipy sparse matrix");
    Csr m; std::vector<long long> sh;
    auto ints = [&](const char* key) {
        std::vector<double> v = array_values(dict_get(o->state, key), sh);
        return std::vector<long long>(v.begin(), v.end());
    };
    m.indptr = ints("indptr"); m.indices = ints("indices");
    m.data = array_values(dict_get(o->state, "data"), sh);
    VP shape = dict_get(o->state, "_shape"); if (!shape) shape = dict_get(o->state, "shape");
    if (!shape || shape->items.size() != 2) throw std::runtime_error("sparse matrix without a shape");
    m.rows = (size_t)shape->items[0]->i; m.cols = (size_t)shape->items[1]->i;
    if (o->s.find("csr") == std::string::npos) throw std::runtime_error("expected a CSR matrix, got " + o->s);
    if (m.indptr.size() != m.rows + 1) throw std::runtime_error("bad CSR indptr");
    return m;
}

inline VP read_pickle(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::string data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return Unpickler(data).load();
}

} // namespace planetoid_detail

// ───────────────────────── conversion ─────────────────────────

// Returns a one-line summary (nodes, features, classes, edges).
inline std::string planetoid_to_linqs(const std::string& raw_dir, const std::string& name, const std::string& out_dir)
{
    using namespace planetoid_detail;
    auto path = [&](const char* part) { return raw_dir + "/ind." + name + "." + part; };

    Csr allx = as_csr(read_pickle(path("allx"))), tx = as_csr(read_pickle(path("tx")));
    Dense ally = as_dense(read_pickle(path("ally"))), ty = as_dense(read_pickle(path("ty")));
    VP graph = read_pickle(path("graph"));

    std::vector<long long> test_idx;
    { std::ifstream f(path("test.index")); long long v; while (f >> v) test_idx.push_back(v); }
    if (test_idx.size() != tx.rows || tx.rows != ty.rows) throw std::runtime_error("test.index does not match tx/ty");
    if (allx.cols != tx.cols || allx.rows != ally.rows) throw std::runtime_error("allx/ally/tx shapes disagree");

    const size_t F = allx.cols, C = ally.cols;
    long long max_id = (long long)allx.rows - 1;
    for (long long t : test_idx) max_id = std::max(max_id, t);
    for (auto& kv : graph->dict) {
        max_id = std::max(max_id, kv.first->i);
        for (auto& n : kv.second->items) max_id = std::max(max_id, n->i);
    }
    const size_t N = (size_t)max_id + 1;

    // per node: (csr matrix, row) of its features and its class (argmax of the one-hot label)
    std::vector<int> label(N, -1);
    std::vector<std::pair<const Csr*, size_t>> frow(N, {nullptr, 0});
    auto set_node = [&](size_t id, const Csr* X, size_t r, const Dense& Y, size_t yr) {
        frow[id] = {X, r};
        double best = 0; int arg = -1;
        for (size_t c = 0; c < C; c++) if (Y.at(yr, c) > best) { best = Y.at(yr, c); arg = (int)c; }
        label[id] = arg;
    };
    for (size_t i = 0; i < allx.rows; i++) set_node(i, &allx, i, ally, i);
    for (size_t k = 0; k < test_idx.size(); k++) set_node((size_t)test_idx[k], &tx, k, ty, k);

    bool binary = true;
    for (double d : allx.data) if (d != 1.0) { binary = false; break; }
    if (binary) for (double d : tx.data) if (d != 1.0) { binary = false; break; }

    std::vector<char> keep(N, 0);
    size_t kept = 0;
    for (size_t i = 0; i < N; i++) if (frow[i].first && label[i] >= 0) { keep[i] = 1; kept++; }

    std::set<std::pair<long long, long long>> edges;
    for (auto& kv : graph->dict)
        for (auto& n : kv.second->items)
            if (kv.first->i != n->i) edges.insert({std::min(kv.first->i, n->i), std::max(kv.first->i, n->i)});

    std::ofstream fc(out_dir + "/" + name + ".content"), fe(out_dir + "/" + name + ".cites");
    if (!fc || !fe) throw std::runtime_error("cannot write into " + out_dir + " (does the directory exist?)");
    std::vector<double> row(F);
    char buf[32];
    for (size_t i = 0; i < N; i++) {
        if (!keep[i]) continue;
        std::fill(row.begin(), row.end(), 0.0);
        const Csr& X = *frow[i].first; size_t r = frow[i].second;
        for (long long k = X.indptr[r]; k < X.indptr[r + 1]; k++) row[(size_t)X.indices[k]] += X.data[k];
        fc << i;
        for (size_t c = 0; c < F; c++) {
            if (row[c] == 0.0) { fc << " 0"; continue; }
            std::snprintf(buf, sizeof buf, binary ? "%.0f" : "%.6g", row[c]);
            fc << ' ' << buf;
        }
        fc << " c" << label[i] << '\n';
    }
    size_t written = 0;
    for (auto& e : edges)
        if (keep[(size_t)e.first] && keep[(size_t)e.second]) { fe << e.first << ' ' << e.second << '\n'; written++; }

    std::ostringstream os;
    os << name << ": " << kept << " nodes (" << N - kept << " without features/label dropped), " << F
       << " features (" << (binary ? "binary" : "real-valued") << "), " << C << " classes, " << written
       << " undirected edges (" << edges.size() - written << " touched dropped nodes)";
    return os.str();
}

#endif
