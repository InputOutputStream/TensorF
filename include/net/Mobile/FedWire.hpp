#pragma once
// FedWire.hpp — wire protocol of the mobile-oriented federated layer.
//
// Differences from the legacy Client/Server protocol (include/net/Network):
//   * every message starts with a fixed 8-byte header (magic, version, type,
//     body length) -> garbage / wrong-version peers are rejected immediately;
//   * all integers are little-endian on the wire (explicit, not "whatever the
//     host is"), so ARM phones and x86 servers always agree;
//   * the session starts with HELLO/WELCOME: client id, optional auth token,
//     algorithm name (e.g. "gcn", "sage", "gat" — checked by the server),
//     parameter count, model fingerprint, device hints (RAM, chunk size,
//     metered link), negotiated payload dtype (fp32 or fp16);
//   * rounds are explicit: READY -> GO|SKIP -> update stream -> RESULT + global
//     stream. A client that is too late / stale is told SKIP instead of
//     silently corrupting a round in progress.
//
// Stream format (both directions): [u64 total_elems] then repeated
// [u32 chunk_len][chunk_len * elem_size bytes]. Sender and receiver pick chunk
// sizes independently; peak extra memory is one chunk.
//
// Header-only, POSIX sockets, no dependency on the rest of TensorF except
// io_utils.hpp (read_exact / write_exact + bandwidth counters).

#include "../Network/io_utils.hpp"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace tf { namespace fed {

constexpr uint32_t MAGIC   = 0x44464654u;  // bytes 'T','F','F','D' on the wire
constexpr uint8_t  VERSION = 2;   // v2: HELLO carries device capability (score, memory), shard messages
constexpr size_t   MAX_BODY       = 1024;          // control messages are tiny
constexpr uint32_t MAX_CHUNK_ELEMS = 16u * 1024 * 1024;
constexpr size_t   MAX_STR        = 200;

enum class Msg : uint8_t { Hello = 1, Welcome = 2, Ready = 3, Go = 4, Skip = 5,
                           Result = 6, SyncReq = 7, Bye = 8,
                           ShardReq = 9,    // client -> server: "send me my data shard" (also enrols the device)
                           ShardHdr = 10,   // server -> client: shard description, followed by one stream per blob
                           ShardWait = 11 };// server -> client: not ready yet / denied
enum class Dtype : uint8_t { F32 = 0, F16 = 1 };
enum class Platform : uint8_t { Unknown = 0, Linux = 1, Android = 2, Ios = 3, Other = 4 };

inline size_t elem_size(Dtype d) { return d == Dtype::F16 ? 2 : 4; }

// ── endianness -------------------------------------------------------------
inline bool host_is_le() { const uint16_t x = 1; uint8_t b; std::memcpy(&b, &x, 1); return b == 1; }
inline uint32_t bswap32(uint32_t v) { return __builtin_bswap32(v); }

// ── fp16 <-> fp32 (IEEE binary16, round-to-nearest-even) ------------------
inline uint16_t float_to_half(float f) {
    uint32_t x; std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e = (x >> 23) & 0xFFu;
    uint32_t m = x & 0x7FFFFFu;
    if (e == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (m ? 0x200u | (m >> 13) : 0u)); // inf / nan
    int32_t he = static_cast<int32_t>(e) - 127 + 15;
    if (he >= 31) return static_cast<uint16_t>(sign | 0x7C00u);            // overflow -> inf
    if (he <= 0) {                                                           // subnormal / zero
        if (he < -10) return static_cast<uint16_t>(sign);
        m |= 0x800000u;
        uint32_t shift = static_cast<uint32_t>(14 - he);
        uint32_t half_m = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u), mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half_m & 1u))) half_m++;
        return static_cast<uint16_t>(sign | half_m);
    }
    uint32_t half_m = m >> 13, rem = m & 0x1FFFu;
    uint32_t h = sign | (static_cast<uint32_t>(he) << 10) | half_m;
    if (rem > 0x1000u || (rem == 0x1000u && (half_m & 1u))) h++;            // carry may bump exponent: correct
    return static_cast<uint16_t>(h);
}
inline float half_to_float(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu, x;
    if (e == 0) {
        if (m == 0) x = sign;
        else { int s = 0; while (!(m & 0x400u)) { m <<= 1; s++; } m &= 0x3FFu;
               x = sign | (static_cast<uint32_t>(113 - s) << 23) | (m << 13); }
    } else if (e == 31) x = sign | 0x7F800000u | (m << 13);
    else x = sign | ((e + 112) << 23) | (m << 13);
    float f; std::memcpy(&f, &x, 4); return f;
}

// ── tiny serializer for control bodies ------------------------------------
struct Wr {
    std::vector<uint8_t> b;
    void u8(uint8_t v)  { b.push_back(v); }
    void u16(uint16_t v){ for (int i = 0; i < 2; i++) b.push_back((v >> (8 * i)) & 0xFF); }
    void u32(uint32_t v){ for (int i = 0; i < 4; i++) b.push_back((v >> (8 * i)) & 0xFF); }
    void u64(uint64_t v){ for (int i = 0; i < 8; i++) b.push_back((v >> (8 * i)) & 0xFF); }
    void str(const std::string& s) {
        size_t n = s.size() > MAX_STR ? MAX_STR : s.size();
        u8(static_cast<uint8_t>(n)); b.insert(b.end(), s.begin(), s.begin() + n);
    }
};
struct Rd {
    const uint8_t* p; size_t n; size_t i = 0; bool ok = true;
    Rd(const std::vector<uint8_t>& v) : p(v.data()), n(v.size()) {}
    bool need(size_t k) { if (i + k > n) { ok = false; return false; } return true; }
    uint8_t u8()  { return need(1) ? p[i++] : 0; }
    uint16_t u16(){ uint16_t v = 0; if (need(2)) { for (int k = 0; k < 2; k++) v |= uint16_t(p[i + k]) << (8 * k); i += 2; } return v; }
    uint32_t u32(){ uint32_t v = 0; if (need(4)) { for (int k = 0; k < 4; k++) v |= uint32_t(p[i + k]) << (8 * k); i += 4; } return v; }
    uint64_t u64(){ uint64_t v = 0; if (need(8)) { for (int k = 0; k < 8; k++) v |= uint64_t(p[i + k]) << (8 * k); i += 8; } return v; }
    std::string str() { uint8_t l = u8(); if (!need(l)) return {}; std::string s(reinterpret_cast<const char*>(p + i), l); i += l; return s; }
    bool done() const { return ok && i == n; }
};

// ── message framing -------------------------------------------------------
inline bool send_msg(int fd, Msg t, const Wr& body) {
    if (body.b.size() > MAX_BODY) return false;
    Wr h; h.u32(MAGIC); h.u8(VERSION); h.u8(static_cast<uint8_t>(t)); h.u16(static_cast<uint16_t>(body.b.size()));
    std::vector<uint8_t> all = h.b; all.insert(all.end(), body.b.begin(), body.b.end());
    return write_exact(fd, all.data(), all.size());       // single write: one TCP segment for small messages
}
/// Returns false on I/O failure OR protocol violation (err says which).
inline bool recv_msg(int fd, Msg& t, std::vector<uint8_t>& body, std::string& err) {
    uint8_t h[8];
    if (!read_exact(fd, h, 8)) { err = "io"; return false; }
    std::vector<uint8_t> hv(h, h + 8);   // named: Rd keeps a pointer into it
    Rd r(hv);
    uint32_t magic = r.u32(); uint8_t ver = r.u8(); uint8_t type = r.u8(); uint16_t len = r.u16();
    if (magic != MAGIC) { err = "bad magic"; return false; }
    if (ver != VERSION) { err = "unsupported protocol version " + std::to_string(ver); return false; }
    if (type < 1 || type > 11) { err = "unknown message type"; return false; }
    if (len > MAX_BODY) { err = "body too large"; return false; }
    body.assign(len, 0);
    if (len && !read_exact(fd, body.data(), len)) { err = "io"; return false; }
    t = static_cast<Msg>(type);
    return true;
}

// ── chunked element streams ----------------------------------------------
using GetChunk = std::function<void(uint64_t offset, size_t len, float* out)>;
using OnChunk  = std::function<bool(uint64_t offset, const float* data, size_t len)>;  // false = abort

inline bool send_stream(int fd, Dtype dt, uint64_t total, size_t chunk_elems, const GetChunk& get) {
    if (!write_exact(fd, std::vector<uint8_t>{
            uint8_t(total), uint8_t(total >> 8), uint8_t(total >> 16), uint8_t(total >> 24),
            uint8_t(total >> 32), uint8_t(total >> 40), uint8_t(total >> 48), uint8_t(total >> 56)}.data(), 8))
        return false;
    if (chunk_elems == 0) chunk_elems = 1;
    if (chunk_elems > MAX_CHUNK_ELEMS) chunk_elems = MAX_CHUNK_ELEMS;
    std::vector<float> stage(chunk_elems);
    std::vector<uint8_t> bytes(4 + chunk_elems * elem_size(dt));
    const bool le = host_is_le();
    for (uint64_t off = 0; off < total;) {
        size_t take = static_cast<size_t>(std::min<uint64_t>(chunk_elems, total - off));
        get(off, take, stage.data());
        for (int k = 0; k < 4; k++) bytes[k] = uint8_t(take >> (8 * k));
        uint8_t* o = bytes.data() + 4;
        if (dt == Dtype::F32) {
            if (le) std::memcpy(o, stage.data(), take * 4);
            else for (size_t i = 0; i < take; i++) { uint32_t v; std::memcpy(&v, &stage[i], 4); v = bswap32(v); std::memcpy(o + 4 * i, &v, 4); }
        } else {
            for (size_t i = 0; i < take; i++) { uint16_t h = float_to_half(stage[i]); o[2 * i] = uint8_t(h); o[2 * i + 1] = uint8_t(h >> 8); }
        }
        if (!write_exact(fd, bytes.data(), 4 + take * elem_size(dt))) return false;
        off += take;
    }
    return true;
}

/// Receives a stream. `expected_total` is checked against the advertised total
/// BEFORE anything is allocated. `on_chunk` returning false aborts (returns false,
/// *aborted=true). `received` (optional) tracks elements delivered so far.
inline bool recv_stream(int fd, Dtype dt, uint64_t expected_total, const OnChunk& on_chunk,
                        std::string& err, bool* aborted = nullptr) {
    uint8_t tb[8];
    if (!read_exact(fd, tb, 8)) { err = "io"; return false; }
    uint64_t total = 0; for (int k = 0; k < 8; k++) total |= uint64_t(tb[k]) << (8 * k);
    if (total != expected_total) {
        err = "size mismatch: stream advertises " + std::to_string(total) + " elements, expected " + std::to_string(expected_total);
        return false;
    }
    std::vector<uint8_t> raw; std::vector<float> stage;
    const bool le = host_is_le();
    for (uint64_t got = 0; got < total;) {
        uint8_t lb[4];
        if (!read_exact(fd, lb, 4)) { err = "io"; return false; }
        uint32_t n = uint32_t(lb[0]) | uint32_t(lb[1]) << 8 | uint32_t(lb[2]) << 16 | uint32_t(lb[3]) << 24;
        if (n == 0 || n > MAX_CHUNK_ELEMS || n > total - got) { err = "bad chunk length"; return false; }
        raw.resize(size_t(n) * elem_size(dt)); stage.resize(n);
        if (!read_exact(fd, raw.data(), raw.size())) { err = "io"; return false; }
        if (dt == Dtype::F32) {
            if (le) std::memcpy(stage.data(), raw.data(), raw.size());
            else for (size_t i = 0; i < n; i++) { uint32_t v; std::memcpy(&v, &raw[4 * i], 4); v = bswap32(v); std::memcpy(&stage[i], &v, 4); }
        } else {
            for (size_t i = 0; i < n; i++) stage[i] = half_to_float(uint16_t(raw[2 * i] | (raw[2 * i + 1] << 8)));
        }
        if (!on_chunk(got, stage.data(), n)) { if (aborted) *aborted = true; err = "aborted"; return false; }
        got += n;
    }
    return true;
}

// ── raw u32 streams (graph structure, labels, flags): same framing, no float conversion ─────
using GetWords = std::function<void(uint64_t offset, size_t len, uint32_t* out)>;
using OnWords  = std::function<bool(uint64_t offset, const uint32_t* data, size_t len)>;

inline bool send_words(int fd, uint64_t total, size_t chunk_elems, const GetWords& get) {
    uint8_t tb[8]; for (int k = 0; k < 8; k++) tb[k] = uint8_t(total >> (8 * k));
    if (!write_exact(fd, tb, 8)) return false;
    if (chunk_elems == 0) chunk_elems = 1;
    if (chunk_elems > MAX_CHUNK_ELEMS) chunk_elems = MAX_CHUNK_ELEMS;
    std::vector<uint32_t> stage(chunk_elems); std::vector<uint8_t> bytes(4 + chunk_elems * 4);
    for (uint64_t off = 0; off < total;) {
        size_t take = static_cast<size_t>(std::min<uint64_t>(chunk_elems, total - off));
        get(off, take, stage.data());
        for (int k = 0; k < 4; k++) bytes[k] = uint8_t(take >> (8 * k));
        for (size_t i = 0; i < take; i++) for (int k = 0; k < 4; k++) bytes[4 + 4 * i + k] = uint8_t(stage[i] >> (8 * k));
        if (!write_exact(fd, bytes.data(), 4 + take * 4)) return false;
        off += take;
    }
    return true;
}
inline bool recv_words(int fd, uint64_t expected_total, const OnWords& on_chunk, std::string& err) {
    uint8_t tb[8];
    if (!read_exact(fd, tb, 8)) { err = "io"; return false; }
    uint64_t total = 0; for (int k = 0; k < 8; k++) total |= uint64_t(tb[k]) << (8 * k);
    if (total != expected_total) { err = "size mismatch in word stream"; return false; }
    std::vector<uint8_t> raw; std::vector<uint32_t> stage;
    for (uint64_t got = 0; got < total;) {
        uint8_t lb[4]; if (!read_exact(fd, lb, 4)) { err = "io"; return false; }
        uint32_t n = uint32_t(lb[0]) | uint32_t(lb[1]) << 8 | uint32_t(lb[2]) << 16 | uint32_t(lb[3]) << 24;
        if (n == 0 || n > MAX_CHUNK_ELEMS || n > total - got) { err = "bad chunk length"; return false; }
        raw.resize(size_t(n) * 4); stage.resize(n);
        if (!read_exact(fd, raw.data(), raw.size())) { err = "io"; return false; }
        for (size_t i = 0; i < n; i++) stage[i] = uint32_t(raw[4*i]) | uint32_t(raw[4*i+1]) << 8 | uint32_t(raw[4*i+2]) << 16 | uint32_t(raw[4*i+3]) << 24;
        if (!on_chunk(got, stage.data(), n)) { err = "aborted"; return false; }
        got += n;
    }
    return true;
}

}} // namespace tf::fed
