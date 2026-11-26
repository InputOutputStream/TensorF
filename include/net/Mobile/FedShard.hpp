#pragma once
// FedShard.hpp — data shards: the server hands each device ITS part of the dataset.
//
// Why: for distributed training with a central data owner (compute offload to edge devices),
// a phone must not need the whole dataset. It gets only its shard, sized for its capability.
// (When data originates on the devices — classic federated learning — you do not use shards.)
//
// A shard is generic: string metadata + named blobs (float / fp16-on-the-wire / u32 arrays).
// The graph task puts node features, labels, split flags and local edges in it; another task
// can ship anything else. The transport never interprets the blobs.
//
// Flow:  client HELLO(score, mem) -> ShardReq -> [ShardWait ...] -> ShardHdr + one stream per blob
// ShardReq also ENROLS the device: the server closes enrollment (min devices + grace), runs the
// ShardProvider once with every device's capability, then serves the shards.

#include "FedWire.hpp"

#include <map>
#include <memory>

namespace tf { namespace fed {

struct Blob {
    enum class Kind : uint8_t { F32 = 0, F16 = 1 /* f32 in memory, half on the wire */, U32 = 2 };
    std::string name; Kind kind = Kind::F32;
    std::vector<float> f; std::vector<uint32_t> u;
    uint64_t count() const { return kind == Kind::U32 ? u.size() : f.size(); }
};

struct Shard {
    std::vector<std::pair<std::string, std::string>> meta;
    std::vector<Blob> blobs;
    void set(const std::string& k, const std::string& v) { for (auto& kv : meta) if (kv.first == k) { kv.second = v; return; } meta.emplace_back(k, v); }
    void set(const std::string& k, long long v) { set(k, std::to_string(v)); }
    std::string str(const std::string& k, const std::string& d = "") const { for (auto& kv : meta) if (kv.first == k) return kv.second; return d; }
    long long i(const std::string& k, long long d = 0) const { for (auto& kv : meta) if (kv.first == k) return std::stoll(kv.second); return d; }
    const Blob* blob(const std::string& n) const { for (auto& b : blobs) if (b.name == n) return &b; return nullptr; }
    uint64_t bytes() const { uint64_t t = 0; for (auto& b : blobs) t += b.count() * (b.kind == Blob::Kind::F16 ? 2 : 4); return t; }
};

/// Device capability as reported in HELLO.
/// Shard preference a device sends with ShardReq: 0 = let the server decide, 1 = partition only,
/// 2 = partition + `halo_hops` neighbourhood, 3 = the whole dataset.
enum class ShardPref : uint8_t { Auto = 0, Partition = 1, Halo = 2, Full = 3 };
struct EnrollInfo { std::string id; float score = 50.f; uint32_t mem_budget_mb = 0; uint8_t platform = 0; uint8_t shard_pref = 0; uint8_t halo_hops = 2; };

/// Server-side source of shards (e.g. a graph partitioner). Called by the coordinator.
class ShardProvider {
public:
    virtual ~ShardProvider() = default;
    /// Called ONCE, off the round lock, when enrollment closes. Throw to refuse (devices get the message).
    virtual void prepare(const std::vector<EnrollInfo>& devices) = 0;
    /// After prepare(): the shard for this device, or nullptr if it was not enrolled.
    virtual std::shared_ptr<const Shard> shard_for(const std::string& id) = 0;
    /// The coordinator installs a sink so providers can publish JSON events (e.g. partition quality) to the dashboard.
    virtual void set_event_sink(std::function<void(const std::string&)>) {}
};

constexpr size_t MAX_SHARD_BLOBS = 16;
constexpr uint64_t MAX_SHARD_ELEMS = 1ull << 31;   // sanity limit per blob on the receiving side

inline bool send_shard(int fd, const Shard& s, size_t chunk_elems) {
    if (s.blobs.size() > MAX_SHARD_BLOBS || s.meta.size() > 32) return false;
    Wr b; b.u8(uint8_t(s.meta.size()));
    for (auto& kv : s.meta) { b.str(kv.first); b.str(kv.second); }
    b.u8(uint8_t(s.blobs.size()));
    for (auto& bl : s.blobs) { b.str(bl.name); b.u8(uint8_t(bl.kind)); b.u64(bl.count()); }
    if (!send_msg(fd, Msg::ShardHdr, b)) return false;
    for (auto& bl : s.blobs) {
        bool ok = bl.kind == Blob::Kind::U32
            ? send_words(fd, bl.u.size(), chunk_elems, [&](uint64_t o, size_t n, uint32_t* out) { std::memcpy(out, bl.u.data() + o, n * 4); })
            : send_stream(fd, bl.kind == Blob::Kind::F16 ? Dtype::F16 : Dtype::F32, bl.f.size(), chunk_elems,
                          [&](uint64_t o, size_t n, float* out) { std::memcpy(out, bl.f.data() + o, n * 4); });
        if (!ok) return false;
    }
    return true;
}

/// Receive the body that follows a ShardHdr message header (`hdr_body` already read), then the blob streams.
inline bool recv_shard(int fd, const std::vector<uint8_t>& hdr_body, Shard& out, std::string& err) {
    Rd r(hdr_body); out = Shard();
    uint8_t nm = r.u8(); for (uint8_t i = 0; i < nm && r.ok; i++) { std::string k = r.str(), v = r.str(); out.meta.emplace_back(k, v); }
    uint8_t nb = r.u8();
    if (!r.ok || nb > MAX_SHARD_BLOBS) { err = "malformed shard header"; return false; }
    std::vector<uint64_t> counts;
    for (uint8_t i = 0; i < nb; i++) {
        Blob bl; bl.name = r.str(); uint8_t k = r.u8(); uint64_t c = r.u64();
        if (!r.ok || k > 2 || c > MAX_SHARD_ELEMS) { err = "malformed shard blob descriptor"; return false; }
        bl.kind = Blob::Kind(k); counts.push_back(c); out.blobs.push_back(std::move(bl));
    }
    if (!r.done()) { err = "malformed shard header"; return false; }
    for (size_t i = 0; i < out.blobs.size(); i++) {
        Blob& bl = out.blobs[i]; const uint64_t c = counts[i];
        if (bl.kind == Blob::Kind::U32) {
            bl.u.assign(c, 0);
            if (!recv_words(fd, c, [&](uint64_t o, const uint32_t* d, size_t n) { std::memcpy(bl.u.data() + o, d, n * 4); return true; }, err)) return false;
        } else {
            bl.f.assign(c, 0.f);
            if (!recv_stream(fd, bl.kind == Blob::Kind::F16 ? Dtype::F16 : Dtype::F32, c,
                             [&](uint64_t o, const float* d, size_t n) { std::memcpy(bl.f.data() + o, d, n * 4); return true; }, err)) return false;
        }
    }
    return true;
}

}} // namespace tf::fed
