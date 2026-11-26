#pragma once
// CapacityPolicy.hpp — turns device profiler scores into partition targets.
// Separates the SCHEDULING POLICY (who should get how much) from the PARTITIONER
// (how to cut the graph); you can change either without touching the other.
//
//   score s in [0,100]  --capacity()-->  relative capacity c(s)  --normalise-->  target fraction t_i
//
// Default policy ("saturating proportional"): c(s) = clamp(s, floor, hi)
//   s <  lo  : share proportional to the score (floored so nobody gets nothing)
//   lo..hi   : same line (continuous)
//   s >= hi  : saturates: every strong device gets the same maximum share — beyond some
//              point extra speed is eaten by synchronisation/network, and it keeps one
//              device from becoming a single point of failure.
// hi=100 gives the plain proportional policy  t_i = s_i / sum_j s_j.
//
// Hard memory limits are a different thing from targets: METIS' ubvec is a tolerance,
// not a limit. DeviceCaps::mem_budget_mb is turned into a node cap that the partitioners
// enforce afterwards (see WeightedPartitioner.hpp::enforce_caps).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf { namespace fed { namespace graph {

struct DeviceCaps {
    std::string id;
    float    score = 50.f;            // 0..100, from the device profiler (see MobileClient quick_score / include/profiler)
    uint32_t mem_budget_mb = 0;       // memory the device is willing to spend on its shard (0 = unlimited)
    uint8_t  platform = 0;
};

struct CapacityPolicy {
    float lo = 30.f, hi = 70.f;       // thresholds from the design discussion
    float floor_score = 5.f;          // minimum effective score
    double capacity(float s) const { return std::max<double>(floor_score, std::min<double>(s, hi)); }
};

/// Target fraction of the total vertex weight for each device (sums to 1).
inline std::vector<double> target_fractions(const std::vector<DeviceCaps>& d, const CapacityPolicy& p) {
    if (d.empty()) throw std::invalid_argument("target_fractions: no devices");
    std::vector<double> c(d.size()); double sum = 0;
    for (size_t i = 0; i < d.size(); i++) { c[i] = p.capacity(d[i].score); sum += c[i]; }
    for (auto& x : c) x /= sum;
    return c;
}

/// Estimated bytes a device needs per local node: features (fp32), labels/flags/ids, adjacency entries.
inline double bytes_per_node(size_t feat_dim, double avg_degree, bool fp16_features = false) {
    return double(feat_dim) * (fp16_features ? 2 : 4) + 16.0 + avg_degree * 8.0 /*src,dst u32*/ + 24.0 /*activations/CSR slack*/;
}

/// Hard cap on the NUMBER OF NODES per device from its memory budget (0 = unlimited -> UINT64_MAX).
/// `usable` is the fraction of the budget available for the shard (the rest is model, optimizer, activations).
inline std::vector<uint64_t> node_caps(const std::vector<DeviceCaps>& d, double bytes_node, double usable = 0.5, double act_factor = 4.0) {
    std::vector<uint64_t> cap(d.size(), UINT64_MAX);
    for (size_t i = 0; i < d.size(); i++)
        if (d[i].mem_budget_mb) cap[i] = uint64_t(double(d[i].mem_budget_mb) * 1048576.0 * usable / (bytes_node * act_factor));
    return cap;
}

}}} // namespace tf::fed::graph
