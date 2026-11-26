#pragma once
// TensorParams.hpp — bridge between ANY TensorF model and the flat vector the
// transport exchanges. Works on a list of tensors (Module::parameters()), so GCN,
// GraphSAGE, GAT, RGCN, an MLP or LoRA adapters all use the same code; the
// transport itself stays free of tensor types.
//
//   ParamView<float> pv(model.parameters());     // or any subset: only adapters, only the head, ...
//   pv.total()                                   // param_count for the TaskSpec
//   pv.fingerprint("gcn")                        // hash of algo + every tensor's shape
//   pv.get(off, len, out) / pv.set(off, data, len)   // chunked flat access, in place

#include "DataStructures/Tensor.hpp"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace tf { namespace fed {

template <typename T = float>
class ParamView {
    std::vector<Tensor_t<T>> ts_;
    std::vector<uint64_t> start_;      // start offset of each tensor in the flat vector
    uint64_t total_ = 0;
public:
    explicit ParamView(std::vector<Tensor_t<T>> tensors) : ts_(std::move(tensors)) {
        for (auto& t : ts_) { start_.push_back(total_); total_ += t->val.data.size(); }
    }
    uint64_t total() const { return total_; }
    size_t tensors() const { return ts_.size(); }

    /// FNV-1a over the algorithm name and each tensor's shape: two models with different
    /// architectures or hyper-parameters that change shapes get different fingerprints.
    uint32_t fingerprint(const std::string& algo) const {
        uint32_t h = 2166136261u;
        auto mix = [&](uint64_t v) { for (int i = 0; i < 8; i++) { h ^= uint32_t((v >> (8 * i)) & 0xFF); h *= 16777619u; } };
        for (unsigned char c : algo) { h ^= c; h *= 16777619u; }
        mix(ts_.size());
        for (auto& t : ts_) { mix(t->val.shape.size()); for (auto d : t->val.shape) mix(uint64_t(d)); }
        return h;
    }

    void get(uint64_t off, size_t len, float* out) const { walk(off, len, [&](Tensor_t<T>& t, size_t at, size_t n, size_t done) {
        for (size_t i = 0; i < n; i++) out[done + i] = static_cast<float>(t->val.data[at + i]); }); }
    void set(uint64_t off, const float* in, size_t len) { walk(off, len, [&](Tensor_t<T>& t, size_t at, size_t n, size_t done) {
        for (size_t i = 0; i < n; i++) t->val.data[at + i] = static_cast<T>(in[done + i]); }); }

    std::vector<float> flatten() const { std::vector<float> v(total_); get(0, total_, v.data()); return v; }
    void assign(const std::vector<float>& v) { if (v.size() != total_) throw std::invalid_argument("ParamView::assign: size mismatch"); set(0, v.data(), total_); }

private:
    template <class F> void walk(uint64_t off, size_t len, F f) const {
        if (off + len > total_) throw std::out_of_range("ParamView: range past end");
        size_t done = 0;
        for (size_t k = 0; k < ts_.size() && done < len; k++) {
            uint64_t s = start_[k], e = s + ts_[k]->val.data.size();
            uint64_t cur = off + done;
            if (cur >= e) continue;
            size_t n = static_cast<size_t>(std::min<uint64_t>(e - cur, len - done));
            f(const_cast<Tensor_t<T>&>(ts_[k]), static_cast<size_t>(cur - s), n, done);
            done += n;
        }
    }
};

}} // namespace tf::fed
