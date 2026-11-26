#pragma once
// FedStrategies.hpp — distribution methods as plug-ins, independent of the model.
//
// Server-side optimizers plug into CoordinatorConfig::post_aggregate. Given the
// previous global vector w and the weighted client average a, define the
// pseudo-gradient  d = a - w  and apply a rule:
//   fedavg   : w' = a                                   (the default, no hook needed)
//   fedavgm  : v = beta*v + d ;  w' = w + lr*v          (Hsu et al. 2019)
//   fedadam  : m = b1*m + (1-b1)d ; v = b2*v + (1-b2)d^2 ;  w' = w + lr*m/(sqrt(v)+tau)   (Reddi et al. 2021)
// Client-side FedProx (Li et al. 2020) is a pull toward the received global vector
// applied after each local step: fedprox_pull().
//
// Spec strings (CLI `--strategy`): "fedavg", "fedavgm:0.9[:lr]", "fedadam:0.01[:b1:b2:tau]".

#include "FedCoordinator.hpp"

#include <cmath>
#include <sstream>

namespace tf { namespace fed {

using PostAggregate = std::function<void(uint64_t, const std::vector<float>&, std::vector<float>&)>;

inline PostAggregate fedavgm(double beta = 0.9, double lr = 1.0) {
    auto v = std::make_shared<std::vector<float>>();
    return [=](uint64_t, const std::vector<float>& old, std::vector<float>& avg) {
        if (v->size() != old.size()) v->assign(old.size(), 0.f);
        for (size_t i = 0; i < old.size(); i++) {
            float d = avg[i] - old[i];
            (*v)[i] = float(beta) * (*v)[i] + d;
            avg[i] = old[i] + float(lr) * (*v)[i];
        }
    };
}

inline PostAggregate fedadam(double lr = 0.01, double b1 = 0.9, double b2 = 0.99, double tau = 1e-3) {
    auto m = std::make_shared<std::vector<float>>(), v = std::make_shared<std::vector<float>>();
    return [=](uint64_t, const std::vector<float>& old, std::vector<float>& avg) {
        if (m->size() != old.size()) { m->assign(old.size(), 0.f); v->assign(old.size(), 0.f); }
        for (size_t i = 0; i < old.size(); i++) {
            float d = avg[i] - old[i];
            (*m)[i] = float(b1) * (*m)[i] + float(1 - b1) * d;
            (*v)[i] = float(b2) * (*v)[i] + float(1 - b2) * d * d;
            avg[i] = old[i] + float(lr) * (*m)[i] / (std::sqrt((*v)[i]) + float(tau));
        }
    };
}

/// Returns an empty function for plain FedAvg. Throws std::invalid_argument on a bad spec.
inline PostAggregate make_strategy(const std::string& spec) {
    std::vector<std::string> p; std::stringstream ss(spec); std::string t;
    while (std::getline(ss, t, ':')) p.push_back(t);
    if (p.empty() || p[0] == "fedavg" || p[0].empty()) return {};
    auto num = [&](size_t i, double d) { return i < p.size() ? std::stod(p[i]) : d; };
    if (p[0] == "fedavgm") return fedavgm(num(1, 0.9), num(2, 1.0));
    if (p[0] == "fedadam") return fedadam(num(1, 0.01), num(2, 0.9), num(3, 0.99), num(4, 1e-3));
    throw std::invalid_argument("unknown strategy '" + spec + "' (fedavg | fedavgm:beta[:lr] | fedadam:lr[:b1:b2:tau])");
}

/// FedProx client term: w -= step * (w - anchor), with step = lr * mu. Call after each local optimizer step,
/// `anchor` being the global vector received at the start of the round.
inline void fedprox_pull(float* w, const float* anchor, size_t n, float step) {
    for (size_t i = 0; i < n; i++) w[i] -= step * (w[i] - anchor[i]);
}

}} // namespace tf::fed
