#pragma once

/*
 * Advisor.hpp
 * ===========
 * The per-algorithm advisor (design doc, section 4). It does NOT choose a model: it searches the
 * tunables an Algorithm declares (micro-batch, threads, checkpointing, precision ...) to maximise
 * predicted throughput inside the constraints, then optionally confirms the best few with a short
 * measured trial.
 *
 *   inputs    Algorithm (factory + tunables), DeviceFingerprint, CapabilityVector, optional
 *             Calibration, Constraints (memory share, thread cap, step-time ceiling ...)
 *   search    exhaustive over the discrete space when it is small, deterministic coordinate ascent
 *             when it is large. Result-preserving tunables are searched first and alone; the
 *             result-changing ones (batch size, precision ...) are searched only if the caller
 *             allows it, and are otherwise held at the caller's base config.
 *   trial     if a trial function is supplied, the top few candidates are run for real and re-ranked
 *             by MEASURED throughput; the predicted-vs-measured error is reported.
 *   output    a config; predicted step time, throughput and memory with an error range; which
 *             constraint is binding (found by relaxing each constraint and re-searching); the
 *             runners-up; and what the model could not vouch for.
 *
 * What the advice is worth is stated in the result, not assumed:
 *   - rel_error comes from the model's calibration (0.60 if uncalibrated) or from the trials, and
 *     the throughput range is derived from it.
 *   - layer kinds that were never measured against real ops are listed in `unvalidated_kinds`.
 *   - if the algorithm declares no "threads" tunable the prediction is single-threaded. The
 *     framework has no thread pool yet, so multi-thread speed-ups are never assumed.
 *   - a candidate whose estimated memory exceeds the allowed share of the device's SAFE budget
 *     (available RAM, cgroup limit, kernel reserve; never total RAM) is infeasible, not merely
 *     penalised: running out of memory on a phone kills the app.
 *
 * Conventions an Algorithm must follow:
 *   - tunable named "threads" is the thread count the advisor passes to predict(); all other
 *     tunables are passed to the factory untouched.
 *   - the factory may throw for an invalid combination; that candidate is infeasible.
 *   - tunable values are listed ascending; the first value is the default for any tunable the
 *     caller's base config does not mention.
 */

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "WorkloadModel.hpp"

namespace tensorf::prof::adv {

using wl::Algorithm;
using wl::Config;
using wl::Prediction;
using wl::Tunable;

// ─── Inputs and outputs ──────────────────────────────────────────────────────

struct Constraints {
    int    max_threads      = 0;       // 0 = no cap beyond the tunable's own values
    double memory_fraction  = 0.6;     // share of the device's safe memory budget this job may use (the governor's cap)
    double max_step_s       = 0;       // 0 = none; a ceiling on predicted seconds per full step
    double min_units_per_s  = 0;       // 0 = none; a floor on predicted throughput
    bool   allow_altering   = false;   // may result-changing tunables (batch size, precision ...) be searched?
};

struct AdvisorOptions {
    size_t max_candidates   = 20000;   // above this the space is searched by coordinate ascent
    int    max_passes       = 8;       // coordinate-ascent sweeps
    int    top_k            = 5;       // runners-up reported
    int    trial_top        = 3;       // candidates confirmed by a measured trial
    double trial_budget_s   = 10.0;    // wall clock for all trials together
    double mem_overhead_factor = 1.25; // measured RSS / tensor bytes (TensorAccounting) when available
    double gain_threshold   = 0.02;    // a relaxed constraint "binds" only if lifting it gains more than this
    double error_floor      = 0.0;     // the advised range is never tighter than this (relative): the device's measured run-to-run noise
};

// Runs ONE step of the algorithm under `cfg` and returns the measured seconds, or <= 0 / NaN on failure.
using TrialFn = std::function<double(const Config&)>;

struct Candidate {
    Config      cfg;
    Prediction  pred;
    bool        feasible = false;
    std::string why;                   // reason when infeasible
    double      score = -1e18;         // throughput when feasible; negative, ordered by closeness to feasible, otherwise
};

struct TrialResult {
    Config cfg;
    double predicted_s = 0, measured_s = 0;
    bool   ok = false;
    std::string note;
};

struct Advice {
    bool        ok = false;
    Config      config;
    Prediction  prediction;                // for the chosen config
    double      units_per_s = 0;           // central estimate: measured if a trial confirmed it, else predicted
    double      units_per_s_low = 0, units_per_s_high = 0;
    double      rel_error = wl::Calibration::kUncalibratedError;   // what the range is built from
    std::string error_basis = "model";     // "model" | "trials"
    std::string binding;                   // which constraint limits throughput (or "hardware: <bound>")
    double      binding_gain = 0;          // throughput gained by lifting it, as a fraction
    std::vector<Candidate> top;            // best feasible candidates, best first (includes the chosen model-best)
    std::vector<TrialResult> trials;
    bool        confirmed = false;         // a measured trial backs the choice
    bool        ranking_changed = false;   // measurement overturned the model's first choice
    double      trial_error = std::numeric_limits<double>::quiet_NaN();   // median |predicted - measured| / measured
    std::vector<wl::Observation> trial_observations;   // feed to Calibration::fit to improve the next prediction
    size_t      evaluated = 0, infeasible_memory = 0;
    std::string search_mode;               // "exhaustive" | "coordinate ascent"
    std::vector<std::string> unvalidated_kinds;
    std::vector<std::string> notes;
};

// ─── Search machinery ────────────────────────────────────────────────────────

namespace detail {

constexpr double kNeverFeasible = -1e18;

struct CfgLess {
    bool operator()(const Config& a, const Config& b) const { return a < b; }
};

inline int threads_of(const Config& c) {
    auto it = c.find("threads");
    return it == c.end() ? 1 : std::max(1, (int)std::llround(it->second));
}

// Strictly better than: higher score; ties go to less memory, then fewer threads, then a stable config order,
// so the choice never depends on enumeration order.
inline bool better(const Candidate& a, const Candidate& b) {
    const double tol = 1e-12 * std::max(1.0, std::max(std::fabs(a.score), std::fabs(b.score)));
    if (std::fabs(a.score - b.score) > tol) return a.score > b.score;
    if (a.pred.memory.process_total != b.pred.memory.process_total) return a.pred.memory.process_total < b.pred.memory.process_total;
    if (threads_of(a.cfg) != threads_of(b.cfg)) return threads_of(a.cfg) < threads_of(b.cfg);
    return a.cfg < b.cfg;
}

struct Searcher {
    const Algorithm& alg;
    const DeviceFingerprint& fp;
    const CapabilityVector& cv;
    const wl::Calibration* cal;
    Constraints cons;
    AdvisorOptions opt;
    bool ignore_memory = false, ignore_thread_cap = false, ignore_step_limit = false;

    Config base;
    std::vector<const Tunable*> dims;
    std::vector<std::vector<double>> dim_values;

    size_t evaluated = 0, infeasible_memory = 0;
    std::vector<Candidate> keep;                    // best feasible so far, sorted, bounded
    size_t keep_n = 8;
    double min_mem_seen = std::numeric_limits<double>::infinity();   // diagnostics when nothing fits
    double min_mem_budget = 0;
    std::map<Config, double, CfgLess> cache;        // config -> score, so a revisit costs nothing
    bool used_coordinate = false;

    Searcher(const Algorithm& a, const DeviceFingerprint& f, const CapabilityVector& c, const wl::Calibration* k,
             const Constraints& cs, const AdvisorOptions& o)
        : alg(a), fp(f), cv(c), cal(k), cons(cs), opt(o) {}

    std::vector<double> values_for(const Tunable& t) const {
        std::vector<double> v = t.values;
        if (t.name == "threads" && cons.max_threads > 0 && !ignore_thread_cap) {
            std::vector<double> f;
            for (double x : v) if (x <= cons.max_threads) f.push_back(x);
            if (f.empty() && !v.empty()) f.push_back(*std::min_element(v.begin(), v.end()));   // never leave a dimension empty
            v = f;
        }
        return v;
    }

    void prepare(const Config& caller_base, std::vector<std::string>* notes) {
        base = caller_base;
        dims.clear(); dim_values.clear();
        for (const auto& t : alg.tunables) {
            if (t.values.empty()) continue;
            if (!base.count(t.name)) {
                base[t.name] = t.values.front();
                if (notes) notes->push_back("base config has no value for '" + t.name + "': using its first candidate " + std::to_string(t.values.front()));
            }
            if (t.preserves_semantics || cons.allow_altering) {
                auto v = values_for(t);
                if (!v.empty()) { dims.push_back(&t); dim_values.push_back(std::move(v)); }
            }
        }
    }

    Candidate eval(const Config& cfg) {
        Candidate c; c.cfg = cfg;
        ++evaluated;
        const int threads = threads_of(cfg);
        if (!ignore_thread_cap && cons.max_threads > 0 && threads > cons.max_threads) {
            c.why = "threads " + std::to_string(threads) + " over the cap of " + std::to_string(cons.max_threads); c.score = kNeverFeasible; return c;
        }
        wl::Workload w;
        try { w = alg.build(cfg); }
        catch (const std::exception& e) { c.why = std::string("factory rejected the config: ") + e.what(); c.score = kNeverFeasible; return c; }
        catch (...) { c.why = "factory rejected the config"; c.score = kNeverFeasible; return c; }

        wl::PredictOptions po;
        po.threads = threads;
        po.memory_fraction = cons.memory_fraction;
        po.mem_overhead_factor = opt.mem_overhead_factor;
        c.pred = wl::predict(w, fp, cv, po, cal);
        if (!c.pred.valid) { c.why = "no prediction (capability vector incomplete or workload empty)"; c.score = kNeverFeasible; return c; }

        if (c.pred.memory.process_total < min_mem_seen) { min_mem_seen = c.pred.memory.process_total; min_mem_budget = c.pred.memory_budget; }
        if (!ignore_memory && !c.pred.memory_fits) {
            ++infeasible_memory;
            c.why = "memory";
            c.score = c.pred.memory_budget > 0 ? -(c.pred.memory.process_total / c.pred.memory_budget) : kNeverFeasible;   // <= -1: nearer to fitting is better
            return c;
        }
        if (!ignore_step_limit && cons.max_step_s > 0 && c.pred.step_s > cons.max_step_s) {
            c.why = "step time"; c.score = -(c.pred.step_s / cons.max_step_s); return c;
        }
        if (cons.min_units_per_s > 0 && c.pred.units_per_s < cons.min_units_per_s) {
            c.why = "throughput floor"; c.score = -(cons.min_units_per_s / std::max(1e-12, c.pred.units_per_s)); return c;
        }
        c.feasible = true;
        c.score = c.pred.units_per_s;
        return c;
    }

    void consider(const Candidate& c) {
        if (!c.feasible) return;
        auto it = keep.begin();
        while (it != keep.end() && !better(c, *it)) ++it;
        keep.insert(it, c);
        if (keep.size() > keep_n) keep.pop_back();
    }

    double score_of(const Config& cfg) {
        auto it = cache.find(cfg);
        if (it != cache.end()) return it->second;
        Candidate c = eval(cfg);
        consider(c);
        cache[cfg] = c.score;
        return c.score;
    }

    static size_t saturating_product(const std::vector<std::vector<double>>& v) {
        size_t n = 1;
        for (const auto& x : v) {
            if (x.empty()) continue;
            if (n > std::numeric_limits<size_t>::max() / x.size()) return std::numeric_limits<size_t>::max();
            n *= x.size();
        }
        return n;
    }

    // Returns the best feasible candidate, or none (keep stays empty).
    void run() {
        const size_t space = saturating_product(dim_values);
        if (dims.empty()) { score_of(base); return; }
        if (space <= opt.max_candidates) {
            used_coordinate = false;
            std::vector<size_t> idx(dims.size(), 0);
            for (;;) {
                Config cfg = base;
                for (size_t d = 0; d < dims.size(); ++d) cfg[dims[d]->name] = dim_values[d][idx[d]];
                score_of(cfg);
                size_t d = 0;
                while (d < dims.size() && ++idx[d] == dim_values[d].size()) idx[d++] = 0;
                if (d == dims.size()) break;
            }
            return;
        }
        used_coordinate = true;
        Config cur = base;
        for (size_t d = 0; d < dims.size(); ++d) {                   // snap the start onto each dimension's grid
            double want = base.at(dims[d]->name), best = dim_values[d].front();
            for (double v : dim_values[d]) if (std::fabs(v - want) < std::fabs(best - want)) best = v;
            cur[dims[d]->name] = best;
        }
        double cur_score = score_of(cur);
        for (int pass = 0; pass < opt.max_passes; ++pass) {
            bool moved = false;
            for (size_t d = 0; d < dims.size(); ++d) {
                Config best_cfg = cur; double best_score = cur_score;
                for (double v : dim_values[d]) {
                    Config t = cur; t[dims[d]->name] = v;
                    const double s = score_of(t);
                    if (s > best_score + 1e-12 * std::max(1.0, std::fabs(best_score))) { best_score = s; best_cfg = t; }
                }
                if (best_cfg != cur) { cur = best_cfg; cur_score = best_score; moved = true; }
            }
            if (!moved) break;
        }
    }

    bool found() const { return !keep.empty(); }
    const Candidate& best() const { return keep.front(); }
};

}  // namespace detail

// ─── The advisor ─────────────────────────────────────────────────────────────

inline Advice advise(const Algorithm& alg, const Config& base_config, const DeviceFingerprint& fp, const CapabilityVector& cv,
                     const wl::Calibration* cal = nullptr, const Constraints& cons = {}, const AdvisorOptions& opt = {},
                     const TrialFn& trial = nullptr) {
    Advice adv;
    if (!alg.build) { adv.notes.push_back("algorithm has no factory"); return adv; }

    detail::Searcher s(alg, fp, cv, cal, cons, opt);
    s.keep_n = (size_t)std::max({opt.top_k, opt.trial_top, 1}) + 3;
    s.prepare(base_config, &adv.notes);
    s.run();
    adv.evaluated = s.evaluated;
    adv.infeasible_memory = s.infeasible_memory;
    adv.search_mode = s.used_coordinate ? "coordinate ascent" : "exhaustive";

    bool has_threads_tunable = false;
    for (const auto& t : alg.tunables) if (t.name == "threads") has_threads_tunable = true;
    if (!has_threads_tunable)
        adv.notes.push_back("no 'threads' tunable declared: predicted single-threaded (the framework has no thread pool yet)");

    // ── nothing feasible: name the constraint that is responsible by lifting each in turn ──
    if (!s.found()) {
        auto feasible_when = [&](auto&& mutate) {
            detail::Searcher r = s;
            r.keep.clear(); r.cache.clear(); r.evaluated = 0; r.infeasible_memory = 0;
            mutate(r);
            r.run();
            return r.found();
        };
        if (s.infeasible_memory > 0 && feasible_when([](detail::Searcher& r) { r.ignore_memory = true; })) {
            adv.binding = "memory";
            adv.notes.push_back("no configuration fits: the smallest needs about " + std::to_string((long long)(s.min_mem_seen / 1048576.0)) +
                                " MB but the allowed budget is " + std::to_string((long long)(s.min_mem_budget / 1048576.0)) +
                                " MB (" + std::to_string((int)std::lround(cons.memory_fraction * 100)) + "% of the device's safe memory)");
        } else if (cons.max_step_s > 0 && feasible_when([](detail::Searcher& r) { r.ignore_step_limit = true; })) {
            adv.binding = "step time";
            adv.notes.push_back("no configuration meets the step-time ceiling of " + std::to_string(cons.max_step_s) + " s");
        } else if (cons.min_units_per_s > 0 && feasible_when([](detail::Searcher& r) { r.cons.min_units_per_s = 0; })) {
            adv.binding = "throughput floor";
            adv.notes.push_back("no configuration reaches the required " + std::to_string(cons.min_units_per_s) + " units/s");
        } else {
            adv.binding = "no feasible configuration";
            adv.notes.push_back("no configuration is feasible (factory rejections, or no usable capability data)");
        }
        return adv;
    }

    // ── ranked list from the model ──
    for (const auto& c : s.keep) { if ((int)adv.top.size() < opt.top_k) adv.top.push_back(c); }
    Candidate chosen = s.best();
    adv.ok = true;

    // ── binding constraint: lift each in turn and see what the search gains ──
    struct Relax { const char* name; double gain; };
    std::vector<Relax> relax;
    const double base_ups = chosen.pred.units_per_s;
    auto gain_when = [&](auto&& mutate) -> double {
        detail::Searcher r = s;
        r.keep.clear(); r.cache.clear(); r.evaluated = 0; r.infeasible_memory = 0;
        mutate(r);
        r.run();
        return r.found() ? r.best().pred.units_per_s / base_ups - 1.0 : 0.0;
    };
    if (s.infeasible_memory > 0 || cons.memory_fraction < 1.0)
        relax.push_back({"memory", gain_when([](detail::Searcher& r) { r.ignore_memory = true; })});
    if (cons.max_threads > 0 && has_threads_tunable)
        relax.push_back({"thread cap", gain_when([](detail::Searcher& r) {
            r.ignore_thread_cap = true;
            r.prepare(r.base, nullptr);                                   // re-derive the dimensions with the cap lifted
        })});
    if (cons.max_step_s > 0)
        relax.push_back({"step time", gain_when([](detail::Searcher& r) { r.ignore_step_limit = true; })});
    if (!cons.allow_altering) {
        bool any_altering = false; for (const auto& t : alg.tunables) if (!t.preserves_semantics) any_altering = true;
        if (any_altering) {
            const double g = gain_when([](detail::Searcher& r) { r.cons.allow_altering = true; r.prepare(r.base, nullptr); });
            if (g > opt.gain_threshold)
                adv.notes.push_back("result-changing tunables could raise throughput by about +" + std::to_string((int)std::lround(g * 100)) +
                                    "% (not searched: allow_altering is off)");
        }
    }
    const Relax* top_relax = nullptr;
    for (const auto& r : relax) if (r.gain > opt.gain_threshold && (!top_relax || r.gain > top_relax->gain)) top_relax = &r;
    if (top_relax) { adv.binding = top_relax->name; adv.binding_gain = top_relax->gain; }
    else adv.binding = std::string("hardware: ") + wl::to_string(chosen.pred.dominant) + "-bound";

    // ── optional measured confirmation of the best few ──
    adv.unvalidated_kinds.clear();
    if (trial && opt.trial_budget_s > 0 && opt.trial_top > 0) {
        const Deadline dl(opt.trial_budget_s);
        const size_t n = std::min<size_t>((size_t)opt.trial_top, s.keep.size());
        std::vector<std::pair<double, size_t>> measured;                 // (units/s measured, index into s.keep)
        for (size_t i = 0; i < n && !dl.expired(); ++i) {
            const Candidate& c = s.keep[i];
            TrialResult tr; tr.cfg = c.cfg; tr.predicted_s = c.pred.step_s;
            try {
                const double m = trial(c.cfg);
                if (std::isfinite(m) && m > 0) { tr.measured_s = m; tr.ok = true; }
                else tr.note = "trial returned no usable time";
            } catch (const std::exception& e) { tr.note = std::string("trial threw: ") + e.what(); }
            catch (...) { tr.note = "trial threw"; }
            if (tr.ok) {
                const double units = c.pred.units_per_s * c.pred.step_s;   // units per full step
                measured.emplace_back(units / tr.measured_s, i);
                adv.trial_observations.push_back({wl::kOpKinds, tr.predicted_s, tr.measured_s});
            }
            adv.trials.push_back(std::move(tr));
        }
        if (!measured.empty()) {
            size_t best_i = 0;
            for (size_t k = 1; k < measured.size(); ++k) if (measured[k].first > measured[best_i].first) best_i = k;
            const size_t chosen_idx = measured[best_i].second;
            adv.ranking_changed = chosen_idx != 0;
            chosen = s.keep[chosen_idx];
            adv.confirmed = true;
            adv.units_per_s = measured[best_i].first;
            std::vector<double> errs;
            for (const auto& t : adv.trials) if (t.ok) errs.push_back(std::fabs(t.predicted_s - t.measured_s) / t.measured_s);
            adv.trial_error = median(errs);
            if (errs.size() >= 2) { adv.rel_error = std::max(adv.trial_error, 0.05); adv.error_basis = "trials"; }
        } else {
            adv.notes.push_back("every trial failed: the choice rests on the model alone");
        }
    } else if (trial) {
        adv.notes.push_back("trial budget is zero: the choice rests on the model alone");
    }

    adv.config = chosen.cfg;
    adv.prediction = chosen.pred;
    if (!adv.confirmed) { adv.units_per_s = chosen.pred.units_per_s; adv.rel_error = chosen.pred.rel_error; adv.error_basis = "model"; }
    // A range tighter than the machine's own repeatability would be a claim nobody can back: floor it at the noise.
    if (opt.error_floor > adv.rel_error) {
        adv.notes.push_back("error range widened from " + std::to_string((int)std::lround(adv.rel_error * 100)) + "% to " +
                            std::to_string((int)std::lround(opt.error_floor * 100)) + "%: the run-to-run noise measured on this device");
        adv.rel_error = opt.error_floor;
        adv.error_basis += " (floored at measured noise)";
    }
    adv.units_per_s_low  = adv.units_per_s / (1.0 + adv.rel_error);
    adv.units_per_s_high = adv.units_per_s * (1.0 + adv.rel_error);

    // what the model could not vouch for
    for (const auto& lp : chosen.pred.layers) {
        const bool known = cal && (cal->calibrated((int)lp.kind));
        if (!known) {
            const std::string k = wl::to_string(lp.kind);
            if (std::find(adv.unvalidated_kinds.begin(), adv.unvalidated_kinds.end(), k) == adv.unvalidated_kinds.end()) adv.unvalidated_kinds.push_back(k);
        }
    }
    if (!cal) adv.notes.push_back("model is uncalibrated: no measurement backs the time prediction (error range " +
                                  std::to_string((int)std::lround(wl::Calibration::kUncalibratedError * 100)) + "%)");
    else if (!adv.unvalidated_kinds.empty()) {
        std::string list; for (const auto& k : adv.unvalidated_kinds) list += (list.empty() ? "" : ", ") + k;
        adv.notes.push_back("layer kinds never measured against real ops (global fallback factor, higher error): " + list);
    }
    for (const auto& n : chosen.pred.notes) adv.notes.push_back(n);

    // are the model's top picks even distinguishable at this error?
    if (adv.top.size() >= 2 && adv.top[0].pred.units_per_s < adv.top[1].pred.units_per_s * (1.0 + adv.rel_error) && !adv.confirmed)
        adv.notes.push_back("the top candidates differ by less than the model's error: the ranking among them is not reliable (supply a trial to settle it)");
    return adv;
}

// ─── JSON / text ─────────────────────────────────────────────────────────────

inline void to_json(Json& j, const Config& c, const char* key) {
    j.begin_object(key);
    for (const auto& kv : c) j.num(kv.first.c_str(), kv.second);
    j.end_object();
}

inline void to_json(Json& j, const Advice& a, const char* key = "advice") {
    j.begin_object(key);
    j.boolean("ok", a.ok).boolean("confirmed", a.confirmed).boolean("ranking_changed", a.ranking_changed)
     .str("binding_constraint", a.binding).num("binding_gain", a.binding_gain)
     .num("units_per_s", a.units_per_s).num("units_per_s_low", a.units_per_s_low).num("units_per_s_high", a.units_per_s_high)
     .num("rel_error", a.rel_error).str("error_basis", a.error_basis).num("trial_error", a.trial_error)
     .uinteger("evaluated", a.evaluated).uinteger("infeasible_memory", a.infeasible_memory).str("search_mode", a.search_mode);
    to_json(j, a.config, "config");
    if (a.ok) wl::to_json(j, a.prediction, "prediction");
    j.begin_array("runners_up");
    for (const auto& c : a.top) {
        j.begin_object().num("units_per_s", c.pred.units_per_s).num("step_s", c.pred.step_s).num("memory_bytes", c.pred.memory.process_total);
        to_json(j, c.cfg, "config");
        j.end_object();
    }
    j.end_array();
    j.begin_array("trials");
    for (const auto& t : a.trials) {
        j.begin_object().boolean("ok", t.ok).num("predicted_s", t.predicted_s).num("measured_s", t.measured_s).str("note", t.note);
        to_json(j, t.cfg, "config");
        j.end_object();
    }
    j.end_array();
    j.begin_array("unvalidated_kinds");
    for (const auto& k : a.unvalidated_kinds) { j.begin_object().str("kind", k).end_object(); }
    j.end_array();
    j.begin_array("notes");
    for (const auto& n : a.notes) { j.begin_object().str("note", n).end_object(); }
    j.end_array();
    j.end_object();
}

inline void print_advice(const Advice& a) {
    if (!a.ok) {
        std::printf("\nAdvice: none (%s)\n", a.binding.c_str());
        for (const auto& n : a.notes) std::printf("  - %s\n", n.c_str());
        return;
    }
    std::printf("\nAdvice  (%s search, %zu configs evaluated%s)\n", a.search_mode.c_str(), a.evaluated, a.confirmed ? ", confirmed by trial" : "");
    std::printf("  config   :");
    for (const auto& kv : a.config) std::printf(" %s=%g", kv.first.c_str(), kv.second);
    std::printf("\n  speed    : %.1f /s  (range %.1f - %.1f, error basis: %s %.0f%%)\n", a.units_per_s, a.units_per_s_low, a.units_per_s_high,
                a.error_basis.c_str(), a.rel_error * 100);
    std::printf("  memory   : %.1f MB estimated (%s)\n", a.prediction.memory.process_total / 1048576.0, a.prediction.memory_fits ? "fits" : "DOES NOT FIT");
    std::printf("  limited by: %s", a.binding.c_str());
    if (a.binding_gain > 0) std::printf("  (lifting it: +%.0f%%)", a.binding_gain * 100);
    std::printf("\n");
    for (const auto& n : a.notes) std::printf("  - %s\n", n.c_str());
}

}  // namespace tensorf::prof::adv
