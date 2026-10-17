// HyperparamAdvisor.hpp
//
// Chooses how to run an algorithm on THIS device: given the algorithm's declared
// tunables, the capability profile, the memory budget and the governor's limits,
// it returns the configuration with the highest predicted throughput that fits,
// says which constraint is holding it back, and (optionally) checks its best
// candidates with short real runs.
//
// The advisor is generic. It works through AlgorithmProfiler, so it knows nothing
// about CNNs or GNNs: the algorithm supplies the knobs and their ranges, the
// profiler prices a configuration, the advisor searches.
//
// Rules it follows
//   * Opt-in. Tunables that change the maths (Tunable::semantic: precision, fanout,
//     truncation length, cutoff, batch size) are only moved for algorithms that
//     opted in at start, by name, with an allowed set (empty set = the whole
//     declared range). Everything else stays at its default or at a fixed value.
//     Execution-only tunables (microbatch, checkpointing) and the thread count are
//     always free: they do not change results.
//   * "Others just slow down". If a limit cannot be met without a maths change and
//     the algorithm did not opt in, the advice is the best it can do inside the
//     limit (fewer threads, smaller microbatch) or "infeasible", plus an `unlock`
//     suggestion naming the opt-in that would help. The suggestion is never applied.
//   * Few threads when threads do not pay. Among thread counts within
//     `thread_tolerance` of the best time the smallest is chosen, so a background
//     run does not occupy cores for a gain of a few percent.
//   * Constraints are reported, not guessed: the search is repeated with each limit
//     removed, and the gain from removing it is returned (`gain_if_relaxed`).
//   * Honest error. The prediction carries the spread found by calibration; if the
//     model was never calibrated the documented analytic spread is reported instead
//     and `calibrated` is false.
//   * Deterministic: the same inputs give the same advice.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "AlgorithmProfiler.hpp"

namespace tensorf::profile {

struct Constraints {
    int max_threads = 0;                    // 0 = as many as the benchmark measured
    double max_step_seconds = 0;            // 0 = no cap; the governor sets it to keep steps short enough to yield
    double thread_tolerance = 0.03;         // prefer fewer threads when within this fraction of the best time
    bool sustained = true;                  // price for a long run (throttled), not a burst
    std::map<std::string, double> fixed;    // caller-forced values (any tunable)
};

// Semantic tunable name -> allowed values. An empty list means "the whole declared range".
using OptIn = std::map<std::string, std::vector<double>>;

struct SearchOptions {
    size_t max_grid = 100000;               // above this the search switches from exhaustive to coordinate descent
    int descent_restarts = 8;
    int descent_sweeps = 20;
    size_t keep_top = 5;
    int trial_count = 3;                    // candidates checked with real runs (needs a probe)
    int trial_repeats = 2;
    double trial_max_seconds = 5.0;         // never run a trial predicted to take longer than this
    double uncalibrated_error = 1.0;        // spread reported when the model was never calibrated (observed analytic range)
    uint64_t seed = 7;
    bool analyse_binding = true;
    bool unlock_hint = true;
};

struct Alternative {
    Config config;
    int threads = 1;
    double seconds = 0;
    double units_per_s = 0;
    uint64_t tensor_bytes = 0;
};

struct Unlock {
    Config config;
    int threads = 1;
    double units_per_s = 0;
    double gain = 1;                        // vs the advised configuration
    std::vector<std::string> changes;       // "fanout: 10 -> 5"
    std::vector<std::string> opt_in;        // tunables the algorithm would have to opt into
};

struct Advice {
    bool feasible = false;
    Config config;
    ExecPlan plan;
    ProfileResult predicted;
    bool measured = false;                  // chosen after real trials
    double measured_seconds = 0;
    double measured_units_per_s = 0;
    int trials_run = 0;
    std::string binding;                    // what limits throughput: "memory", "threads", "step_time", or the cost-model bound
    std::map<std::string, double> gain_if_relaxed;   // throughput ratio if that limit were removed
    double error = 0;
    bool calibrated = false;
    std::vector<Alternative> alternatives;
    std::optional<Unlock> unlock;
    size_t evaluated = 0;
    bool exhaustive = true;
    std::vector<std::string> notes;

    std::string to_json() const;
    std::string summary() const;
};

class HyperparamAdvisor {
public:
    explicit HyperparamAdvisor(AlgorithmProfiler& profiler) : prof_(profiler) {}

    // probe: runs ONE real step for (config, plan) and returns seconds (optional).
    // allow_trial: governor hook, checked before each real run (optional).
    Advice advise(const Constraints& c = {}, const OptIn& opt_in = {}, const SearchOptions& so = {},
                  const ProbeFn* probe = nullptr, const std::function<bool()>& allow_trial = {}) {
        Advice a;
        std::vector<std::string> notes;
        validate_inputs(c, opt_in, notes);

        Search main = run_search(c, opt_in, so, false);
        a.evaluated = main.evaluated;
        a.exhaustive = main.exhaustive;

        if (main.top.empty()) {
            fill_infeasible(a, main, c, so);
            a.notes.insert(a.notes.end(), notes.begin(), notes.end());
            if (so.unlock_hint) a.unlock = find_unlock(a, c, opt_in, so);
            return a;
        }

        a.feasible = true;
        Scored best = main.top[0];

        // ---- trials ----
        if (probe && so.trial_count > 0) {
            std::vector<AlgorithmProfiler::Sample> samples;
            struct Trial { Scored s; double measured = 0; };
            std::vector<Trial> trials;
            int considered = 0;
            bool skipped_long = false;
            for (const auto& cand : main.top) {
                if (considered >= so.trial_count) break;
                if (cand.seconds > so.trial_max_seconds) { skipped_long = true; continue; }   // never start a run predicted to be too long
                if (allow_trial && !allow_trial()) { notes.push_back("governor declined further trials"); break; }
                std::vector<double> t;
                for (int i = 0; i < std::max(1, so.trial_repeats); ++i) t.push_back((*probe)(cand.config, ExecPlan{cand.threads, c.sustained}));
                std::sort(t.begin(), t.end());
                const double med = t[t.size() / 2];
                ++considered;
                if (!(med > 0) || !std::isfinite(med)) { notes.push_back("a trial returned an unusable time and was ignored"); continue; }
                trials.push_back({cand, med});
                samples.push_back({cand.config, ExecPlan{cand.threads, c.sustained}, med});
            }
            a.trials_run = int(trials.size());
            if (skipped_long && trials.empty()) notes.push_back("no trials run: predicted steps are longer than trial_max_seconds");
            if (!trials.empty()) {
                auto fit = prof_.fit_samples(samples);
                (void)fit;
                // pick the best MEASURED candidate that respects the step cap
                const Trial* pick = nullptr;
                for (const auto& t : trials) {
                    if (c.max_step_seconds > 0 && t.measured > c.max_step_seconds) continue;
                    const double ups = t.s.units / t.measured;
                    if (!pick || ups > pick->s.units / pick->measured) pick = &t;
                }
                if (pick) {
                    best = pick->s;
                    a.measured = true;
                    a.measured_seconds = pick->measured;
                    a.measured_units_per_s = pick->s.units / pick->measured;
                } else notes.push_back("every trial exceeded the step-time cap; keeping the predicted choice");
            }
        }

        a.config = best.config;
        a.plan = ExecPlan{best.threads, c.sustained};
        a.predicted = prof_.evaluate(a.config, a.plan);
        a.calibrated = prof_.error() > 0 || prof_.scale() != 1.0;
        a.error = a.calibrated ? prof_.error() : so.uncalibrated_error;
        if (!a.calibrated) notes.push_back("prediction is uncalibrated: expect errors up to the reported spread until a real run is measured");

        for (size_t i = 0; i < main.top.size() && i < so.keep_top; ++i) {
            const auto& s = main.top[i];
            Alternative alt;
            alt.config = s.config; alt.threads = s.threads;
            alt.seconds = s.seconds;                                           // includes the calibration scale in force during the search
            alt.units_per_s = s.units_per_s; alt.tensor_bytes = s.mem;
            a.alternatives.push_back(alt);
        }

        if (so.analyse_binding) analyse_binding(a, best, main, c, opt_in, so);
        else a.binding = a.predicted.time.bound;
        if (so.unlock_hint) a.unlock = find_unlock(a, c, opt_in, so);
        a.notes.insert(a.notes.end(), notes.begin(), notes.end());
        for (const auto& n : a.predicted.notes) a.notes.push_back(n);
        return a;
    }

private:
    struct Axis {
        std::string name;
        std::vector<double> values;
        bool semantic = false;
    };
    struct Scored {
        Config config;
        int threads = 1;
        double seconds = 0, units_per_s = 0, core_seconds = 0, units = 0;
        uint64_t mem = 0;
    };
    struct Search {
        std::vector<Scored> top;            // best first
        bool have_smallest = false;
        Scored smallest;                    // smallest-memory configuration seen (for the infeasible report)
        bool any_fit = false;               // some configuration fit the memory budget
        double fastest_fitting_seconds = 0; // fastest memory-fitting candidate (for the step-cap report)
        size_t evaluated = 0;
        bool exhaustive = true;
    };

    AlgorithmProfiler& prof_;

    // ---------- inputs ----------
    void validate_inputs(const Constraints& c, const OptIn& opt_in, std::vector<std::string>& notes) const {
        const auto ts = prof_.model().tunables();
        auto find = [&](const std::string& n) -> const Tunable* { for (const auto& t : ts) if (t.name == n) return &t; return nullptr; };
        for (const auto& kv : opt_in) {
            const Tunable* t = find(kv.first);
            if (!t) { notes.push_back("opt-in for unknown tunable '" + kv.first + "' ignored"); continue; }
            if (!t->semantic) notes.push_back("'" + kv.first + "' does not change the maths; opt-in not needed");
            for (double v : kv.second)
                if (std::find(t->values.begin(), t->values.end(), v) == t->values.end())
                    notes.push_back("opt-in value " + fmt(v) + " for '" + kv.first + "' is outside the declared range and was dropped");
        }
        for (const auto& kv : c.fixed) {
            const Tunable* t = find(kv.first);
            if (!t) notes.push_back("fixed value for unknown tunable '" + kv.first + "' ignored");
            else if (std::find(t->values.begin(), t->values.end(), kv.second) == t->values.end() && kv.first != "microbatch")
                notes.push_back("fixed value " + fmt(kv.second) + " for '" + kv.first + "' is outside the declared range");
        }
    }

    static std::string fmt(double v) {
        char b[48];
        std::snprintf(b, sizeof b, "%.6g", v);
        return b;
    }

    std::vector<Axis> build_axes(const Constraints& c, const OptIn& opt_in, bool unlock_all) const {
        std::vector<Axis> axes;
        for (const auto& t : prof_.model().tunables()) {
            Axis ax;
            ax.name = t.name; ax.semantic = t.semantic;
            auto fixed = c.fixed.find(t.name);
            if (fixed != c.fixed.end()) {
                ax.values = {fixed->second};
            } else if (t.semantic) {
                auto oi = opt_in.find(t.name);
                if (unlock_all) ax.values = t.values;
                else if (oi == opt_in.end()) ax.values = {t.def};
                else if (oi->second.empty()) ax.values = t.values;
                else for (double v : t.values) if (std::find(oi->second.begin(), oi->second.end(), v) != oi->second.end()) ax.values.push_back(v);
                if (ax.values.empty()) ax.values = {t.def};
            } else {
                ax.values = t.values;
                if (std::find(ax.values.begin(), ax.values.end(), t.def) == ax.values.end()) ax.values.push_back(t.def);   // e.g. "microbatch = whole batch"
            }
            std::sort(ax.values.begin(), ax.values.end());
            ax.values.erase(std::unique(ax.values.begin(), ax.values.end()), ax.values.end());
            axes.push_back(std::move(ax));
        }
        return axes;
    }

    std::vector<int> thread_options(const Constraints& c) const {
        CostModel cm(prof_.capability());
        int mx = std::max(1, cm.device().max_threads());
        if (c.max_threads > 0) mx = std::min(mx, c.max_threads);
        std::vector<int> v;
        for (int t = 1; t < mx; t *= 2) v.push_back(t);
        v.push_back(mx);
        return v;
    }

    Config normalize(Config cfg) const {
        auto mb = cfg.find("microbatch");
        if (mb != cfg.end()) {
            const double batch = std::max(1.0, cfg_get(cfg, "batch", 1));
            mb->second = std::min(mb->second, batch);
        }
        return cfg;
    }

    static std::string key_of(const Config& c) {
        std::string k;
        for (const auto& kv : c) { k += kv.first; k += '='; k += fmt(kv.second); k += ';'; }
        return k;
    }

    // ---------- the search ----------
    struct Context {
        const Constraints& c;
        const SearchOptions& so;
        std::vector<int> threads;
        CostModel cm;
        double scale;
        uint64_t budget;
        Search* out;
        std::unordered_map<std::string, double> memo;   // key -> descent value
    };

    // Evaluates one configuration over all thread options. Returns its descent value:
    // positive throughput when feasible, negative (more negative = more memory) when not.
    double consider(Context& cx, const Config& raw) const {
        Config cfg = normalize(raw);
        const std::string key = key_of(cfg);
        auto m = cx.memo.find(key);
        if (m != cx.memo.end()) return m->second;
        double value;
        if (!prof_.model().valid(cfg)) { cx.memo[key] = -1e30; return -1e30; }
        ++cx.out->evaluated;
        StepCosts sc = prof_.model().costs(cfg);
        const uint64_t mem = sc.memory.total();
        if (!cx.out->have_smallest || mem < cx.out->smallest.mem) {
            cx.out->have_smallest = true;
            cx.out->smallest = Scored{cfg, 1, 0, 0, 0, sc.units, mem};
        }
        if (mem > cx.budget) {
            value = -1.0 - double(mem) / double(std::max<uint64_t>(1, cx.budget));
            cx.memo[key] = value;
            return value;
        }
        cx.out->any_fit = true;
        // time at every thread count (costs do not depend on threads)
        std::vector<double> secs(cx.threads.size()), core(cx.threads.size());
        double best_t = 1e300;
        for (size_t i = 0; i < cx.threads.size(); ++i) {
            auto pr = cx.cm.predict(sc, ExecPlan{cx.threads[i], cx.c.sustained});
            secs[i] = pr.seconds * cx.scale;
            core[i] = pr.core_seconds * cx.scale;
            best_t = std::min(best_t, secs[i]);
        }
        if (cx.out->fastest_fitting_seconds == 0 || best_t < cx.out->fastest_fitting_seconds) cx.out->fastest_fitting_seconds = best_t;
        if (cx.c.max_step_seconds > 0 && best_t > cx.c.max_step_seconds) {
            value = -0.5;                                                   // fits memory but too slow: better than not fitting
            cx.memo[key] = value;
            return value;
        }
        // smallest thread count within tolerance of the best (and under the cap)
        size_t pick = 0;
        for (size_t i = 0; i < cx.threads.size(); ++i)
            if (secs[i] <= best_t * (1.0 + cx.c.thread_tolerance) && (cx.c.max_step_seconds <= 0 || secs[i] <= cx.c.max_step_seconds)) { pick = i; break; }
        Scored s;
        s.config = cfg; s.threads = cx.threads[pick]; s.seconds = secs[pick]; s.core_seconds = core[pick];
        s.units = sc.units; s.units_per_s = secs[pick] > 0 ? sc.units / secs[pick] : 0; s.mem = mem;
        cx.out->top.push_back(s);
        value = s.units_per_s;
        cx.memo[key] = value;
        return value;
    }

    static bool better(const Scored& a, const Scored& b) {
        const double tol = 1e-9 * std::max(a.units_per_s, b.units_per_s);
        if (std::abs(a.units_per_s - b.units_per_s) > tol) return a.units_per_s > b.units_per_s;
        if (a.core_seconds != b.core_seconds) return a.core_seconds < b.core_seconds;
        if (a.mem != b.mem) return a.mem < b.mem;
        return key_of(a.config) < key_of(b.config);
    }

    Search run_search(const Constraints& c, const OptIn& opt_in, const SearchOptions& so, bool unlock_all) const {
        Search out;
        auto axes = build_axes(c, opt_in, unlock_all);
        Context cx{c, so, thread_options(c), CostModel(prof_.capability()), prof_.scale(), prof_.budget().tensor_budget_bytes, &out, {}};

        double grid = 1;
        for (const auto& ax : axes) grid *= double(ax.values.size());
        out.exhaustive = grid <= double(so.max_grid);

        auto make = [&](const std::vector<size_t>& idx) {
            Config cfg;
            for (size_t i = 0; i < axes.size(); ++i) cfg[axes[i].name] = axes[i].values[idx[i]];
            return cfg;
        };

        if (out.exhaustive) {
            std::vector<size_t> idx(axes.size(), 0);
            for (;;) {
                consider(cx, make(idx));
                size_t k = 0;
                while (k < axes.size()) {
                    if (++idx[k] < axes[k].values.size()) break;
                    idx[k] = 0; ++k;
                }
                if (k == axes.size()) break;
            }
        } else {
            uint64_t rng = so.seed ? so.seed : 1;
            auto next = [&] { rng ^= rng >> 12; rng ^= rng << 25; rng ^= rng >> 27; return rng * 2685821657736338717ull; };
            auto nearest_default = [&](const Axis& ax) -> size_t {
                double d = 0;
                for (const auto& t : prof_.model().tunables()) if (t.name == ax.name) d = t.def;
                size_t best = 0;
                for (size_t i = 0; i < ax.values.size(); ++i)
                    if (std::abs(ax.values[i] - d) < std::abs(ax.values[best] - d)) best = i;
                return best;
            };
            for (int r = 0; r <= so.descent_restarts; ++r) {
                std::vector<size_t> idx(axes.size());
                for (size_t i = 0; i < axes.size(); ++i) idx[i] = r == 0 ? nearest_default(axes[i]) : size_t(next() % axes[i].values.size());
                double cur = consider(cx, make(idx));
                for (int sweep = 0; sweep < so.descent_sweeps; ++sweep) {
                    bool moved = false;
                    for (size_t a = 0; a < axes.size(); ++a) {
                        size_t best_i = idx[a];
                        double best_v = cur;
                        for (size_t v = 0; v < axes[a].values.size(); ++v) {
                            if (v == idx[a]) continue;
                            auto trial = idx; trial[a] = v;
                            const double val = consider(cx, make(trial));
                            if (val > best_v) { best_v = val; best_i = v; }
                        }
                        if (best_i != idx[a]) { idx[a] = best_i; cur = best_v; moved = true; }
                    }
                    if (!moved) break;
                }
            }
        }
        std::sort(out.top.begin(), out.top.end(), better);
        // one entry per distinct config (descent can revisit through different memo paths: it cannot, but be safe)
        std::vector<Scored> uniq;
        std::set<std::string> seen;
        for (auto& s : out.top) if (seen.insert(key_of(s.config)).second) uniq.push_back(std::move(s));
        out.top = std::move(uniq);
        return out;
    }

    // ---------- reporting ----------
    void fill_infeasible(Advice& a, const Search& s, const Constraints& c, const SearchOptions&) {
        a.feasible = false;
        if (s.have_smallest) {
            a.config = s.smallest.config;
            a.plan = ExecPlan{1, c.sustained};
            a.predicted = prof_.evaluate(a.config, a.plan);
        }
        if (!s.any_fit) {
            a.binding = "memory";
            std::ostringstream o;
            o << "no configuration fits: the smallest needs " << (s.smallest.mem >> 20) << " MB of tensors, the budget is "
              << (prof_.budget().tensor_budget_bytes >> 20) << " MB";
            a.notes.push_back(o.str());
        } else if (c.max_step_seconds > 0) {
            a.binding = "step_time";
            std::ostringstream o;
            o << "no configuration meets the step-time cap of " << c.max_step_seconds << " s: the fastest that fits memory takes "
              << s.fastest_fitting_seconds << " s";
            a.notes.push_back(o.str());
        } else {
            a.binding = "constraints";
            a.notes.push_back("no configuration satisfies the constraints");
        }
        a.error = prof_.error() > 0 ? prof_.error() : 1.0;
    }

    double best_units(const Constraints& c, const OptIn& opt_in, const SearchOptions& so, bool unlock_all) const {
        Search s = run_search(c, opt_in, so, unlock_all);
        return s.top.empty() ? 0.0 : s.top[0].units_per_s;
    }

    void analyse_binding(Advice& a, const Scored& best, const Search&, const Constraints& c, const OptIn& opt_in, const SearchOptions& so) {
        const double base = std::max(1e-300, a.measured ? a.measured_units_per_s : best.units_per_s);
        const double pred_base = std::max(1e-300, best.units_per_s);
        // memory
        {
            AlgorithmProfiler& p = prof_;
            MemoryBudget saved = p.budget();
            MemoryBudget open = saved;
            open.tensor_budget_bytes = ~0ull >> 2;
            p.set_budget(open);
            const double g = best_units(c, opt_in, so, false) / pred_base;
            p.set_budget(saved);
            a.gain_if_relaxed["memory"] = std::max(1.0, g);
        }
        if (c.max_step_seconds > 0) {
            Constraints r = c; r.max_step_seconds = 0;
            a.gain_if_relaxed["step_time"] = std::max(1.0, best_units(r, opt_in, so, false) / pred_base);
        }
        if (c.max_threads > 0) {
            Constraints r = c; r.max_threads = 0;
            a.gain_if_relaxed["threads"] = std::max(1.0, best_units(r, opt_in, so, false) / pred_base);
        }
        (void)base;
        std::string bind = a.predicted.time.bound;
        double top = 1.05;                               // a limit only "binds" if removing it gains more than 5%
        for (const auto& kv : a.gain_if_relaxed) if (kv.second > top) { top = kv.second; bind = kv.first; }
        a.binding = bind;
    }

    std::optional<Unlock> find_unlock(const Advice& a, const Constraints& c, const OptIn& opt_in, const SearchOptions& so) {
        // Only worth looking if some semantic tunable was held fixed.
        std::vector<std::string> held;
        for (const auto& t : prof_.model().tunables())
            if (t.semantic && !c.fixed.count(t.name) && !opt_in.count(t.name) && t.values.size() > 1) held.push_back(t.name);
        if (held.empty()) return std::nullopt;
        Search s = run_search(c, opt_in, so, true);
        if (s.top.empty()) return std::nullopt;
        const Scored& u = s.top[0];
        const double base = a.feasible ? (a.predicted.time.units_per_s > 0 ? a.predicted.time.units_per_s : 1e-300) : 0.0;
        const double gain = a.feasible ? u.units_per_s / base : std::numeric_limits<double>::infinity();
        if (a.feasible && gain < 1.10) return std::nullopt;       // not worth mentioning
        Unlock un;
        un.config = u.config; un.threads = u.threads; un.units_per_s = u.units_per_s; un.gain = gain;
        for (const auto& kv : u.config) {
            const double from = cfg_get(a.config, kv.first, kv.second);
            if (a.config.empty() || std::abs(from - kv.second) > 1e-12) {
                un.changes.push_back(kv.first + ": " + (a.config.empty() ? std::string("?") : fmt(from)) + " -> " + fmt(kv.second));
                for (const auto& h : held) if (h == kv.first) un.opt_in.push_back(h);
            }
        }
        if (un.opt_in.empty()) return std::nullopt;               // the improvement came from free tunables only
        return un;
    }
};

// ───────────────────────────── output ─────────────────────────────

inline std::string Advice::to_json() const {
    detail::JsonOut j;
    j.begin_obj();
    j.kv("feasible", feasible);
    j.key("config"); j.begin_obj(); for (const auto& kv : config) j.kv(kv.first, kv.second); j.end_obj();
    j.kv("threads", plan.threads); j.kv("sustained", plan.sustained);
    j.kv("units_per_s", predicted.time.units_per_s); j.kv("step_seconds", predicted.time.seconds);
    j.kv("bound", predicted.time.bound);
    j.kv("tensor_bytes", uint64_t(predicted.costs.memory.total()));
    j.kv("measured", measured); j.kv("measured_seconds", measured_seconds); j.kv("measured_units_per_s", measured_units_per_s);
    j.kv("trials_run", trials_run);
    j.kv("binding", binding);
    j.key("gain_if_relaxed"); j.begin_obj(); for (const auto& kv : gain_if_relaxed) j.kv(kv.first, kv.second); j.end_obj();
    j.kv("error", error); j.kv("calibrated", calibrated);
    j.kv("evaluated", uint64_t(evaluated)); j.kv("exhaustive", exhaustive);
    j.key("alternatives"); j.begin_arr();
    for (const auto& alt : alternatives) {
        j.begin_obj();
        j.key("config"); j.begin_obj(); for (const auto& kv : alt.config) j.kv(kv.first, kv.second); j.end_obj();
        j.kv("threads", alt.threads); j.kv("seconds", alt.seconds); j.kv("units_per_s", alt.units_per_s);
        j.kv("tensor_bytes", uint64_t(alt.tensor_bytes));
        j.end_obj();
    }
    j.end_arr();
    j.key("unlock");
    if (!unlock) j.null();
    else {
        j.begin_obj();
        j.kv("gain", std::isfinite(unlock->gain) ? unlock->gain : -1.0); j.kv("units_per_s", unlock->units_per_s);
        j.key("changes"); j.begin_arr(); for (const auto& s : unlock->changes) j.str(s); j.end_arr();
        j.key("opt_in"); j.begin_arr(); for (const auto& s : unlock->opt_in) j.str(s); j.end_arr();
        j.end_obj();
    }
    j.key("notes"); j.begin_arr(); for (const auto& n : notes) j.str(n); j.end_arr();
    j.end_obj();
    return j.str();
}

inline std::string Advice::summary() const {
    std::ostringstream o;
    o << (feasible ? "advice" : "NO FEASIBLE CONFIGURATION (best effort shown)") << ": ";
    for (const auto& kv : config) o << kv.first << '=' << kv.second << ' ';
    o << "threads=" << plan.threads << '\n';
    o << "  predicted " << predicted.time.units_per_s << " units/s, step " << predicted.time.seconds * 1e3 << " ms, bound " << predicted.time.bound
      << ", tensors " << (predicted.costs.memory.total() >> 20) << " MB, error +/-" << error * 100 << "%" << (calibrated ? "" : " (uncalibrated)") << '\n';
    if (measured) o << "  measured " << measured_units_per_s << " units/s over " << trials_run << " trial(s)\n";
    o << "  limited by: " << binding;
    for (const auto& kv : gain_if_relaxed) if (kv.second > 1.05) o << "  [removing " << kv.first << " gains x" << kv.second << "]";
    o << '\n';
    if (unlock) {
        o << "  suggestion (needs opt-in): ";
        for (const auto& s : unlock->changes) o << s << "; ";
        o << "gain x" << unlock->gain << '\n';
    }
    for (const auto& n : notes) o << "  note: " << n << '\n';
    return o.str();
}

}  // namespace tensorf::profile
