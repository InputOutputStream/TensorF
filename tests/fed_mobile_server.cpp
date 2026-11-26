// fed_mobile_server — algorithm-agnostic federated coordinator.
// The server never sees data or training code: it needs the task's parameter layout
// (from the registry) and a server-side strategy. Any registered algorithm works.
//   ./fed_mobile_server --algo gcn --opt data=Datasets/cora --min-clients 3 --rounds 20 --strategy fedavgm:0.9
//   ./fed_mobile_server --algo synthetic --params 100000 --min-clients 2
//   ./fed_mobile_server --list
// Prints "@@TFEVT {json}" events that the gateway relays to the dashboard.
#include "net/Mobile/BuiltinTasks.hpp"
#include "net/Mobile/FedStrategies.hpp"
#include <csignal>
#include <cstdlib>
#include <cstring>
using namespace tf::fed;
static std::atomic<bool> g_stop{false};
int main(int argc, char** argv) {
    std::signal(SIGINT, [](int) { g_stop = true; }); std::signal(SIGTERM, [](int) { g_stop = true; });
    register_builtin_tasks();
    CoordinatorConfig c; c.min_clients = 2;
    std::string algo = "synthetic", strategy = "fedavg"; TaskOptions opts; bool dims_only = false;
    auto arg = [&](int& i) { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "missing value for %s\n", argv[i]), std::exit(2), nullptr); };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list") { for (auto& kv : TaskRegistry::get().all()) std::printf("%-10s %s\n", kv.first.c_str(), kv.second.help.c_str()); return 0; }
        else if (a == "--algo") algo = arg(i);
        else if (a == "--dims") dims_only = true;
        else if (a == "--opt") { if (!opts.parse(arg(i))) { std::fprintf(stderr, "--opt expects key=value\n"); return 2; } }
        else if (a == "--params") opts.kv["params"] = arg(i);
        else if (a == "--strategy") strategy = arg(i);
        else if (a == "--weighting") c.weighting = std::string(arg(i)) == "uniform" ? CoordinatorConfig::Weighting::Uniform : CoordinatorConfig::Weighting::Samples;
        else if (a == "--port") c.port = uint16_t(std::atoi(arg(i)));
        else if (a == "--bind") c.bind_addr = arg(i);
        else if (a == "--min-clients") c.min_clients = std::strtoul(arg(i), nullptr, 10);
        else if (a == "--max-clients") c.max_clients = std::strtoul(arg(i), nullptr, 10);
        else if (a == "--rounds") c.max_rounds = std::strtoull(arg(i), nullptr, 10);
        else if (a == "--token") c.auth_token = arg(i);
        else if (a == "--grace-ms") c.grace_ms = std::atoi(arg(i));
        else if (a == "--enroll-min") c.enroll_min = std::strtoul(arg(i), nullptr, 10);
        else if (a == "--enroll-grace-ms") c.enroll_grace_ms = std::atoi(arg(i));
        else if (a == "--deadline-ms") c.upload_deadline_ms = std::atoi(arg(i));
        else if (a == "--fp16-down") c.fp16_download = true;
        else if (a == "--no-fp16-up") c.allow_fp16_upload = false;
        else if (a == "--staleness-decay") c.staleness_decay = std::atof(arg(i));
        else { std::fprintf(stderr, "unknown option %s\n(--algo --dims --opt k=v --params --strategy --weighting --port --bind --min-clients --max-clients --rounds --token --grace-ms --enroll-min --enroll-grace-ms --deadline-ms --fp16-down --no-fp16-up --staleness-decay --list)\n", a.c_str()); return 2; }
    }
    if (dims_only) {   // print the layout parameters a shard-mode client needs: features / classes
        try { size_t F = 0, C = 0; tf::fed::graph::dataset_dims(opts, F, C); std::printf("features=%zu classes=%zu\n", F, C); return 0; }
        catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    }
    try {
        ServerTaskInfo info = TaskRegistry::get().find(algo).server(opts);
        configure(c, info);
        c.post_aggregate = make_strategy(strategy);
        FedCoordinator* sp = nullptr;
        if (info.eval) c.on_round = [&](const RoundReport& r, const std::vector<float>& g) { std::string j = info.eval(g, r.round); if (!j.empty() && sp) sp->publish_event(j); };
        FedCoordinator srv(c); sp = &srv;
        if (!info.initial.empty() && !srv.set_initial_global(info.initial)) { std::fprintf(stderr, "initial parameter size mismatch\n"); return 1; }
        if (!srv.start()) return 1;
        while (!g_stop && !srv.finished()) std::this_thread::sleep_for(Ms(100));
        std::this_thread::sleep_for(Ms(300));   // let the last RESULTs finish downloading
        srv.stop();
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
    return 0;
}
