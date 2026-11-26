// fed_mobile_client — algorithm-agnostic federated client (host or `adb shell`).
// Picks a registered algorithm, builds its ClientTask and runs the mobile client loop.
//   ./fed_mobile_client --algo gcn --id phoneA --opt data=Datasets/cora --opt parts=3 --opt part=0 --opt eval_full=1
//   ./fed_mobile_client --algo synthetic --params 100000 --id phoneB
//   On a phone, server on the host:  adb reverse tcp:9740 tcp:9740  and  --server 127.0.0.1
#include "net/Mobile/BuiltinTasks.hpp"
#include <csignal>
#include <cstdlib>
#include <fstream>
using namespace tf::fed;
static MobileClient* g_c = nullptr;
int main(int argc, char** argv) {
    register_builtin_tasks();
    ClientConfig c; c.client_id = "client"; std::string algo = "synthetic", gate_file; TaskOptions opts;
    auto arg = [&](int& i) { return i + 1 < argc ? argv[++i] : (std::fprintf(stderr, "missing value for %s\n", argv[i]), std::exit(2), nullptr); };
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--list") { for (auto& kv : TaskRegistry::get().all()) std::printf("%-10s %s\n", kv.first.c_str(), kv.second.help.c_str()); return 0; }
        else if (a == "--algo") algo = arg(i);
        else if (a == "--opt") { if (!opts.parse(arg(i))) { std::fprintf(stderr, "--opt expects key=value\n"); return 2; } }
        else if (a == "--params") opts.kv["params"] = arg(i);
        else if (a == "--train-ms") opts.kv["train_ms"] = arg(i);
        else if (a == "--server") c.host = arg(i);
        else if (a == "--port") c.port = uint16_t(std::atoi(arg(i)));
        else if (a == "--id") c.client_id = arg(i);
        else if (a == "--token") c.auth_token = arg(i);
        else if (a == "--rounds") c.max_rounds = std::strtoull(arg(i), nullptr, 10);
        else if (a == "--chunk-kb") c.chunk_kb = std::strtoul(arg(i), nullptr, 10);
        else if (a == "--no-fp16") c.prefer_fp16 = false;
        else if (a == "--metered") c.metered = true;
        else if (a == "--score") c.score = std::strtof(arg(i), nullptr);          // device capability 0..100
        else if (a == "--mem-mb") c.mem_budget_mb = uint32_t(std::strtoul(arg(i), nullptr, 10));   // hard cap on the shard size
        else if (a == "--shard") {                                                 // what data to receive
            std::string m = arg(i);
            if (m == "auto") c.shard_pref = 0; else if (m == "partition") c.shard_pref = 1; else if (m == "full") c.shard_pref = 3;
            else if (m.rfind("halo", 0) == 0) { c.shard_pref = 2; if (m.size() > 5 && m[4] == ':') c.shard_halo = uint8_t(std::atoi(m.c_str() + 5)); }
            else { std::fprintf(stderr, "--shard: auto | partition | halo[:K] | full\n"); return 2; }
        }
        else if (a == "--gate-file") gate_file = arg(i);   // participate only while the file's first char is '1'
        else 
        { 
            std::fprintf(stderr, "unknown option %s\n(--algo --opt k=v --params --train-ms --server --port --id --token --rounds \
                --chunk-kb --no-fp16 --metered --score --mem-mb --shard --gate-file --list)\n", a.c_str()); return 2; }
    }
    opts.kv["id"] = c.client_id;
    try {
        auto task = TaskRegistry::get().find(algo).client(opts);
        configure(c, task->spec());
        ClientHooks h = hooks_for(*task);
        h.on_log = [&](const std::string& s) { std::fprintf(stderr, "[%s] %s\n", c.client_id.c_str(), s.c_str()); };
        if (!gate_file.empty()) h.can_participate = [&] { std::ifstream f(gate_file); char ch = '1'; f >> ch; return ch == '1'; };
        MobileClient m(c, h); g_c = &m;
        std::signal(SIGINT, [](int) { if (g_c) g_c->stop(); }); std::signal(SIGTERM, [](int) { if (g_c) g_c->stop(); });
        m.run();
        std::printf("[%s] done: rounds=%llu syncs=%llu skips=%llu connects=%llu up=%.1fKB down=%.1fKB\n", c.client_id.c_str(),
            (unsigned long long)m.stats().rounds_done.load(), (unsigned long long)m.stats().syncs.load(), (unsigned long long)m.stats().skips.load(),
            (unsigned long long)m.stats().connects.load(), m.stats().bytes_up / 1e3, m.stats().bytes_down / 1e3);
        return m.failed() ? 3 : 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "error: %s\n", e.what()); return 1; }
}
