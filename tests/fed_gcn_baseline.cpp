// fed_gcn_baseline — reference numbers for the federated GCN demo (no network).
// Reports, on the SAME full-graph test nodes: centralized training (all data on one machine),
// and "local only" (each device trains alone on its partition, never talks to anyone).
//   ./fed_gcn_baseline --opt data=Datasets/cora --opt parts=4 --opt partition=bfs --epochs 150
#include "net/Mobile/BuiltinTasks.hpp"
#include <cstdlib>
using namespace tf::fed; using namespace tf::fed::graph;
int main(int argc, char** argv) 
{
    TaskOptions o; int epochs = 150;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--opt" && i + 1 < argc) 
            o.parse(argv[++i]);
        else if (a == "--epochs" && i + 1 < argc) 
            epochs = std::atoi(argv[++i]);
        else 
        { 
            std::fprintf(stderr, "usage: %s [--opt k=v]... [--epochs N]\n", argv[0]); 
            return 2; 
        }
    }
    o.kv["verbose"] = "0";
    const size_t K = size_t(o.i("parts", 4));
    {   TaskOptions c = o; c.kv["parts"] = "1"; 
        c.kv["part"] = "0"; 
        c.kv["eval_full"] = "1";
        NodeClassifierTask<GCN2> t(c); 
        t.run_local(epochs);
        std::printf("centralized (1 machine, %zu train nodes, %d epochs): test %.1f%%\n", t.n_train(), epochs, t.eval_full_test()); }
    double sum = 0;
    for (size_t p = 0; p < K; p++) {
        TaskOptions c = o; c.kv["part"] = std::to_string(p); 
        c.kv["eval_full"] = "1";
        NodeClassifierTask<GCN2> t(c); 
        t.run_local(epochs);
        double a = t.eval_full_test(); 
        sum += a;
        std::printf("local only, device %zu (%zu nodes, %zu train nodes, edge cut %.0f%%): test %.1f%%\n", p, t.n_nodes(), t.n_train(), 100 * t.cut(), a);
    }
    std::printf("local only, mean over %zu devices: test %.1f%%\n", K, sum / double(K));
    return 0;
}
