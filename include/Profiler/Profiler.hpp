#pragma once

/*
 * Profiler.hpp
 * ============
 * Top-level orchestrator. Runs the full profiling pipeline:
 *
 *   1. HardwareFingerprint  — detect CPU gen, ISA, RAM type/freq, cache, storage
 *   2. MemoryProfiler       — snapshot baseline RAM usage
 *   3. BenchmarkRunner      — measure matmul GFLOP/s, bandwidth, latency, storage
 *   4. HyperparamAdvisor    — compute optimal batch/block/n_embed + algo policy
 *   5. JSON serializer      — produce a compact payload for the server
 *
 * The JSON payload is sent to the server which:
 *   - stores the client's capability profile
 *   - assigns workloads proportional to client capability
 *   - may override hyperparams for federated training consistency
 *
 * Usage:
 *
 *   Profiler p;
 *   p.run();                             // runs all benchmarks
 *   std::string json = p.to_json();      // serialize for server
 *   p.print_summary();                   // human-readable report
 *   const HyperparamConfig& cfg = p.config();  // use in training loop
 */

#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <cstdio>

#include "HardwareFingerprint.hpp"
#include "MemoryProfiler.hpp"
#include "BenchmarkRunner.hpp"
#include "HyperparamAdvisor.hpp"

class Profiler {
public:
    void run(size_t vocab_size = 50257, bool verbose = true) {
        if (verbose) printf("\n[TensorF Profiler] Starting hardware detection...\n");

        fingerprint_ = detect_hardware();

        if (verbose) {
            printf("[TensorF Profiler] CPU: %s\n", fingerprint_.cpu.model_name.c_str());
            printf("[TensorF Profiler] ISA: %s\n", isa_string(fingerprint_.cpu.best_isa()));
            printf("[TensorF Profiler] RAM: %u MB available, %s %.0f MT/s\n",
                   fingerprint_.ram.available_mb,
                   ram_type_str(fingerprint_.ram.type),
                   (double)fingerprint_.ram.speed_mts);
            printf("[TensorF Profiler] L3 cache: %zu MB\n",
                   fingerprint_.cache.l3_bytes / 1024 / 1024);
        }

        mem_profiler_.snapshot(MemStage::BASELINE);

        if (verbose) printf("[TensorF Profiler] Running benchmarks...\n");

        BenchmarkRunner runner;
        runner.set_cache_sizes(
            fingerprint_.cache.l1d_bytes,
            fingerprint_.cache.l2_bytes,
            fingerprint_.cache.l3_bytes);

        bench_result_ = runner.run_all(verbose);

        config_ = HyperparamAdvisor::advise(fingerprint_, bench_result_, vocab_size);

        if (verbose) HyperparamAdvisor::print_config(config_, fingerprint_);

        vocab_size_ = vocab_size;
        ran_ = true;
    }

    void profile_load(std::function<void()> fn)       { mem_profiler_.profile_workload(MemStage::LOADED, fn); }
    void profile_train_step(std::function<void()> fn) { mem_profiler_.profile_workload(MemStage::TRAIN,  fn); }
    void profile_infer_step(std::function<void()> fn) { mem_profiler_.profile_workload(MemStage::INFER,  fn); }

    const HardwareFingerprint& fingerprint()   const { return fingerprint_; }
    const BenchmarkResult&     benchmarks()    const { return bench_result_; }
    const HyperparamConfig&    config()        const { return config_; }
    MemoryReport               memory_report() const { return mem_profiler_.report(); }

    void print_summary() const {
        if (!ran_) { printf("[Profiler] Not yet run.\n"); return; }

        printf("\n════════════════════════════════════════════════════\n");
        printf("  TensorF Client Profiler — Full Report\n");
        printf("════════════════════════════════════════════════════\n");

        printf("\n[ Hardware ]\n");
        printf("  CPU      : %s\n",  fingerprint_.cpu.model_name.c_str());
        printf("  Gen      : %s\n",  gen_string(fingerprint_.cpu.gen));
        printf("  Cores    : %u physical / %u logical\n",
               fingerprint_.cpu.physical_cores, fingerprint_.cpu.logical_cores);
        printf("  ISA      : %s\n",  isa_string(fingerprint_.cpu.best_isa()));
        printf("  Freq     : %.0f MHz (max)\n", fingerprint_.cpu.max_freq_mhz);

        printf("\n[ Cache ]\n");
        printf("  L1d : %zu KB\n", fingerprint_.cache.l1d_bytes / 1024);
        printf("  L2  : %zu KB\n", fingerprint_.cache.l2_bytes  / 1024);
        printf("  L3  : %zu MB\n", fingerprint_.cache.l3_bytes  / (1024*1024));

        printf("\n[ RAM ]\n");
        printf("  Type      : %s @ %u MT/s\n",
               ram_type_str(fingerprint_.ram.type), fingerprint_.ram.speed_mts);
        printf("  Total     : %u MB\n",     fingerprint_.ram.total_mb);
        printf("  Available : %u MB\n",     fingerprint_.ram.available_mb);
        printf("  Channels  : %u\n",        fingerprint_.ram.channels);
        printf("  Theor BW  : %.1f GB/s\n", fingerprint_.ram.bandwidth_gbs);

        printf("\n[ Storage ]\n");
        for (const auto& s : fingerprint_.storage)
            printf("  /dev/%-8s : %s\n", s.name.c_str(), storage_type_str(s.type));

        printf("\n[ Benchmarks ]\n");
        printf("  Matmul (L3)   : %.2f GFLOP/s\n", bench_result_.matmul.l3_gflops);
        printf("  Matmul (RAM)  : %.2f GFLOP/s\n", bench_result_.matmul.ram_gflops);
        printf("  Mem BW (read) : %.2f GB/s\n",    bench_result_.bandwidth.read_gbs);
        printf("  L1 latency    : %.1f ns\n",       bench_result_.latency.l1_latency_ns);
        printf("  L3 latency    : %.1f ns\n",       bench_result_.latency.l3_latency_ns);
        printf("  RAM latency   : %.1f ns\n",       bench_result_.latency.ram_latency_ns);
        printf("  Storage rd    : %.0f MB/s\n",     bench_result_.storage.seq_read_mbs);

        auto mr = mem_profiler_.report();
        if (mr.loaded_rss_mb > 0 || mr.train_rss_mb > 0)
            mem_profiler_.print_summary();

        HyperparamAdvisor::print_config(config_, fingerprint_);
    }

    // ── JSON serializer ──────────────────────────────────────────────────────

    std::string to_json() const {
        std::ostringstream j;
        j << std::fixed << std::setprecision(4);

        auto now = std::chrono::system_clock::now();
        auto tt  = std::chrono::system_clock::to_time_t(now);
        char tbuf[32];
        strftime(tbuf, sizeof(tbuf), "%Y-%m-%dT%H:%M:%SZ", gmtime(&tt));

        j << "{\n";
        j << "  \"profiler_version\": \"1.0\",\n";
        j << "  \"timestamp\": \"" << tbuf << "\",\n";

        j << "  \"hardware\": {\n";
        j << "    \"cpu\": {\n";
        j << "      \"model\": "          << qs(fingerprint_.cpu.model_name)      << ",\n";
        j << "      \"generation\": \""   << gen_string(fingerprint_.cpu.gen)     << "\",\n";
        j << "      \"physical_cores\": " << fingerprint_.cpu.physical_cores      << ",\n";
        j << "      \"logical_cores\": "  << fingerprint_.cpu.logical_cores       << ",\n";
        j << "      \"max_freq_mhz\": "   << fingerprint_.cpu.max_freq_mhz        << ",\n";
        j << "      \"isa_best\": \""     << isa_string(fingerprint_.cpu.best_isa()) << "\",\n";
        j << "      \"avx2\": "           << jb(fingerprint_.cpu.has_avx2)        << ",\n";
        j << "      \"avx512f\": "        << jb(fingerprint_.cpu.has_avx512f)     << ",\n";
        j << "      \"avx512_vnni\": "    << jb(fingerprint_.cpu.has_avx512_vnni) << ",\n";
        j << "      \"amx\": "            << jb(fingerprint_.cpu.has_amx)         << ",\n";
        j << "      \"embed_alignment\": "<< fingerprint_.cpu.embed_alignment()   << "\n";
        j << "    },\n";
        j << "    \"cache\": { \"l1d\": " << fingerprint_.cache.l1d_bytes
          << ", \"l2\": "                 << fingerprint_.cache.l2_bytes
          << ", \"l3\": "                 << fingerprint_.cache.l3_bytes << " },\n";
        j << "    \"ram\": {\n";
        j << "      \"type\": \""         << ram_type_str(fingerprint_.ram.type)  << "\",\n";
        j << "      \"speed_mts\": "      << fingerprint_.ram.speed_mts           << ",\n";
        j << "      \"total_mb\": "       << fingerprint_.ram.total_mb            << ",\n";
        j << "      \"available_mb\": "   << fingerprint_.ram.available_mb        << ",\n";
        j << "      \"channels\": "       << fingerprint_.ram.channels            << ",\n";
        j << "      \"bandwidth_gbs\": "  << fingerprint_.ram.bandwidth_gbs       << "\n";
        j << "    },\n";
        j << "    \"capability_score\": " << fingerprint_.capability_score        << "\n";
        j << "  },\n";

        j << "  \"benchmarks\": {\n";
        j << "    \"matmul_peak_gflops\": "   << bench_result_.matmul.peak_gflops    << ",\n";
        j << "    \"matmul_l3_gflops\": "     << bench_result_.matmul.l3_gflops      << ",\n";
        j << "    \"matmul_ram_gflops\": "    << bench_result_.matmul.ram_gflops     << ",\n";
        j << "    \"mem_read_gbs\": "         << bench_result_.bandwidth.read_gbs    << ",\n";
        j << "    \"mem_write_gbs\": "        << bench_result_.bandwidth.write_gbs   << ",\n";
        j << "    \"latency_l1_ns\": "        << bench_result_.latency.l1_latency_ns << ",\n";
        j << "    \"latency_l3_ns\": "        << bench_result_.latency.l3_latency_ns << ",\n";
        j << "    \"latency_ram_ns\": "       << bench_result_.latency.ram_latency_ns<< ",\n";
        j << "    \"storage_read_mbs\": "     << bench_result_.storage.seq_read_mbs  << "\n";
        j << "  },\n";

        auto mr = mem_profiler_.report();
        j << "  \"memory_profile\": {\n";
        j << "    \"baseline_mb\": "          << mr.baseline_rss_mb     << ",\n";
        j << "    \"loaded_mb\": "            << mr.loaded_rss_mb       << ",\n";
        j << "    \"train_peak_mb\": "        << mr.train_rss_mb        << ",\n";
        j << "    \"infer_peak_mb\": "        << mr.infer_rss_mb        << ",\n";
        j << "    \"param_cost_mb\": "        << mr.param_ram_mb        << ",\n";
        j << "    \"train_overhead_mb\": "    << mr.train_overhead_mb   << ",\n";
        j << "    \"total_peak_mb\": "        << mr.total_peak_mb       << ",\n";
        j << "    \"recommended_free_mb\": "  << mr.recommended_free_mb << "\n";
        j << "  },\n";

        j << "  \"recommended_config\": {\n";
        j << "    \"batch_size\": "       << config_.batch_size        << ",\n";
        j << "    \"block_size\": "       << config_.block_size        << ",\n";
        j << "    \"n_embed\": "          << config_.n_embed           << ",\n";
        j << "    \"n_layers\": "         << config_.n_layers          << ",\n";
        j << "    \"n_heads\": "          << config_.n_heads           << ",\n";
        j << "    \"grad_accum_steps\": " << config_.grad_accum_steps  << ",\n";
        j << "    \"quantization\": \""   << quant_name(config_.algo.quant) << "\",\n";
        j << "    \"num_threads\": "      << config_.algo.num_threads  << ",\n";
        j << "    \"matmul_tile\": "      << config_.algo.matmul_tile_size << ",\n";
        j << "    \"use_blas\": "         << jb(config_.algo.use_blas) << ",\n";
        j << "    \"use_mmap\": "         << jb(config_.algo.use_mmap_weights) << ",\n";
        j << "    \"flash_attn\": "       << jb(config_.algo.use_flash_attn_style) << ",\n";
        j << "    \"estimated_param_mb\": "<< config_.estimated_param_mb << ",\n";
        j << "    \"estimated_train_mb\": "<< config_.estimated_train_mb << ",\n";
        j << "    \"estimated_infer_mb\": "<< config_.estimated_infer_mb << ",\n";
        j << "    \"fit_score\": "         << config_.fit_score          << "\n";
        j << "  }\n";
        j << "}\n";
        return j.str();
    }

private:
    HardwareFingerprint fingerprint_;
    BenchmarkResult     bench_result_;
    HyperparamConfig    config_;
    MemoryProfiler      mem_profiler_;
    size_t              vocab_size_ = 50257;
    bool                ran_        = false;

    static const char* isa_string(ISA i) {
        switch (i) {
            case ISA::AMX:         return "AMX";
            case ISA::AVX512_VNNI: return "AVX512_VNNI";
            case ISA::AVX512:      return "AVX512";
            case ISA::AVX2:        return "AVX2";
            case ISA::AVX:         return "AVX";
            default:               return "BASELINE";
        }
    }
    static const char* gen_string(CPUGen g) {
        switch (g) {
            case CPUGen::INTEL_PRE_HASWELL:     return "intel_pre_haswell";
            case CPUGen::INTEL_HASWELL:         return "intel_haswell";
            case CPUGen::INTEL_SKYLAKE:         return "intel_skylake";
            case CPUGen::INTEL_ICE_LAKE:        return "intel_ice_lake";
            case CPUGen::INTEL_ALDER_LAKE:      return "intel_alder_lake";
            case CPUGen::INTEL_SAPPHIRE_RAPIDS: return "intel_sapphire_rapids";
            case CPUGen::AMD_ZEN1:              return "amd_zen1";
            case CPUGen::AMD_ZEN2:              return "amd_zen2";
            case CPUGen::AMD_ZEN3:              return "amd_zen3";
            case CPUGen::AMD_ZEN4:              return "amd_zen4";
            case CPUGen::AMD_ZEN5:              return "amd_zen5";
            default:                            return "unknown";
        }
    }
    static const char* ram_type_str(RAMType t) {
        switch (t) {
            case RAMType::DDR3:   return "DDR3";
            case RAMType::DDR4:   return "DDR4";
            case RAMType::DDR5:   return "DDR5";
            case RAMType::LPDDR4: return "LPDDR4";
            case RAMType::LPDDR5: return "LPDDR5";
            default:              return "unknown";
        }
    }
    static const char* storage_type_str(StorageType t) {
        switch (t) {
            case StorageType::HDD:       return "HDD";
            case StorageType::SATA_SSD:  return "SATA_SSD";
            case StorageType::NVME_GEN3: return "NVMe_Gen3";
            case StorageType::NVME_GEN4: return "NVMe_Gen4";
            case StorageType::NVME_GEN5: return "NVMe_Gen5";
            default:                     return "unknown";
        }
    }
    static std::string qs(const std::string& s) {
        std::ostringstream o;
        o << '"';
        for (char c : s) {
            if (c == '"') o << "\\\"";
            else if (c == '\\') o << "\\\\";
            else if (c == '\n') o << "\\n";
            else o << c;
        }
        o << '"';
        return o.str();
    }
    static const char* jb(bool b) { return b ? "true" : "false"; }
};
