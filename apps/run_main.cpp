// Comparison runner: evaluate all allocators on the same prompt stream.
// Learning allocators (Q-learning, DQN) are optionally trained for --episodes
// episodes before the final comparison run.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <torch/torch.h>
#if defined(__linux__)
#include <unistd.h>
#endif

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

// ---------------------------------------------------------------------------
// Build-time metadata (injected by CMake via target_compile_definitions).
// Fallbacks keep the file compilable if built without those definitions.
// ---------------------------------------------------------------------------
#ifndef LLM_DEVELOPER
#define LLM_DEVELOPER "unknown"
#endif
#ifndef LLM_DEV_SINCE
#define LLM_DEV_SINCE "unknown"
#endif
#ifndef LLM_VERSION
#define LLM_VERSION "0.0.0"
#endif
#ifndef LLM_GIT_HASH
#define LLM_GIT_HASH "unknown"
#endif
#ifndef LLM_BUILD_DATE
#define LLM_BUILD_DATE __DATE__ " " __TIME__
#endif
#ifndef LLM_COMPILER
#define LLM_COMPILER "unknown"
#endif
#ifndef LLM_BUILD_TYPE
#define LLM_BUILD_TYPE "unknown"
#endif

using namespace llm_alloc;

namespace {

struct Args {
    int n_prompts = 200;
    uint64_t seed = 42;
    std::string objective = "energy";
    std::string engine = "static";
    int episodes = 0;
    int small_gpus = 2;
    int medium_gpus = 2;
    int large_gpus = 1;
    std::string json_out;  // empty = no JSON
};

void usage() {
    std::printf(
        "LLM Allocation Simulator (C++ port)\n"
        "Usage: llm_run [options]\n"
        "  -n, --n-prompts N     prompts per episode (default 200)\n"
        "  --seed N              RNG seed (default 42)\n"
        "  --objective O         energy|balanced|quality|latency (default energy)\n"
        "  --engine E            static|batched (default static)\n"
        "  --episodes N          training episodes for learners (0=none)\n"
        "  --small-gpus N        (default 2)\n"
        "  --medium-gpus N       (default 2)\n"
        "  --large-gpus N        (default 1)\n"
        "  --json PATH           write metrics to a JSON file\n");
}

// ---------------------------------------------------------------------------
// Host hardware introspection (best-effort, Linux-focused).
// ---------------------------------------------------------------------------
struct HostInfo {
    unsigned hw_threads = 0;   // logical cores reported by the OS
    long online_cpus = 0;      // sysconf online CPUs
    double ram_total_gb = 0.0; // physical RAM
    double ram_avail_gb = 0.0; // available RAM
    int torch_threads = 0;     // intra-op threads LibTorch will use
    int torch_interop = 0;     // inter-op threads LibTorch will use
    std::string cpu_model;     // model name from /proc/cpuinfo
};

HostInfo probe_host() {
    HostInfo h;
    h.hw_threads = std::thread::hardware_concurrency();
#if defined(__linux__)
    h.online_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_sz = sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_sz > 0)
        h.ram_total_gb = double(pages) * double(page_sz) / (1024.0 * 1024.0 * 1024.0);

    // Available RAM and CPU model from /proc (optional, ignore if absent).
    std::ifstream mem("/proc/meminfo");
    std::string key;
    while (mem >> key) {
        if (key == "MemAvailable:") {
            double kb = 0; mem >> kb;
            h.ram_avail_gb = kb / (1024.0 * 1024.0);
            break;
        }
        mem.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    std::ifstream cpu("/proc/cpuinfo");
    std::string line;
    while (std::getline(cpu, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) {
                h.cpu_model = line.substr(pos + 1);
                size_t b = h.cpu_model.find_first_not_of(" \t");
                if (b != std::string::npos) h.cpu_model = h.cpu_model.substr(b);
            }
            break;
        }
    }
#endif
    h.torch_threads = torch::get_num_threads();
    h.torch_interop = torch::get_num_interop_threads();
    return h;
}

// Print the startup banner: who/when/what/how/where.
void print_banner(const Args& a) {
    HostInfo h = probe_host();

    std::time_t now = std::time(nullptr);
    char now_buf[64] = {0};
    std::strftime(now_buf, sizeof(now_buf), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&now));

    const std::string bar(78, '=');
    std::printf("%s\n", bar.c_str());
    std::printf(
        "   ___  ____ _____ ___ __  __ _   _ ____\n"
        "  / _ \\|  _ \\_   _|_ _|  \\/  | | | / ___|\n"
        " | | | | |_) || |  | || |\\/| | | | \\___ \\\n"
        " | |_| |  __/ | |  | || |  | | |_| |___) |\n"
        "  \\___/|_|    |_| |___|_|  |_|\\___/|____/   optimus23579\n");
    std::printf("  LLM allocation RL simulator\n");
    std::printf("%s\n", bar.c_str());

    // --- Provenance ---
    std::printf("  Developer        : %s\n", LLM_DEVELOPER);
    std::printf("  Developed since  : %s\n", LLM_DEV_SINCE);
    std::printf("  Version          : %s  (build %s, %s)\n",
                LLM_VERSION, LLM_GIT_HASH, LLM_BUILD_TYPE);
    std::printf("  Built            : %s UTC   [%s]\n",
                LLM_BUILD_DATE, LLM_COMPILER);
    std::printf("  Started          : %s (local)\n", now_buf);

    // --- Launch configuration ---
    std::printf("%s\n", std::string(78, '-').c_str());
    std::printf("  Engine           : %s\n", a.engine.c_str());
    std::printf("  Objective        : %s\n", a.objective.c_str());
    std::printf("  Launch params    : n_prompts=%d  seed=%llu  episodes=%d\n",
                a.n_prompts, (unsigned long long)a.seed, a.episodes);
    std::printf("  GPU fleet        : small=%d  medium=%d  large=%d  (total %d)\n",
                a.small_gpus, a.medium_gpus, a.large_gpus,
                a.small_gpus + a.medium_gpus + a.large_gpus);
    std::printf("  JSON output      : %s\n",
                a.json_out.empty() ? "(none)" : a.json_out.c_str());

    // --- What will be computed, and how ---
    std::printf("%s\n", std::string(78, '-').c_str());
    std::printf("  Task             : allocate %d prompts to (GPU, model) pairs,\n",
                a.n_prompts);
    std::printf("                     score each by reward = -(latency,energy,quality loss)\n");
    std::printf("  Heuristics       : random, always_small, always_large, difficulty,\n");
    std::printf("                     least_loaded, energy_aware\n");
    std::printf("  Learners         : epsilon_greedy (bandit), q_learning (tabular TD),\n");
    std::printf("                     dqn (LibTorch MLP, Double-DQN + replay)\n");
    if (a.episodes > 0)
        std::printf("  Training         : %d episodes per learner (convergence shown below)\n",
                    a.episodes);
    else
        std::printf("  Training         : none (episodes=0) — learners run untrained\n");

    // --- Hardware & parallelism (honest: this build is CPU-only, serial) ---
    std::printf("%s\n", std::string(78, '-').c_str());
    if (!h.cpu_model.empty())
        std::printf("  CPU              : %s\n", h.cpu_model.c_str());
    std::printf("  Logical cores    : %u (OS online: %ld)\n",
                h.hw_threads, h.online_cpus);
    std::printf("  RAM              : %.1f GB total, %.1f GB available\n",
                h.ram_total_gb, h.ram_avail_gb);
    std::printf("  Compute backend  : CPU only (no CUDA/GPU used)\n");
    std::printf("  Parallelism      : single-threaded simulation; no OpenMP, no MPI\n");
    std::printf("  LibTorch threads : intra-op=%d, inter-op=%d (used only inside DQN tensor ops)\n",
                h.torch_threads, h.torch_interop);
    std::printf("%s\n", bar.c_str());
}

bool parse_args(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&](const char* name) -> const char* {
            if (i + 1 >= argc) { std::printf("missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if (s == "-h" || s == "--help") { usage(); return false; }
        else if (s == "-n" || s == "--n-prompts") { auto v = next("--n-prompts"); if (!v) return false; a.n_prompts = std::atoi(v); }
        else if (s == "--seed") { auto v = next("--seed"); if (!v) return false; a.seed = std::strtoull(v, nullptr, 10); }
        else if (s == "--objective") { auto v = next("--objective"); if (!v) return false; a.objective = v; }
        else if (s == "--engine") { auto v = next("--engine"); if (!v) return false; a.engine = v; }
        else if (s == "--episodes") { auto v = next("--episodes"); if (!v) return false; a.episodes = std::atoi(v); }
        else if (s == "--small-gpus") { auto v = next("--small-gpus"); if (!v) return false; a.small_gpus = std::atoi(v); }
        else if (s == "--medium-gpus") { auto v = next("--medium-gpus"); if (!v) return false; a.medium_gpus = std::atoi(v); }
        else if (s == "--large-gpus") { auto v = next("--large-gpus"); if (!v) return false; a.large_gpus = std::atoi(v); }
        else if (s == "--json") { auto v = next("--json"); if (!v) return false; a.json_out = v; }
        else { std::printf("unknown arg: %s\n", s.c_str()); usage(); return false; }
    }
    return true;
}

// Run a fresh episode (fresh GPU pool) for one allocator on the given prompts.
AggregatedMetrics run_episode(const Args& a, const RewardWeights& weights,
                              const std::vector<LLM>& llm_pool, Allocator& alloc,
                              const std::vector<Prompt>& prompts) {
    std::vector<GPU> gpus = create_gpu_pool(a.small_gpus, a.medium_gpus, a.large_gpus);
    if (a.engine == "batched") {
        BatchedSimulator sim(gpus, llm_pool, BatchedConfig{weights});
        auto results = sim.run(alloc, prompts, false);
        return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped);
    }
    SimulatorConfig scfg;
    scfg.n_prompts = a.n_prompts;
    scfg.reward_weights = weights;
    Simulator sim(gpus, llm_pool, scfg);
    auto results = sim.run(alloc, prompts, false);
    return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped);
}

void print_report(const std::vector<AggregatedMetrics>& metrics, const std::string& objective) {
    std::vector<const AggregatedMetrics*> sorted;
    for (const auto& m : metrics) sorted.push_back(&m);
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const AggregatedMetrics* a, const AggregatedMetrics* b) {
            return a->mean_reward > b->mean_reward;
        });

    std::printf("\n  Objective: %s\n", objective.c_str());
    std::string sep(120, '-');
    std::printf("%s\n", sep.c_str());
    for (const AggregatedMetrics* m : sorted) {
        double util_sum = 0.0;
        for (const auto& kv : m->gpu_utilisation) util_sum += kv.second;
        double util_avg = util_sum / std::max<size_t>(1, m->gpu_utilisation.size());
        std::printf("%-20s | reward %8.3f | latency %7.2fs (p95 %7.2fs) | "
                    "energy %8.0fJ (%5.0f+%5.0f) | quality %.3f | GPU-util %.1f%% | drop %d\n",
                    m->allocator_name.c_str(), m->mean_reward, m->mean_latency,
                    m->p95_latency, m->system_energy, m->total_energy, m->idle_energy,
                    m->mean_quality, util_avg * 100.0, m->dropped);
    }
    std::printf("%s\n", sep.c_str());
}

// Minimal hand-rolled JSON writer (no external dependency needed for output).
void write_json(const std::string& path, const Args& a,
                const std::vector<AggregatedMetrics>& metrics) {
    std::ofstream f(path);
    f << "{\n  \"objective\": \"" << a.objective << "\",\n";
    f << "  \"engine\": \"" << a.engine << "\",\n";
    f << "  \"episodes\": " << a.episodes << ",\n";
    f << "  \"metrics\": [\n";
    for (size_t i = 0; i < metrics.size(); ++i) {
        const AggregatedMetrics& m = metrics[i];
        f << "    {\"allocator\": \"" << m.allocator_name << "\", "
          << "\"n_jobs\": " << m.n_jobs << ", "
          << "\"mean_reward\": " << m.mean_reward << ", "
          << "\"mean_latency\": " << m.mean_latency << ", "
          << "\"p95_latency\": " << m.p95_latency << ", "
          << "\"total_energy\": " << m.total_energy << ", "
          << "\"idle_energy\": " << m.idle_energy << ", "
          << "\"system_energy\": " << m.system_energy << ", "
          << "\"mean_quality\": " << m.mean_quality << ", "
          << "\"dropped\": " << m.dropped << "}";
        f << (i + 1 < metrics.size() ? ",\n" : "\n");
    }
    f << "  ]\n}\n";
}

// Final summary: a plain-language verdict on what the run produced.
void print_run_summary(const Args& a,
                       const std::vector<AggregatedMetrics>& metrics) {
    const std::string bar(78, '=');
    std::printf("\n%s\n", bar.c_str());
    std::printf("  RUN SUMMARY\n");
    std::printf("%s\n", bar.c_str());

    if (metrics.empty()) {
        std::printf("  No allocators were evaluated — nothing to report.\n");
        std::printf("%s\n", bar.c_str());
        return;
    }

    // Best (highest reward) and worst, plus totals.
    const AggregatedMetrics* best = &metrics[0];
    const AggregatedMetrics* worst = &metrics[0];
    int total_jobs = 0, total_dropped = 0;
    for (const auto& m : metrics) {
        if (m.mean_reward > best->mean_reward) best = &m;
        if (m.mean_reward < worst->mean_reward) worst = &m;
        total_jobs += m.n_jobs;
        total_dropped += m.dropped;
    }

    // Did the learners beat the heuristic field? Compare the best learner to
    // the best heuristic by reward (names of the three learners are known).
    auto is_learner_name = [](const std::string& n) {
        return n == "epsilon_greedy" || n == "q_learning" || n == "dqn";
    };
    const AggregatedMetrics* best_learner = nullptr;
    const AggregatedMetrics* best_heur = nullptr;
    for (const auto& m : metrics) {
        if (is_learner_name(m.allocator_name)) {
            if (!best_learner || m.mean_reward > best_learner->mean_reward) best_learner = &m;
        } else {
            if (!best_heur || m.mean_reward > best_heur->mean_reward) best_heur = &m;
        }
    }

    std::printf("  Status           : completed successfully\n");
    std::printf("  Engine/objective : %s / %s\n", a.engine.c_str(), a.objective.c_str());
    std::printf("  Allocators run   : %zu  (on %d prompts each)\n",
                metrics.size(), a.n_prompts);
    std::printf("  Training         : %s\n",
                a.episodes > 0 ? "learners trained before evaluation"
                               : "none (learners evaluated untrained)");
    std::printf("  Prompts dropped  : %d (summed across allocators)\n", total_dropped);
    std::printf("%s\n", std::string(78, '-').c_str());
    std::printf("  Best  policy     : %-14s reward %8.3f | sys_energy %8.0fJ | quality %.3f\n",
                best->allocator_name.c_str(), best->mean_reward,
                best->system_energy, best->mean_quality);
    std::printf("  Worst policy     : %-14s reward %8.3f | sys_energy %8.0fJ | quality %.3f\n",
                worst->allocator_name.c_str(), worst->mean_reward,
                worst->system_energy, worst->mean_quality);
    if (best_learner && best_heur) {
        std::printf("%s\n", std::string(78, '-').c_str());
        std::printf("  Best learner     : %-14s reward %8.3f\n",
                    best_learner->allocator_name.c_str(), best_learner->mean_reward);
        std::printf("  Best heuristic   : %-14s reward %8.3f\n",
                    best_heur->allocator_name.c_str(), best_heur->mean_reward);
        if (best_learner->mean_reward > best_heur->mean_reward) {
            std::printf("  Verdict          : the best learner BEAT the best heuristic "
                        "(by %.3f reward).\n",
                        best_learner->mean_reward - best_heur->mean_reward);
        } else {
            std::printf("  Verdict          : no learner beat the best heuristic "
                        "(gap %.3f reward)%s.\n",
                        best_heur->mean_reward - best_learner->mean_reward,
                        a.episodes > 0 ? "" : " — try --episodes > 0 to train them");
        }
    }
    std::printf("%s\n", bar.c_str());
}

// Evaluate every allocator under each objective profile on the SAME prompt
// stream, and report which policy wins each profile. Rewards are NOT comparable
// across profiles (different weightings), so we only pick the best WITHIN each.
void print_profile_winners(const Args& a, const std::vector<LLM>& llm_pool,
                           const std::vector<Prompt>& prompts,
                           std::vector<std::unique_ptr<Allocator>>& heuristics,
                           std::vector<std::unique_ptr<Allocator>>& learners) {
    const char* profiles[] = {"energy", "balanced", "quality", "latency"};
    const std::string bar(78, '=');
    std::printf("\n%s\n", bar.c_str());
    std::printf("  OBJECTIVE-PROFILE WINNERS  (same %d prompts, each profile judged on its own reward)\n",
                a.n_prompts);
    std::printf("%s\n", bar.c_str());
    std::printf("  %-10s | %-14s | %10s | %10s | %8s\n",
                "profile", "winner", "reward", "sys_energy", "quality");
    std::printf("%s\n", std::string(78, '-').c_str());

    for (const char* prof : profiles) {
        RewardWeights w;
        try {
            w = RewardWeights::profile(prof);
        } catch (const std::exception&) {
            continue;
        }

        const AggregatedMetrics* best = nullptr;
        AggregatedMetrics best_copy;
        auto consider = [&](Allocator& alloc) {
            AggregatedMetrics m = run_episode(a, w, llm_pool, alloc, prompts);
            if (!best || m.mean_reward > best_copy.mean_reward) {
                best_copy = m;
                best = &best_copy;
            }
        };
        for (auto& h : heuristics) consider(*h);
        for (auto& l : learners) consider(*l);

        if (best) {
            std::printf("  %-10s | %-14s | %10.3f | %9.0fJ | %8.3f\n",
                        prof, best_copy.allocator_name.c_str(),
                        best_copy.mean_reward, best_copy.system_energy,
                        best_copy.mean_quality);
        }
    }
    std::printf("%s\n", std::string(78, '-').c_str());
    if (a.episodes > 0)
        std::printf("  Note: learners were trained on the '%s' profile only; their scores\n"
                    "        under other profiles reflect that fixed policy, not retraining.\n",
                    a.objective.c_str());
    else
        std::printf("  Note: learners are untrained here (episodes=0).\n");
    std::printf("%s\n", bar.c_str());
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 0;

    RewardWeights weights;
    try {
        weights = RewardWeights::profile(a.objective);
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
    std::vector<LLM> llm_pool = default_llm_pool();

    print_banner(a);

    // --- Learning allocators ---
    std::vector<std::unique_ptr<Allocator>> learners;
    learners.push_back(std::make_unique<EpsilonGreedyAllocator>(0.15, 0.999, 5, a.seed));
    learners.push_back(std::make_unique<QLearningAllocator>(
        0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, a.seed));
    learners.push_back(std::make_unique<DQNAllocator>(
        0.3, 0.995, 0.01, 0.95, 1e-3, 64, 5000, 50, 64, 0.0, a.seed));

    // --- Train learners if requested ---
    if (a.episodes > 0) {
        std::printf("\n============================================================\n");
        std::printf("  Training learning allocators (%d episodes, %s objective)\n",
                    a.episodes, a.objective.c_str());
        std::printf("============================================================\n");
        for (auto& learner : learners) {
            std::printf("\n--- %s ---\n", learner->name().c_str());
            train_learner(*learner, a.episodes, a.n_prompts, a.seed,
                          a.small_gpus, a.medium_gpus, a.large_gpus,
                          a.objective, true, a.engine);
        }
    }

    // --- Final evaluation on a common prompt stream ---
    std::printf("\n%s\n", std::string(78, '=').c_str());
    std::printf("  EVALUATION — all allocators on one shared stream of %d prompts\n",
                a.n_prompts);
    std::printf("%s\n", std::string(78, '=').c_str());

    PromptGenerator gen(0.5, 0.2, 1.0, 256, a.seed + 9999);
    std::vector<Prompt> prompts = gen.generate(a.n_prompts);

    std::vector<AggregatedMetrics> all_metrics;

    auto heuristics = all_heuristic_allocators(a.seed);
    const size_t total_allocs = heuristics.size() + learners.size();
    size_t done = 0;

    auto run_and_log = [&](Allocator& alloc, const char* kind) {
        std::printf("  [%2zu/%2zu] %-13s %-14s ... ",
                    ++done, total_allocs, kind, alloc.name().c_str());
        std::fflush(stdout);
        AggregatedMetrics m = run_episode(a, weights, llm_pool, alloc, prompts);
        std::printf("reward %8.3f | sys_energy %8.0fJ | quality %.3f | drop %d\n",
                    m.mean_reward, m.system_energy, m.mean_quality, m.dropped);
        std::fflush(stdout);
        all_metrics.push_back(std::move(m));
    };

    for (auto& alloc : heuristics) run_and_log(*alloc, "heuristic");
    for (auto& learner : learners) run_and_log(*learner, "learner");

    print_report(all_metrics, a.objective);

    if (!a.json_out.empty()) {
        write_json(a.json_out, a, all_metrics);
        std::printf("[INFO] metrics written to %s\n", a.json_out.c_str());
    }

    // --- Final run summary: what happened, did it succeed, what we got ---
    print_run_summary(a, all_metrics);

    // --- Cross-profile winners: which policy wins energy/balanced/quality/latency ---
    print_profile_winners(a, llm_pool, prompts, heuristics, learners);
    return 0;
}
