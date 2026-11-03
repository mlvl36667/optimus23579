// Comparison runner: evaluate all allocators on the same prompt stream.
// Learning allocators (Q-learning, DQN) are optionally trained for --episodes
// episodes before the final comparison run.
#include <algorithm>
#include <cstdio>
#include <cmath>
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
    double mean_interarrival = 1.0;
    double gamma = 0.95;
    bool debug = false;
    double q_epsilon = 0.2;
    double q_alpha = 0.1;
    double q_epsilon_decay = 0.998;
    double q_epsilon_min = 0.01;
    std::string q_visits;
    std::string q_td_trace;
    std::string q_reward_stats;
    std::string q_transition_model;
    std::string q_policy;
    std::string dqn_save;
    std::string dqn_load;
    std::string dqn_trace;
    std::string epsilon_trace;
    std::string q_action_trace;
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
        "  --objective O         energy|energy_q93|balanced|quality|latency (default energy)\n"
        "  --engine E            static|batched (default static)\n"
        "  --episodes N          training episodes for learners (0=none)\n"
        "  --mean-interarrival S mean time between prompts in seconds (default 1.0)\n"
        "  --gamma X             learner discount factor (default 0.95)\n"
        "  --small-gpus N        (default 2)\n"
        "  --medium-gpus N       (default 2)\n"
        "  --large-gpus N        (default 1)\n"
        "  --json PATH           write metrics to a JSON file\n");
    std::printf("  --debug               reward breakdown and placement diagnostics per evaluation\n");
    std::printf("  --q-epsilon X         Q-learning initial exploration (default 0.2)\n"
                "  --q-epsilon-decay X   per-update exploration multiplier (default 0.998)\n"
                "  --q-epsilon-min X     exploration floor (default 0.01)\n"
                "  --q-visits PATH       export Q-learning training visits and update counts as CSV\n"
                "  --q-td-trace PATH     stream every Q-learning TD update to CSV\n"
                "  --q-alpha X          Q-learning step size in (0,1] (default 0.1)\n"
                "  --q-reward-stats PATH export reward/TD dispersion by state/action/window\n");
    std::printf("  --q-transition-model PATH export empirical state transition matrix (batched Q-learning)\n");
    std::printf("  --q-policy PATH        evaluate a value_iteration.py policy alongside learners\n");
    std::printf("  --dqn-save PATH        save trained DQN weights and a checked .meta manifest\n"
                "  --dqn-load PATH        load DQN weights (matching .meta required)\n"
                "  --dqn-trace PATH       CSV of DQN evaluation states, actions, and all Q-values\n"
                "  --epsilon-trace PATH   CSV of epsilon-greedy evaluation states/actions/values\n"
                "  --q-action-trace PATH CSV of Q-learning evaluation states/actions/values\n");
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
    std::printf("  Launch params    : n_prompts=%d  seed=%llu  episodes=%d  mean_interarrival=%.3fs\n",
                a.n_prompts, (unsigned long long)a.seed, a.episodes,
                a.mean_interarrival);
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
    std::printf("                     least_loaded, energy_aware, consolidate_small\n");
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
        else if (s == "--debug") { a.debug = true; }
        else if (s == "--q-visits") { auto value = next("--q-visits"); if (!value) return false; a.q_visits = value; }
        else if (s == "--q-td-trace") { auto value = next("--q-td-trace"); if (!value) return false; a.q_td_trace = value; }
        else if (s == "--q-reward-stats") { auto value = next("--q-reward-stats"); if (!value) return false; a.q_reward_stats = value; }
        else if (s == "--q-transition-model") { auto value = next("--q-transition-model"); if (!value) return false; a.q_transition_model = value; }
        else if (s == "--q-policy") { auto value = next("--q-policy"); if (!value) return false; a.q_policy = value; }
        else if (s == "--dqn-save") { auto value = next("--dqn-save"); if (!value) return false; a.dqn_save = value; }
        else if (s == "--dqn-load") { auto value = next("--dqn-load"); if (!value) return false; a.dqn_load = value; }
        else if (s == "--dqn-trace") { auto value = next("--dqn-trace"); if (!value) return false; a.dqn_trace = value; }
        else if (s == "--epsilon-trace") { auto value = next("--epsilon-trace"); if (!value) return false; a.epsilon_trace = value; }
        else if (s == "--q-action-trace") { auto value = next("--q-action-trace"); if (!value) return false; a.q_action_trace = value; }
        else if (s == "--q-epsilon" || s == "--q-epsilon-decay" || s == "--q-epsilon-min" || s == "--q-alpha") {
            auto value = next(s.c_str());
            if (!value) return false;
            char* end = nullptr;
            double parsed = std::strtod(value, &end);
            if (end == value || *end != '\0' || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0 ||
                ((s == "--q-epsilon-decay" || s == "--q-alpha") && parsed == 0.0)) {
                std::fprintf(stderr, "invalid probability/multiplier for %s: %s\n", s.c_str(), value);
                return false;
            }
            if (s == "--q-alpha") a.q_alpha = parsed;
            else if (s == "--q-epsilon") a.q_epsilon = parsed;
            else if (s == "--q-epsilon-decay") a.q_epsilon_decay = parsed;
            else a.q_epsilon_min = parsed;
        }
        else if (s == "-n" || s == "--n-prompts") { auto v = next("--n-prompts"); if (!v) return false; a.n_prompts = std::atoi(v); }
        else if (s == "--seed") { auto v = next("--seed"); if (!v) return false; a.seed = std::strtoull(v, nullptr, 10); }
        else if (s == "--objective") { auto v = next("--objective"); if (!v) return false; a.objective = v; }
        else if (s == "--engine") { auto v = next("--engine"); if (!v) return false; a.engine = v; }
        else if (s == "--episodes") { auto v = next("--episodes"); if (!v) return false; a.episodes = std::atoi(v); }
        else if (s == "--mean-interarrival") {
            auto v = next("--mean-interarrival"); if (!v) return false;
            char* end = nullptr;
            double parsed = std::strtod(v, &end);
            if (end == v || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0) {
                std::fprintf(stderr, "--mean-interarrival must be a finite number greater than zero\n");
                return false;
            }
            a.mean_interarrival = parsed;
        }
        else if (s == "--gamma") {
            auto v = next("--gamma"); if (!v) return false;
            char* end = nullptr;
            double parsed = std::strtod(v, &end);
            if (end == v || *end != '\0' || !std::isfinite(parsed) || parsed <= 0.0 || parsed > 1.0) {
                std::fprintf(stderr, "--gamma must be a finite number in (0,1]\n");
                return false;
            }
            a.gamma = parsed;
        }
        else if (s == "--small-gpus") { auto v = next("--small-gpus"); if (!v) return false; a.small_gpus = std::atoi(v); }
        else if (s == "--medium-gpus") { auto v = next("--medium-gpus"); if (!v) return false; a.medium_gpus = std::atoi(v); }
        else if (s == "--large-gpus") { auto v = next("--large-gpus"); if (!v) return false; a.large_gpus = std::atoi(v); }
        else if (s == "--json") { auto v = next("--json"); if (!v) return false; a.json_out = v; }
        else { std::printf("unknown arg: %s\n", s.c_str()); usage(); return false; }
    }
    if (a.q_epsilon_min > a.q_epsilon) {
        std::fprintf(stderr, "--q-epsilon-min must not exceed --q-epsilon\n");
        return false;
    }
    if (!a.q_transition_model.empty() && a.engine != "batched") {
        std::fprintf(stderr, "--q-transition-model requires --engine batched\n");
        return false;
    }
    if (!a.q_transition_model.empty() && a.episodes == 0) {
        std::fprintf(stderr, "--q-transition-model requires --episodes > 0 to collect transition samples\n");
        return false;
    }
    if (!a.q_policy.empty() && a.engine != "batched") {
        std::fprintf(stderr, "--q-policy requires --engine batched\n");
        return false;
    }
    if (!a.dqn_load.empty() && !a.dqn_save.empty() && a.dqn_load == a.dqn_save) {
        std::fprintf(stderr, "--dqn-load and --dqn-save must use different paths when continuing training\n");
        return false;
    }
    if (!a.dqn_save.empty() && a.episodes == 0) {
        std::fprintf(stderr, "--dqn-save requires --episodes > 0 so the model is trained before saving\n");
        return false;
    }
    return true;
}

// Run a fresh episode (fresh GPU pool) for one allocator on the given prompts.
void print_debug(const Allocator& allocator, const RewardWeights& weights,
                 const std::vector<JobResult>& results, double residual_energy) {
    double latency_cost = 0.0, energy_cost = 0.0, quality_cost = 0.0;
    double floor_cost = 0.0, reward_sum = 0.0, attributed_energy = 0.0;
    int violations = 0;
    std::map<int, int> gpu_jobs;
    std::map<std::string, int> model_jobs;
    for (const auto& result : results) {
        latency_cost += weights.latency * result.latency;
        energy_cost += weights.energy * result.energy;
        quality_cost += weights.quality_loss * result.quality_loss;
        if (result.quality < weights.min_quality) {
            ++violations;
            floor_cost += weights.min_quality_penalty * (weights.min_quality - result.quality);
        }
        reward_sum += result.reward;
        attributed_energy += result.energy;
        ++gpu_jobs[result.gpu.id];
        ++model_jobs[result.llm.name];
    }
    const double count = static_cast<double>(std::max<size_t>(1, results.size()));
    std::printf("\n[DEBUG %s] completed=%zu effective_epsilon=%.6f configured_epsilon=%.6f weights=(latency=%.6g energy=%.6g quality=%.6g floor=%.6g penalty=%.6g)\n",
                allocator.name().c_str(), results.size(), allocator.evaluation_mode ? 0.0 : allocator.epsilon(), allocator.epsilon(),
                weights.latency, weights.energy, weights.quality_loss,
                weights.min_quality, weights.min_quality_penalty);
    std::printf("[DEBUG] mean cost: latency=%.6f energy=%.6f quality=%.6f floor=%.6f; mean reward=%.6f reconstruction_error=%.9g\n",
                latency_cost / count, energy_cost / count, quality_cost / count,
                floor_cost / count, reward_sum / count,
                (reward_sum + latency_cost + energy_cost + quality_cost + floor_cost) / count);
    std::printf("[DEBUG] attributed_energy=%.3fJ residual_system_energy=%.3fJ floor_violations=%d/%zu\n",
                attributed_energy, residual_energy, violations, results.size());
    std::printf("[DEBUG] placements by GPU:");
    for (const auto& entry : gpu_jobs) std::printf(" gpu%d=%d", entry.first, entry.second);
    std::printf("\n[DEBUG] placements by model:");
    for (const auto& entry : model_jobs) std::printf(" %s=%d", entry.first.c_str(), entry.second);
    std::printf("\n");
    if (allocator.is_learner() && !allocator.evaluation_mode && allocator.epsilon() > 0.0)
        std::printf("[DEBUG WARNING] evaluation retains nonzero exploration; repeated runs can choose different placements.\n");
}

AggregatedMetrics run_episode(const Args& a, const RewardWeights& weights,
                              const std::vector<LLM>& llm_pool, Allocator& alloc,
                              const std::vector<Prompt>& prompts,
                              std::vector<JobResult>* saved_results = nullptr) {
    alloc.reset_for_episode();
    alloc.evaluation_mode = true;
    std::vector<GPU> gpus = create_gpu_pool(a.small_gpus, a.medium_gpus, a.large_gpus);
    if (a.engine == "batched") {
        BatchedSimulator sim(gpus, llm_pool, BatchedConfig{weights});
        auto results = sim.run(alloc, prompts, false);
        if (saved_results) *saved_results = results;
        if (a.debug) std::printf("\n[DEBUG physics] active=%.6f idle=%.6f sleep=%.6f wake=%.6f accounting_error=%.9gJ; evaluation frozen\n",
                                sim.active_energy, sim.idle_energy, sim.sleep_energy,
                                sim.wake_energy, sim.accounting_error);
        if (a.debug) {
            double transition_energy = 0.0, transition_reward = 0.0, reported_reward = 0.0;
            for (const auto& transition : sim.arrival_transitions) {
                transition_energy += transition.energy;
                transition_reward += transition.reward;
            }
            for (const auto& result : results) reported_reward += result.reward;
            std::printf("[DEBUG arrival transitions] count=%zu energy=%.6fJ reward=%.6f reporting_reward_error=%.9g terminal=%s\n",
                        sim.arrival_transitions.size(), transition_energy, transition_reward,
                        transition_reward - reported_reward,
                        !sim.arrival_transitions.empty() && sim.arrival_transitions.back().terminal ? "yes" : "no");
        }
        if (a.debug) print_debug(alloc, weights, results, sim.total_idle_energy);
        alloc.evaluation_mode = false;
        return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped);
    }
    SimulatorConfig scfg;
    scfg.n_prompts = a.n_prompts;
    scfg.reward_weights = weights;
    Simulator sim(gpus, llm_pool, scfg);
    auto results = sim.run(alloc, prompts, false);
    if (saved_results) *saved_results = results;
    if (a.debug) print_debug(alloc, weights, results, sim.total_idle_energy);
    alloc.evaluation_mode = false;
    return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped);
}

bool better_for_objective(const AggregatedMetrics& lhs, const AggregatedMetrics& rhs,
                          const std::string& objective) {
    if (objective == "energy_q93") {
        constexpr double floor = 0.93;
        const bool lhs_feasible = lhs.mean_quality + 1e-12 >= floor;
        const bool rhs_feasible = rhs.mean_quality + 1e-12 >= floor;
        if (lhs_feasible != rhs_feasible) return lhs_feasible;
        if (lhs_feasible) return lhs.system_energy < rhs.system_energy;
        if (lhs.mean_quality != rhs.mean_quality) return lhs.mean_quality > rhs.mean_quality;
        return lhs.system_energy < rhs.system_energy;
    }
    return lhs.mean_reward > rhs.mean_reward;
}

void print_report(const std::vector<AggregatedMetrics>& metrics, const std::string& objective) {
    std::vector<const AggregatedMetrics*> sorted;
    for (const auto& m : metrics) sorted.push_back(&m);
    std::stable_sort(sorted.begin(), sorted.end(),
        [&](const AggregatedMetrics* a, const AggregatedMetrics* b) {
            return better_for_objective(*a, *b, objective);
        });

    std::printf("\n  Objective: %s\n", objective.c_str());
    if (objective == "energy_q93")
        std::printf("  Constraint: mean quality >= 0.930; feasible policies ranked by minimum system energy\n");
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
    f << "  \"mean_interarrival\": " << a.mean_interarrival << ",\n";
    f << "  \"gamma\": " << a.gamma << ",\n";
    f << "  \"q_alpha\": " << a.q_alpha << ",\n";
    f << "  \"q_exploration\": {\"initial\": " << a.q_epsilon
      << ", \"decay_per_update\": " << a.q_epsilon_decay
      << ", \"minimum\": " << a.q_epsilon_min << "},\n";
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

    // Best and worst under the active objective, plus totals.
    const AggregatedMetrics* best = &metrics[0];
    const AggregatedMetrics* worst = &metrics[0];
    int total_jobs = 0, total_dropped = 0;
    for (const auto& m : metrics) {
        if (better_for_objective(m, *best, a.objective)) best = &m;
        if (better_for_objective(*worst, m, a.objective)) worst = &m;
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
            if (!best_learner || better_for_objective(m, *best_learner, a.objective)) best_learner = &m;
        } else {
            if (!best_heur || better_for_objective(m, *best_heur, a.objective)) best_heur = &m;
        }
    }

    std::printf("  Status           : completed successfully\n");
    std::printf("  Engine/objective : %s / %s\n", a.engine.c_str(), a.objective.c_str());
    if (a.objective == "energy_q93")
        std::printf("  Constraint       : mean quality >= 0.930; minimize system energy\n");
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
        if (better_for_objective(*best_learner, *best_heur, a.objective)) {
            if (a.objective == "energy_q93") {
                std::printf("  Verdict          : best learner wins the constrained comparison\n");
            } else {
            std::printf("  Verdict          : the best learner BEAT the best heuristic "
                        "(by %.3f reward).\n",
                        best_learner->mean_reward - best_heur->mean_reward);
            }
        } else {
            if (a.objective == "energy_q93") {
                std::printf("  Verdict          : best heuristic wins the constrained comparison\n");
            } else {
            std::printf("  Verdict          : no learner beat the best heuristic "
                        "(gap %.3f reward)%s.\n",
                        best_heur->mean_reward - best_learner->mean_reward,
                        a.episodes > 0 ? "" : " — try --episodes > 0 to train them");
            }
        }
    }
    std::printf("%s\n", bar.c_str());
}

// Evaluate every allocator under each objective profile on the SAME prompt
// stream, and report which policy wins each profile. Rewards are NOT comparable
// across profiles (different weightings), so we only pick the best WITHIN each.
void print_profile_winners(const Args& a, const std::vector<LLM>& llm_pool,
                           const std::vector<Prompt>& prompts,
                           const std::map<std::string, std::vector<JobResult>>& cached_results,
                           const std::vector<AggregatedMetrics>& cached_metrics) {
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
        if (a.debug) std::printf("\n[DEBUG] profile=%s: rescoring cached, frozen evaluation results.\n", prof);
        RewardWeights w;
        try {
            w = RewardWeights::profile(prof);
        } catch (const std::exception&) {
            continue;
        }

        const AggregatedMetrics* best = nullptr;
        AggregatedMetrics best_copy;
        auto consider = [&](const AggregatedMetrics& original) {
            auto results = cached_results.at(original.allocator_name);
            for (auto& result : results) {
                double penalty = w.min_quality_penalty * std::max(0.0, w.min_quality - result.quality);
                result.reward = -(w.latency * result.latency + w.energy * result.energy +
                                  w.quality_loss * result.quality_loss + penalty);
            }
            AggregatedMetrics m = aggregate(original.allocator_name, results, original.idle_energy, original.dropped);
            if (!best || m.mean_reward > best_copy.mean_reward) {
                best_copy = m;
                best = &best_copy;
            }
        };
        for (const auto& original : cached_metrics) consider(original);

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
    auto q_learning = std::make_unique<QLearningAllocator>(
        a.q_alpha, a.gamma, a.q_epsilon, a.q_epsilon_decay,
        a.q_epsilon_min, 5, 3, 2, 0.0, a.seed);
    q_learning->use_model_only_actions();
    learners.push_back(std::move(q_learning));
    auto dqn = std::make_unique<DQNAllocator>(
        0.3, 0.995, 0.01, a.gamma, 1e-3, 64, 5000, 50, 64, 0.0, a.seed);
    if (!a.dqn_load.empty()) {
        try {
            dqn->load_model(a.dqn_load, a.engine);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "cannot configure DQN model load: %s\n", e.what());
            return 1;
        }
    }
    learners.push_back(std::move(dqn));

    std::vector<std::unique_ptr<Allocator>> fixed_policies;
    if (!a.q_policy.empty()) {
        std::ifstream policy_file(a.q_policy);
        if (!policy_file) { std::fprintf(stderr, "cannot read policy file %s\n", a.q_policy.c_str()); return 1; }
        auto policy = std::make_unique<QLearningAllocator>(
            a.q_alpha, a.gamma, 0.0, 1.0, 0.0, 5, 3, 2, 0.0, a.seed);
        policy->use_model_only_actions();
        try {
            policy->load_fixed_policy(policy_file);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "invalid --q-policy file: %s\n", e.what());
            return 1;
        }
        fixed_policies.push_back(std::move(policy));
    }

    // --- Train learners if requested ---
    std::ofstream td_trace;
    if (!a.q_td_trace.empty()) {
        td_trace.open(a.q_td_trace);
        if (!td_trace) { std::fprintf(stderr, "cannot write %s\n", a.q_td_trace.c_str()); return 1; }
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<QLearningAllocator*>(learner.get())) tabular->set_td_trace(&td_trace);
    }
    if (a.episodes > 0) {
        std::printf("\n============================================================\n");
        std::printf("  Training learning allocators (%d episodes, %s objective)\n",
                    a.episodes, a.objective.c_str());
        std::printf("============================================================\n");
        for (auto& learner : learners) {
            std::printf("\n--- %s ---\n", learner->name().c_str());
            train_learner(*learner, a.episodes, a.n_prompts, a.seed,
                          a.small_gpus, a.medium_gpus, a.large_gpus,
                          a.objective, true, a.engine, a.mean_interarrival);
        }
    }

    if (!a.dqn_save.empty()) {
        try {
            for (auto& learner : learners)
                if (auto* tabular = dynamic_cast<DQNAllocator*>(learner.get()))
                    tabular->save_model(a.dqn_save, a.engine);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "failed to save DQN model: %s\n", e.what());
            return 1;
        }
        std::printf("[INFO] DQN model weights and manifest saved to %s[.meta]\n", a.dqn_save.c_str());
    }

    std::ofstream dqn_trace;
    if (!a.dqn_trace.empty()) {
        dqn_trace.open(a.dqn_trace);
        if (!dqn_trace) { std::fprintf(stderr, "cannot write %s\n", a.dqn_trace.c_str()); return 1; }
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<DQNAllocator*>(learner.get())) tabular->set_action_trace(&dqn_trace, a.engine);
    }
    std::ofstream epsilon_trace, q_action_trace;
    if (!a.epsilon_trace.empty()) {
        epsilon_trace.open(a.epsilon_trace);
        if (!epsilon_trace) { std::fprintf(stderr, "cannot write %s\n", a.epsilon_trace.c_str()); return 1; }
        for (auto& learner : learners)
            if (auto* bandit = dynamic_cast<EpsilonGreedyAllocator*>(learner.get())) bandit->set_action_trace(&epsilon_trace);
    }
    if (!a.q_action_trace.empty()) {
        q_action_trace.open(a.q_action_trace);
        if (!q_action_trace) { std::fprintf(stderr, "cannot write %s\n", a.q_action_trace.c_str()); return 1; }
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<QLearningAllocator*>(learner.get()))
                if (tabular->name() == "q_learning") tabular->set_action_trace(&q_action_trace);
    }

    // --- Final evaluation on a common prompt stream ---
    if (!a.q_td_trace.empty()) {
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<QLearningAllocator*>(learner.get())) tabular->set_td_trace(nullptr);
        td_trace.close();
        if (!td_trace) { std::fprintf(stderr, "failed to write %s\n", a.q_td_trace.c_str()); return 1; }
        std::printf("[INFO] Q-learning TD updates written to %s\n", a.q_td_trace.c_str());
    }
    std::printf("[Q-learning] alpha=%.6f\n", a.q_alpha);
    for (const auto& learner : learners) {
        if (const auto* tabular = dynamic_cast<const QLearningAllocator*>(learner.get())) {
            std::printf("[Q-learning] states=%d actions=%d (model size; GPU=least_loaded)\n",
                        tabular->state_space_size(), tabular->action_space_size());
        }
    }
    if (!a.q_reward_stats.empty()) {
        std::ofstream output(a.q_reward_stats);
        if (!output) { std::fprintf(stderr, "cannot write %s\n", a.q_reward_stats.c_str()); return 1; }
        for (const auto& learner : learners)
            if (const auto* tabular = dynamic_cast<const QLearningAllocator*>(learner.get())) tabular->write_reward_stats(output);
        output.close();
        if (!output) { std::fprintf(stderr, "failed to write %s\n", a.q_reward_stats.c_str()); return 1; }
        std::printf("[INFO] Q-learning reward dispersion written to %s\n", a.q_reward_stats.c_str());
    }
    if (!a.q_transition_model.empty()) {
        std::ofstream output(a.q_transition_model);
        if (!output) { std::fprintf(stderr, "cannot write %s\n", a.q_transition_model.c_str()); return 1; }
        for (const auto& learner : learners)
            if (const auto* tabular = dynamic_cast<const QLearningAllocator*>(learner.get()))
                tabular->write_transition_model(output);
        output.close();
        if (!output) { std::fprintf(stderr, "failed to write %s\n", a.q_transition_model.c_str()); return 1; }
        std::printf("[INFO] empirical Q-learning transition model written to %s\n", a.q_transition_model.c_str());
    }
    std::printf("\n[Q-learning exploration] initial=%.6f decay/update=%.8f floor=%.6f\n",
                a.q_epsilon, a.q_epsilon_decay, a.q_epsilon_min);
    if (!a.q_visits.empty()) {
        std::ofstream output(a.q_visits);
        if (!output) { std::fprintf(stderr, "cannot write %s\n", a.q_visits.c_str()); return 1; }
        for (const auto& learner : learners)
            if (const auto* tabular = dynamic_cast<const QLearningAllocator*>(learner.get())) tabular->write_visits(output);
        output.close();
        if (!output) { std::fprintf(stderr, "failed to write %s\n", a.q_visits.c_str()); return 1; }
        std::printf("[INFO] Q-learning training coverage written to %s\n", a.q_visits.c_str());
    }
    std::printf("\n%s\n", std::string(78, '=').c_str());
    std::printf("  EVALUATION — all allocators on one shared stream of %d prompts\n",
                a.n_prompts);
    std::printf("%s\n", std::string(78, '=').c_str());

    PromptGenerator gen(0.5, 0.2, a.mean_interarrival, 256, a.seed + 9999);
    std::vector<Prompt> prompts = gen.generate(a.n_prompts);

    std::vector<AggregatedMetrics> all_metrics;
    std::map<std::string, std::vector<JobResult>> cached_results;

    auto heuristics = all_heuristic_allocators(a.seed);
    const size_t total_allocs = heuristics.size() + learners.size() + fixed_policies.size();
    size_t done = 0;

    auto run_and_log = [&](Allocator& alloc, const char* kind) {
        std::printf("  [%2zu/%2zu] %-13s %-14s ... ",
                    ++done, total_allocs, kind, alloc.name().c_str());
        std::fflush(stdout);
        AggregatedMetrics m = run_episode(a, weights, llm_pool, alloc, prompts, &cached_results[alloc.name()]);
        std::printf("reward %8.3f | sys_energy %8.0fJ | quality %.3f | drop %d\n",
                    m.mean_reward, m.system_energy, m.mean_quality, m.dropped);
        std::fflush(stdout);
        all_metrics.push_back(std::move(m));
    };

    for (auto& alloc : heuristics) run_and_log(*alloc, "heuristic");
    for (auto& learner : learners) run_and_log(*learner, "learner");
    for (auto& policy : fixed_policies) run_and_log(*policy, "model_policy");
    for (const auto& policy : fixed_policies) {
        if (const auto* tabular = dynamic_cast<const QLearningAllocator*>(policy.get()))
            std::printf("[INFO] value_iteration policy fallback actions (unseen states): %zu\n",
                        tabular->fixed_policy_misses());
    }
    if (!a.dqn_trace.empty()) {
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<DQNAllocator*>(learner.get())) tabular->set_action_trace(nullptr, a.engine);
        dqn_trace.close();
        if (!dqn_trace) { std::fprintf(stderr, "failed to write %s\n", a.dqn_trace.c_str()); return 1; }
        std::printf("[INFO] DQN decision and Q-value trace written to %s\n", a.dqn_trace.c_str());
    }
    if (!a.epsilon_trace.empty()) {
        for (auto& learner : learners)
            if (auto* bandit = dynamic_cast<EpsilonGreedyAllocator*>(learner.get())) bandit->set_action_trace(nullptr);
        epsilon_trace.close();
        if (!epsilon_trace) { std::fprintf(stderr, "failed to write %s\n", a.epsilon_trace.c_str()); return 1; }
        std::printf("[INFO] epsilon-greedy decision trace written to %s\n", a.epsilon_trace.c_str());
    }
    if (!a.q_action_trace.empty()) {
        for (auto& learner : learners)
            if (auto* tabular = dynamic_cast<QLearningAllocator*>(learner.get()))
                if (tabular->name() == "q_learning") tabular->set_action_trace(nullptr);
        q_action_trace.close();
        if (!q_action_trace) { std::fprintf(stderr, "failed to write %s\n", a.q_action_trace.c_str()); return 1; }
        std::printf("[INFO] Q-learning decision trace written to %s\n", a.q_action_trace.c_str());
    }

    print_report(all_metrics, a.objective);

    if (!a.json_out.empty()) {
        write_json(a.json_out, a, all_metrics);
        std::printf("[INFO] metrics written to %s\n", a.json_out.c_str());
    }

    // --- Final run summary: what happened, did it succeed, what we got ---
    print_run_summary(a, all_metrics);

    // --- Cross-profile winners: which policy wins energy/balanced/quality/latency ---
    print_profile_winners(a, llm_pool, prompts, cached_results, all_metrics);
    return 0;
}
