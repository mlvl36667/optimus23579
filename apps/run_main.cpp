// Comparison runner: evaluate all allocators on the same prompt stream.
// Learning allocators (Q-learning, DQN) are optionally trained for --episodes
// episodes before the final comparison run.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

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

    std::printf("  Engine: %s\n", a.engine.c_str());

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
    PromptGenerator gen(0.5, 0.2, 1.0, 256, a.seed + 9999);
    std::vector<Prompt> prompts = gen.generate(a.n_prompts);

    std::vector<AggregatedMetrics> all_metrics;

    auto heuristics = all_heuristic_allocators(a.seed);
    for (auto& alloc : heuristics) {
        all_metrics.push_back(run_episode(a, weights, llm_pool, *alloc, prompts));
    }
    for (auto& learner : learners) {
        all_metrics.push_back(run_episode(a, weights, llm_pool, *learner, prompts));
    }

    print_report(all_metrics, a.objective);

    if (!a.json_out.empty()) {
        write_json(a.json_out, a, all_metrics);
        std::printf("[INFO] metrics written to %s\n", a.json_out.c_str());
    }
    return 0;
}
