// Benchmark runner: runs a single configuration and exports results to JSON.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;
using json = nlohmann::json;

namespace {

struct Config {
    std::string description;
    std::string engine = "static";
    int n_prompts = 200;
    int episodes = 100;
    std::string objective = "energy";
    int n_small = 2, n_medium = 2, n_large = 1;
    int dqn_hidden = 64;
    int dqn_batch_size = 64;
    int dqn_replay_size = 5000;
    double beta_d = 0.0;
    uint64_t seed = 42;
};

std::map<std::string, Config> make_configs() {
    std::map<std::string, Config> c;
    c["small_energy"] = Config{"Small-scale energy optimisation (quick validation)",
                               "static", 200, 100, "energy", 2, 2, 1, 64, 64, 5000, 0.0, 42};
    c["medium_energy"] = Config{"Medium-scale energy optimisation",
                                "static", 500, 300, "energy", 3, 3, 2, 128, 64, 10000, 0.0, 42};
    c["medium_balanced"] = Config{"Medium-scale balanced objective",
                                  "static", 500, 300, "balanced", 3, 3, 2, 128, 64, 10000, 0.0, 42};
    c["medium_quality"] = Config{"Medium-scale quality objective",
                                 "static", 500, 300, "quality", 3, 3, 2, 128, 64, 10000, 0.0, 42};
    c["medium_latency"] = Config{"Medium-scale latency objective",
                                 "static", 500, 300, "latency", 3, 3, 2, 128, 64, 10000, 0.0, 42};
    c["night_batched_energy"] = Config{"[batched] energy objective, deep training",
                                       "batched", 600, 300, "energy", 4, 4, 2, 128, 128, 20000, 0.0, 42};
    c["night_batched_balanced"] = Config{"[batched] balanced objective",
                                         "batched", 600, 300, "balanced", 4, 4, 2, 128, 128, 20000, 0.0, 42};
    c["night_batched_quality"] = Config{"[batched] quality objective",
                                        "batched", 600, 300, "quality", 4, 4, 2, 128, 128, 20000, 0.0, 42};
    c["night_batched_latency"] = Config{"[batched] latency objective",
                                        "batched", 600, 300, "latency", 4, 4, 2, 128, 128, 20000, 0.0, 42};
    return c;
}

void list_configs(const std::map<std::string, Config>& configs) {
    std::printf("\nAvailable benchmark configs:\n\n");
    for (const auto& kv : configs) {
        std::printf("  %-24s %s\n", kv.first.c_str(), kv.second.description.c_str());
    }
    std::printf("\n");
}

json curve_to_json(const TrainingCurve& c) {
    json j;
    j["episode"] = c.episode;
    j["mean_reward"] = c.mean_reward;
    j["system_energy"] = c.system_energy;
    j["mean_quality"] = c.mean_quality;
    j["epsilon"] = c.epsilon;
    return j;
}

std::string run_benchmark(const std::string& name, const Config& cfg,
                          const std::string& outdir) {
    std::printf("\n============================================================\n");
    std::printf("  BENCHMARK: %s\n  %s\n", name.c_str(), cfg.description.c_str());
    std::printf("  %d episodes x %d prompts, GPUs=(%d+%d+%d), objective=%s, engine=%s\n",
                cfg.episodes, cfg.n_prompts, cfg.n_small, cfg.n_medium, cfg.n_large,
                cfg.objective.c_str(), cfg.engine.c_str());
    std::printf("============================================================\n\n");

    RewardWeights weights = RewardWeights::profile(cfg.objective);
    std::vector<LLM> llm_pool = default_llm_pool();

    // --- Train learners ---
    std::vector<std::pair<std::string, std::unique_ptr<Allocator>>> learners;
    learners.emplace_back("epsilon_greedy",
        std::make_unique<EpsilonGreedyAllocator>(0.15, 0.999, 5, cfg.seed));
    learners.emplace_back("q_learning",
        std::make_unique<QLearningAllocator>(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, cfg.beta_d, cfg.seed));
    learners.emplace_back("dqn",
        std::make_unique<DQNAllocator>(0.3, 0.995, 0.01, 0.95, 1e-3, cfg.dqn_batch_size,
                                       cfg.dqn_replay_size, 50, cfg.dqn_hidden, cfg.beta_d, cfg.seed));

    json training_curves;
    for (auto& lr : learners) {
        std::printf("  Training %s...\n", lr.first.c_str());
        TrainingCurve curve = train_learner(
            *lr.second, cfg.episodes, cfg.n_prompts, cfg.seed,
            cfg.n_small, cfg.n_medium, cfg.n_large, cfg.objective, true, cfg.engine);
        training_curves[lr.first] = curve_to_json(curve);
    }

    // --- Final evaluation ---
    std::printf("\n  Final evaluation...\n");
    PromptGenerator gen(0.5, 0.2, 1.0, 256, cfg.seed + 9999);
    std::vector<Prompt> eval_prompts = gen.generate(cfg.n_prompts);

    auto make_sim_eval = [&](Allocator& alloc) -> AggregatedMetrics {
        std::vector<GPU> gpus = create_gpu_pool(cfg.n_small, cfg.n_medium, cfg.n_large);
        if (cfg.engine == "batched") {
            BatchedSimulator sim(gpus, llm_pool, BatchedConfig{weights});
            auto results = sim.run(alloc, eval_prompts, false);
            return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped, &gpus);
        }
        SimulatorConfig scfg;
        scfg.n_prompts = cfg.n_prompts;
        scfg.reward_weights = weights;
        Simulator sim(gpus, llm_pool, scfg);
        auto results = sim.run(alloc, eval_prompts, false);
        return aggregate(alloc.name(), results, sim.total_idle_energy, sim.dropped, &gpus);
    };

    std::vector<AggregatedMetrics> final_metrics;
    auto heuristics = all_heuristic_allocators(cfg.seed);
    for (auto& h : heuristics) final_metrics.push_back(make_sim_eval(*h));
    for (auto& lr : learners) final_metrics.push_back(make_sim_eval(*lr.second));

    std::stable_sort(final_metrics.begin(), final_metrics.end(),
        [](const AggregatedMetrics& a, const AggregatedMetrics& b) {
            return a.mean_reward > b.mean_reward;
        });

    json fm = json::array();
    for (const AggregatedMetrics& m : final_metrics) {
        double util_sum = 0.0;
        for (const auto& kv : m.gpu_utilisation) util_sum += kv.second;
        double util_avg = util_sum / std::max<size_t>(1, m.gpu_utilisation.size());
        fm.push_back({
            {"allocator", m.allocator_name}, {"n_jobs", m.n_jobs},
            {"mean_reward", m.mean_reward}, {"total_reward", m.total_reward},
            {"mean_latency", m.mean_latency}, {"p95_latency", m.p95_latency},
            {"total_energy", m.total_energy}, {"idle_energy", m.idle_energy},
            {"system_energy", m.system_energy}, {"mean_quality", m.mean_quality},
            {"gpu_utilisation_avg", util_avg},
            {"n_gpus", m.gpu_utilisation.size()}, {"dropped", m.dropped},
        });
    }

    json result;
    result["config_name"] = name;
    result["seed"] = cfg.seed;
    result["training_curves"] = training_curves;
    result["final_metrics"] = fm;

    std::string out_path = outdir + "/" + name + ".json";
    std::ofstream f(out_path);
    f << result.dump(2) << "\n";

    std::printf("\n============================================================\n");
    std::printf("  DONE: %s  ->  %s\n", name.c_str(), out_path.c_str());
    std::printf("============================================================\n");
    std::printf("\n  %-20s | %8s | %10s | %7s\n", "Allocator", "reward", "sys_energy", "quality");
    for (const AggregatedMetrics& m : final_metrics) {
        std::printf("  %-20s | %8.3f | %10.0f | %.3f\n", m.allocator_name.c_str(),
                    m.mean_reward, m.system_energy, m.mean_quality);
    }
    return out_path;
}

}  // namespace

int main(int argc, char** argv) {
    auto configs = make_configs();
    std::string config_name, outdir = "benchmarks";
    bool run_all = false, list = false;

    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--list") list = true;
        else if (s == "--all") run_all = true;
        else if (s == "--config" && i + 1 < argc) config_name = argv[++i];
        else if (s == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (s == "-h" || s == "--help") {
            std::printf("Usage: llm_benchmark [--config NAME | --all | --list] [--outdir DIR]\n");
            return 0;
        }
    }

    if (list) { list_configs(configs); return 0; }

    // ensure outdir exists (best-effort via std::ofstream failing silently otherwise)
    std::string mkdir_cmd = "mkdir -p '" + outdir + "'";
    (void)std::system(mkdir_cmd.c_str());

    if (run_all) {
        for (const auto& kv : configs) run_benchmark(kv.first, kv.second, outdir);
        return 0;
    }
    if (!config_name.empty()) {
        auto it = configs.find(config_name);
        if (it == configs.end()) {
            std::printf("Unknown config '%s'. Use --list.\n", config_name.c_str());
            return 1;
        }
        run_benchmark(config_name, it->second, outdir);
        return 0;
    }
    std::printf("Usage: llm_benchmark [--config NAME | --all | --list] [--outdir DIR]\n");
    return 0;
}
