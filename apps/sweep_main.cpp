// Parameter sweeps for the LLM-allocation study.
// Each sweep varies a single environment/workload parameter and reports a
// converged-policy metric as a function of that parameter.
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;
using json = nlohmann::json;

namespace {

constexpr int EPISODES = 150;
constexpr int N_PROMPTS = 150;

struct GpuCfg { int s, m, l; };

std::unique_ptr<Allocator> make_learner(const std::string& name, uint64_t seed) {
    if (name == "q_learning")
        return std::make_unique<QLearningAllocator>(0.1, 0.95, 0.3, 0.998, 0.05, 5, 4, 3, 0.0, seed);
    if (name == "epsilon_greedy")
        return std::make_unique<EpsilonGreedyAllocator>(0.15, 0.999, 5, seed);
    if (name == "dqn")
        return std::make_unique<DQNAllocator>(0.3, 0.995, 0.01, 0.95, 1e-3, 32, 3000, 50, 64, 0.0, seed);
    return nullptr;
}

std::vector<Prompt> gen_prompts(int n, double interarrival, double difficulty, uint64_t seed) {
    PromptGenerator g(difficulty, 0.2, interarrival, 256, seed);
    return g.generate(n);
}

// Train a fresh learner at one operating point, then greedily evaluate it.
AggregatedMetrics train_and_eval(const std::string& learner_name,
                                 const std::string& objective, GpuCfg gpus_cfg,
                                 int n_prompts, double interarrival, double difficulty,
                                 double sleep_timeout, int episodes, uint64_t seed = 42) {
    std::vector<LLM> pool = default_llm_pool();
    RewardWeights weights = RewardWeights::profile(objective);
    auto alloc = make_learner(learner_name, seed);
    if (!alloc) return AggregatedMetrics{};

    for (int ep = 0; ep < episodes; ++ep) {
        std::vector<GPU> g = create_gpu_pool(gpus_cfg.s, gpus_cfg.m, gpus_cfg.l);
        BatchedConfig bc{weights};
        bc.sleep_timeout = sleep_timeout;
        BatchedSimulator sim(g, pool, bc);
        alloc->reset_for_episode();
        sim.run(*alloc, gen_prompts(n_prompts, interarrival, difficulty, seed + ep), true);
        alloc->end_episode();
    }

    // greedy eval
    std::vector<GPU> g = create_gpu_pool(gpus_cfg.s, gpus_cfg.m, gpus_cfg.l);
    BatchedConfig bc{weights};
    bc.sleep_timeout = sleep_timeout;
    BatchedSimulator sim(g, pool, bc);
    // force greedy: QLearning/DQN expose set via epsilon() but we need a setter;
    // QLearningAllocator has set_epsilon; for others epsilon decays near min anyway.
    if (auto* ql = dynamic_cast<QLearningAllocator*>(alloc.get())) ql->set_epsilon(0.0);
    auto results = sim.run(*alloc, gen_prompts(n_prompts, interarrival, difficulty, seed + 999), false);
    return aggregate(learner_name, results, sim.total_idle_energy, sim.dropped);
}

json sweep_arrival(const std::string& metric) {
    std::vector<double> interarrivals = {2.0, 1.0, 0.5, 0.3, 0.2, 0.12, 0.08};
    std::vector<double> rates;
    for (double ia : interarrivals) rates.push_back(1.0 / ia);
    std::vector<std::string> learners = {"q_learning", "dqn", "epsilon_greedy"};
    json series;
    for (const auto& ln : learners) {
        std::vector<double> ys;
        for (double ia : interarrivals) {
            auto m = train_and_eval(ln, "energy", {2, 2, 1}, N_PROMPTS, ia, 0.5, 5.0, EPISODES);
            ys.push_back(metric == "energy" ? m.system_energy / 1000.0 : m.mean_latency);
        }
        series[ln] = ys;
    }
    json j;
    j["sweep"] = "arrival_" + metric;
    j["xlabel"] = "Arrival rate (req/s)";
    j["ylabel"] = metric == "energy" ? "System energy (kJ)" : "Mean latency (s)";
    j["x"] = rates;
    j["series"] = series;
    return j;
}

json sweep_fleet() {
    std::vector<GpuCfg> fleets = {{1,1,1},{2,2,1},{3,3,2},{4,4,2},{6,4,2},{8,6,4}};
    std::vector<int> xs;
    std::vector<double> ys;
    for (const GpuCfg& f : fleets) {
        xs.push_back(f.s + f.m + f.l);
        auto m = train_and_eval("q_learning", "energy", f, N_PROMPTS, 0.3, 0.5, 5.0, EPISODES);
        ys.push_back(m.system_energy / std::max(1, m.n_jobs));
    }
    json j;
    j["sweep"] = "fleet_energy";
    j["xlabel"] = "Fleet size (number of GPUs)";
    j["ylabel"] = "Energy per request (J)";
    j["x"] = xs;
    j["series"] = {{"q_learning", ys}};
    return j;
}

json sweep_sleep() {
    std::vector<double> timeouts = {1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0};
    std::vector<double> energy;
    for (double st : timeouts) {
        auto m = train_and_eval("q_learning", "energy", {2, 2, 1}, N_PROMPTS, 0.2, 0.5, st, EPISODES);
        energy.push_back(m.system_energy / 1000.0);
    }
    json j;
    j["sweep"] = "sleep_energy";
    j["xlabel"] = "Sleep timeout (s)";
    j["ylabel"] = "System energy (kJ)";
    j["x"] = timeouts;
    j["series"] = {{"system_energy_kJ", energy}};
    return j;
}

json sweep_difficulty() {
    std::vector<double> diffs = {0.2, 0.35, 0.5, 0.65, 0.8, 0.95};
    std::vector<std::string> objectives = {"energy", "balanced", "quality", "latency"};
    json series;
    for (const auto& obj : objectives) {
        std::vector<double> ys;
        for (double d : diffs) {
            auto m = train_and_eval("q_learning", obj, {2, 2, 1}, N_PROMPTS, 0.5, d, 5.0, EPISODES);
            ys.push_back(m.mean_quality);
        }
        series[obj] = ys;
    }
    json j;
    j["sweep"] = "difficulty_quality";
    j["xlabel"] = "Mean prompt difficulty";
    j["ylabel"] = "Mean answer quality";
    j["x"] = diffs;
    j["series"] = series;
    return j;
}

json run_sweep(const std::string& name) {
    if (name == "arrival_energy") return sweep_arrival("energy");
    if (name == "arrival_latency") return sweep_arrival("latency");
    if (name == "fleet_energy") return sweep_fleet();
    if (name == "sleep_energy") return sweep_sleep();
    if (name == "difficulty_quality") return sweep_difficulty();
    return json();
}

}  // namespace

int main(int argc, char** argv) {
    std::string sweep, outdir = "sweeps";
    bool list = false;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        if (s == "--list") list = true;
        else if (s == "--sweep" && i + 1 < argc) sweep = argv[++i];
        else if (s == "--outdir" && i + 1 < argc) outdir = argv[++i];
        else if (s == "-h" || s == "--help") {
            std::printf("Usage: llm_sweep [--sweep NAME | --list] [--outdir DIR]\n");
            return 0;
        }
    }

    std::vector<std::string> names = {"arrival_energy", "arrival_latency",
                                      "fleet_energy", "sleep_energy", "difficulty_quality"};
    if (list) {
        for (const auto& n : names) std::printf("%s\n", n.c_str());
        return 0;
    }
    if (sweep.empty()) {
        std::printf("--sweep must be one of: ");
        for (const auto& n : names) std::printf("%s ", n.c_str());
        std::printf("\n");
        return 1;
    }

    (void)std::system(("mkdir -p '" + outdir + "'").c_str());
    std::printf("============================================================\n");
    std::printf("  SWEEP: %s  (episodes/point=%d, prompts/episode=%d)\n",
                sweep.c_str(), EPISODES, N_PROMPTS);
    std::printf("============================================================\n");

    json result = run_sweep(sweep);
    if (result.is_null()) {
        std::printf("unknown sweep: %s\n", sweep.c_str());
        return 1;
    }

    std::string out = outdir + "/" + sweep + ".json";
    std::ofstream f(out);
    f << result.dump(2) << "\n";
    std::printf("\n  DONE -> %s\n", out.c_str());
    return 0;
}
