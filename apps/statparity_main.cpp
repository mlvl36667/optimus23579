// Multi-seed statistical comparison for non-random heuristics.
#include <cmath>
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;

static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return s / v.size();
}
static double pstdev(const std::vector<double>& v) {
    double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / v.size());
}

int main() {
    auto pool = default_llm_pool();
    SimulatorConfig cfg;
    cfg.reward_weights = RewardWeights::profile("energy");

    auto run = [&](std::function<std::unique_ptr<Allocator>()> factory,
                   int n_seeds) {
        std::vector<double> rew, en, q;
        for (int s = 0; s < n_seeds; ++s) {
            auto gpus = create_gpu_pool(2, 2, 1);
            Simulator sim(gpus, pool, cfg);
            PromptGenerator gen(0.5, 0.2, 1.0, 256, static_cast<uint64_t>(s));
            auto prompts = gen.generate(200);
            auto alloc = factory();
            auto res = sim.run(*alloc, prompts);
            auto m = aggregate("x", res, sim.total_idle_energy);
            rew.push_back(m.mean_reward);
            en.push_back(m.system_energy);
            q.push_back(m.mean_quality);
        }
        return std::make_tuple(rew, en, q);
    };

    struct Case { const char* name; std::function<std::unique_ptr<Allocator>()> f; };
    std::vector<Case> cases = {
        {"least_loaded", []{ return std::make_unique<LeastLoadedAllocator>(); }},
        {"energy_aware", []{ return std::make_unique<EnergyAwareAllocator>(); }},
    };

    for (auto& c : cases) {
        auto [rew, en, q] = run(c.f, 30);
        std::printf("[CPP] %-13s reward mean=%.4f sd=%.4f | energy mean=%.1f sd=%.1f | "
                    "quality mean=%.4f sd=%.4f\n",
                    c.name, mean(rew), pstdev(rew), mean(en), pstdev(en),
                    mean(q), pstdev(q));
    }
    return 0;
}
