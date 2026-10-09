// Deterministic parity check against the Python reference values.
#include <cstdio>
#include <vector>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;

namespace {
// A fixed allocator that always returns (gpus[0], pool[0]).
class FixedAllocator : public Allocator {
public:
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>& gpus, const std::vector<LLM>& pool,
        double, const EnvObservation*) override {
        return AllocationDecision(&gpus[0], pool[0]);
    }
    std::string name() const override { return "fixed"; }
};
}  // namespace

int main() {
    auto pool = default_llm_pool();

    std::printf("== physics helpers ==\n");
    std::vector<std::pair<int, Size>> gcfg = {{0, Size::SMALL}, {1, Size::MEDIUM}, {2, Size::LARGE}};
    for (auto& gc : gcfg) {
        GPU g(gc.first, gc.second);
        for (const auto& llm : pool) {
            if (!llm.fits_on(g)) continue;
            Prompt p; p.id = 0; p.arrival_time = 0.0; p.difficulty = 0.6; p.input_tokens = 200;
            int ot = compute_output_tokens(p, llm);
            double pt = compute_processing_time(p, llm, g);
            double en = compute_energy(pt, llm, g);
            double q = compute_quality(p, llm);
            std::printf("g=%-6s llm=%-8s out=%d proc=%.6f energy=%.6f quality=%.6f\n",
                        size_name(g.vram), llm.name.c_str(), ot, pt, en, q);
        }
    }

    std::printf("== batched physics ==\n");
    {
        GPU g(0, Size::LARGE);
        const LLM& llm = pool[2];
        for (int b : {1, 4, 16, 64}) {
            std::printf("b=%3d step=%.6f tps=%.6f power=%.6f\n", b,
                        decode_step_duration(llm, g, b),
                        batch_tokens_per_sec(llm, g, b),
                        step_power(g, b, 64));
        }
    }

    std::printf("== kv_cache_slots ==\n");
    for (Size gsize : {Size::SMALL, Size::MEDIUM, Size::LARGE}) {
        GPU g(0, gsize);
        std::printf("%s ", size_name(gsize));
        for (const auto& m : pool) if (m.fits_on(g)) std::printf("%d ", g.kv_cache_slots(m));
        std::printf("\n");
    }

    std::printf("== reward weights ==\n");
    for (const char* name : {"energy", "balanced", "quality", "latency"}) {
        auto w = RewardWeights::profile(name);
        std::printf("%s %g %g %g %g %g\n", name, w.latency, w.energy,
                    w.quality_loss, w.min_quality, w.min_quality_penalty);
    }

    std::printf("== single static job reward (fixed) ==\n");
    {
        SimulatorConfig cfg;
        cfg.reward_weights = RewardWeights::profile("energy");
        Simulator sim(create_gpu_pool(1, 1, 1), pool, cfg);
        Prompt p; p.id = 0; p.arrival_time = 0.0; p.difficulty = 0.6; p.input_tokens = 200;
        FixedAllocator a;
        auto res = sim.run(a, {p});
        const auto& r = res[0];
        std::printf("latency=%.6f proc=%.6f energy=%.6f quality=%.6f reward=%.6f\n",
                    r.latency, r.processing_time, r.energy, r.quality, r.reward);
    }
    return 0;
}
