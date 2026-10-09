#include "llm_alloc/allocators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace llm_alloc {

namespace {
// Seed a mt19937_64 either from a given seed or random_device.
std::mt19937_64 make_rng(std::optional<uint64_t> seed) {
    if (seed.has_value()) return std::mt19937_64(*seed);
    std::random_device rd;
    return std::mt19937_64((static_cast<uint64_t>(rd()) << 32) ^ rd());
}

// Pick a uniformly-random element index in [0, n).
size_t rand_index(std::mt19937_64& rng, size_t n) {
    std::uniform_int_distribution<size_t> dist(0, n - 1);
    return dist(rng);
}
}  // namespace

std::vector<AllocationDecision> valid_pairs(std::vector<GPU>& gpus,
                                            const std::vector<LLM>& llm_pool) {
    std::vector<AllocationDecision> pairs;
    for (GPU& g : gpus) {
        for (const LLM& m : llm_pool) {
            if (m.fits_on(g)) pairs.emplace_back(&g, m);
        }
    }
    return pairs;
}

// --- RandomAllocator ---
RandomAllocator::RandomAllocator(std::optional<uint64_t> seed) : rng_(make_rng(seed)) {}

std::optional<AllocationDecision> RandomAllocator::allocate(
    const Prompt&, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    auto pairs = valid_pairs(gpus, llm_pool);
    if (pairs.empty()) return std::nullopt;
    return pairs[rand_index(rng_, pairs.size())];
}

// --- AlwaysSmall ---
AlwaysSmallAllocator::AlwaysSmallAllocator(std::optional<uint64_t> seed) : rng_(make_rng(seed)) {}

std::optional<AllocationDecision> AlwaysSmallAllocator::allocate(
    const Prompt&, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    const LLM* small = nullptr;
    for (const LLM& m : llm_pool) {
        if (m.size == Size::SMALL) { small = &m; break; }
    }
    if (!small) return std::nullopt;
    std::vector<GPU*> ok;
    for (GPU& g : gpus) if (small->fits_on(g)) ok.push_back(&g);
    if (ok.empty()) return std::nullopt;
    return AllocationDecision(ok[rand_index(rng_, ok.size())], *small);
}

// --- AlwaysLarge ---
AlwaysLargeAllocator::AlwaysLargeAllocator(std::optional<uint64_t> seed) : rng_(make_rng(seed)) {}

std::optional<AllocationDecision> AlwaysLargeAllocator::allocate(
    const Prompt&, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    const LLM* large = nullptr;
    for (const LLM& m : llm_pool) {
        if (m.size == Size::LARGE) { large = &m; break; }
    }
    if (!large) return std::nullopt;
    std::vector<GPU*> ok;
    for (GPU& g : gpus) if (large->fits_on(g)) ok.push_back(&g);
    if (ok.empty()) return std::nullopt;
    return AllocationDecision(ok[rand_index(rng_, ok.size())], *large);
}

// --- DifficultyAllocator ---
DifficultyAllocator::DifficultyAllocator(std::optional<uint64_t> seed) : rng_(make_rng(seed)) {}

std::optional<AllocationDecision> DifficultyAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    constexpr double t0 = 0.35, t1 = 0.70;
    Size target;
    if (prompt.difficulty < t0) target = Size::SMALL;
    else if (prompt.difficulty < t1) target = Size::MEDIUM;
    else target = Size::LARGE;

    const LLM* chosen = nullptr;
    for (const LLM& m : llm_pool) {
        if (m.size == target) { chosen = &m; break; }
    }
    if (!chosen) return std::nullopt;
    std::vector<GPU*> ok;
    for (GPU& g : gpus) if (chosen->fits_on(g)) ok.push_back(&g);
    if (ok.empty()) return std::nullopt;
    return AllocationDecision(ok[rand_index(rng_, ok.size())], *chosen);
}

// --- LeastLoadedAllocator ---
std::optional<AllocationDecision> LeastLoadedAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    // Sort GPU pointers by earliest availability.
    std::vector<GPU*> sorted_gpus;
    for (GPU& g : gpus) sorted_gpus.push_back(&g);
    std::stable_sort(sorted_gpus.begin(), sorted_gpus.end(),
                     [](const GPU* a, const GPU* b) { return a->busy_until < b->busy_until; });

    for (GPU* gpu : sorted_gpus) {
        // Smallest capable LLM that fits with capability >= difficulty*0.8.
        std::vector<const LLM*> fitting;
        for (const LLM& m : llm_pool) {
            if (m.fits_on(*gpu) && m.capability >= prompt.difficulty * 0.8) {
                fitting.push_back(&m);
            }
        }
        std::stable_sort(fitting.begin(), fitting.end(),
                         [](const LLM* a, const LLM* b) {
                             return size_value(a->size) < size_value(b->size);
                         });
        if (!fitting.empty()) return AllocationDecision(gpu, *fitting[0]);
    }
    // Fallback: any valid pair on least-loaded GPU.
    for (GPU* gpu : sorted_gpus) {
        for (const LLM& m : llm_pool) {
            if (m.fits_on(*gpu)) return AllocationDecision(gpu, m);
        }
    }
    return std::nullopt;
}

// --- EnergyAwareAllocator ---
EnergyAwareAllocator::EnergyAwareAllocator(double quality_threshold)
    : quality_threshold_(quality_threshold) {}

std::optional<AllocationDecision> EnergyAwareAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    std::optional<AllocationDecision> best;
    double best_energy = std::numeric_limits<double>::infinity();

    for (GPU& gpu : gpus) {
        for (const LLM& llm : llm_pool) {
            if (!llm.fits_on(gpu)) continue;
            double gap = prompt.difficulty - llm.capability;
            double quality_est = std::max(0.0, 1.0 - 1.5 * std::pow(std::max(0.0, gap), 0.8));
            if (quality_est < quality_threshold_) continue;

            int est_output = std::max(16, static_cast<int>(
                prompt.input_tokens * 0.6 * (0.5 + 0.5 * llm.capability) *
                (0.5 + 0.5 * prompt.difficulty)));
            double vram_bonus = 1.0 + 0.15 * (size_value(gpu.vram) - size_value(llm.size));
            double proc_time = est_output / (llm.base_speed * vram_bonus);
            double power = llm.energy_per_token * llm.base_speed * gpu.power_multiplier();
            double energy = power * proc_time;
            if (energy < best_energy) {
                best_energy = energy;
                best = AllocationDecision(&gpu, llm);
            }
        }
    }
    return best;
}

// --- all_heuristic_allocators ---
std::vector<std::unique_ptr<Allocator>> all_heuristic_allocators(uint64_t seed) {
    std::vector<std::unique_ptr<Allocator>> v;
    v.push_back(std::make_unique<RandomAllocator>(seed));
    v.push_back(std::make_unique<AlwaysSmallAllocator>(seed));
    v.push_back(std::make_unique<AlwaysLargeAllocator>(seed));
    v.push_back(std::make_unique<DifficultyAllocator>(seed));
    v.push_back(std::make_unique<LeastLoadedAllocator>());
    v.push_back(std::make_unique<EnergyAwareAllocator>());
    return v;
}

}  // namespace llm_alloc
