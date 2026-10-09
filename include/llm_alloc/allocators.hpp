// Heuristic allocators for the LLM-allocation problem.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "llm_alloc/simulator.hpp"

namespace llm_alloc {

// Return all (GPU*, LLM) pairs where the model fits on the GPU.
std::vector<AllocationDecision> valid_pairs(std::vector<GPU>& gpus,
                                            const std::vector<LLM>& llm_pool);

// --- RandomAllocator ---
class RandomAllocator : public Allocator {
public:
    explicit RandomAllocator(std::optional<uint64_t> seed = std::nullopt);
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "random"; }
private:
    std::mt19937_64 rng_;
};

// --- AlwaysSmall ---
class AlwaysSmallAllocator : public Allocator {
public:
    explicit AlwaysSmallAllocator(std::optional<uint64_t> seed = std::nullopt);
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "always_small"; }
private:
    std::mt19937_64 rng_;
};

// --- AlwaysLarge ---
class AlwaysLargeAllocator : public Allocator {
public:
    explicit AlwaysLargeAllocator(std::optional<uint64_t> seed = std::nullopt);
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "always_large"; }
private:
    std::mt19937_64 rng_;
};

// --- DifficultyAllocator ---
class DifficultyAllocator : public Allocator {
public:
    explicit DifficultyAllocator(std::optional<uint64_t> seed = std::nullopt);
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "difficulty"; }
private:
    std::mt19937_64 rng_;
};

// --- LeastLoadedAllocator ---
class LeastLoadedAllocator : public Allocator {
public:
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "least_loaded"; }
};

// --- EnergyAwareAllocator ---
class EnergyAwareAllocator : public Allocator {
public:
    explicit EnergyAwareAllocator(double quality_threshold = 0.7);
    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "energy_aware"; }
private:
    double quality_threshold_;
};

// Convenience: all heuristic allocators.
std::vector<std::unique_ptr<Allocator>> all_heuristic_allocators(uint64_t seed = 42);

}  // namespace llm_alloc
