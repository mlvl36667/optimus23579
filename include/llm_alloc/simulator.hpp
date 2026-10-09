// Discrete-event simulation engine with cost/quality/energy reward model.
#pragma once

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "llm_alloc/domain.hpp"

namespace llm_alloc {

// ---------------------------------------------------------------------------
// Allocation decision
// ---------------------------------------------------------------------------
struct AllocationDecision {
    GPU* gpu = nullptr;   // non-owning pointer into the engine's GPU vector
    LLM llm;              // the chosen model (value copy — pool is small)

    AllocationDecision() = default;
    AllocationDecision(GPU* g, LLM m) : gpu(g), llm(std::move(m)) {}
};

// ---------------------------------------------------------------------------
// Environment observation — per-GPU runtime dynamics exposed to allocators
// ---------------------------------------------------------------------------
struct GPUObservation {
    int gpu_id = 0;
    Size vram = Size::SMALL;
    std::string power_state;                 // "sleep"|"idle"|"active"|"waking"
    std::optional<Size> loaded_llm_size;
    int active_count = 0;
    int capacity = 0;
    int waiting_count = 0;

    double batch_fill() const {
        return capacity > 0 ? static_cast<double>(active_count) / capacity : 0.0;
    }
};

struct EnvObservation {
    double time = 0.0;
    std::map<int, GPUObservation> gpus;

    const GPUObservation* get(int gpu_id) const {
        auto it = gpus.find(gpu_id);
        return it == gpus.end() ? nullptr : &it->second;
    }
};

// ---------------------------------------------------------------------------
// Opaque learner decision snapshot (for correct credit assignment)
// ---------------------------------------------------------------------------
struct DecisionSnapshot;  // defined in learning.hpp; carried via shared_ptr

// ---------------------------------------------------------------------------
// Allocator interface — any allocator implements this
// ---------------------------------------------------------------------------
class Allocator {
public:
    virtual ~Allocator() = default;

    // Pick a (GPU, LLM) pair for a prompt. Returns std::nullopt if no valid
    // assignment exists. env_state may be null (static engine).
    virtual std::optional<AllocationDecision> allocate(
        const Prompt& prompt, std::vector<GPU>& gpus,
        const std::vector<LLM>& llm_pool, double current_time,
        const EnvObservation* env_state) = 0;

    virtual std::string name() const = 0;

    // --- optional learner hooks (default no-ops) ---
    virtual bool is_learner() const { return false; }
    virtual void reset_for_episode() {}
    virtual void update(double /*reward*/) {}
    virtual void end_episode() {}
    virtual double epsilon() const { return 0.0; }

    // Correct (s,a,r,s') interface for the batched engine.
    virtual bool supports_observe_transition() const { return false; }
    virtual std::shared_ptr<DecisionSnapshot> pop_last_decision() { return nullptr; }
    virtual void observe_transition(const std::shared_ptr<DecisionSnapshot>& /*decision*/,
                                    double /*reward*/, const EnvObservation* /*next_state*/,
                                    bool /*done*/, std::optional<double> /*holding_time*/) {}
};

// ---------------------------------------------------------------------------
// Reward / physics model
// ---------------------------------------------------------------------------
struct RewardWeights {
    double latency = 1.0;
    double energy = 1.0;
    double quality_loss = 2.0;
    double min_quality = 0.0;
    double min_quality_penalty = 50.0;

    static RewardWeights profile(const std::string& name);
};

struct JobResult {
    Prompt prompt;
    GPU gpu;         // snapshot of the GPU at completion
    LLM llm;
    double start_time = 0.0;
    double end_time = 0.0;
    double latency = 0.0;
    double processing_time = 0.0;
    double energy = 0.0;
    double quality = 0.0;
    double quality_loss = 0.0;
    double reward = 0.0;
};

// Physics helpers (shared with the batched engine)
int compute_output_tokens(const Prompt& prompt, const LLM& llm);
double compute_processing_time(const Prompt& prompt, const LLM& llm, const GPU& gpu);
double compute_energy(double processing_time, const LLM& llm, const GPU& gpu);
double compute_quality(const Prompt& prompt, const LLM& llm);

// ---------------------------------------------------------------------------
// Static Simulator
// ---------------------------------------------------------------------------
struct SimulatorConfig {
    int n_prompts = 200;
    RewardWeights reward_weights;
    bool include_idle_energy = true;
};

class Simulator {
public:
    Simulator(std::vector<GPU> gpus, std::vector<LLM> llm_pool,
              SimulatorConfig config = SimulatorConfig());

    std::vector<JobResult> run(Allocator& allocator,
                               std::vector<Prompt> prompts,
                               bool online_update = false);

    // Populated after run()
    double total_idle_energy = 0.0;
    int dropped = 0;

    std::vector<GPU>& gpus() { return gpus_; }

private:
    void reset_gpus();
    double compute_idle_energy(const std::vector<JobResult>& results) const;
    JobResult process(const Prompt& prompt, GPU& gpu, const LLM& llm);

    std::vector<GPU> gpus_;
    std::vector<LLM> llm_pool_;
    SimulatorConfig config_;
};

}  // namespace llm_alloc
