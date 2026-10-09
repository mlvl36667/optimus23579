// Event-driven continuous-batching simulator (realistic LLM serving model).
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "llm_alloc/simulator.hpp"

namespace llm_alloc {

// --- Throughput / energy physics for batched decoding ---
double decode_step_duration(const LLM& llm, const GPU& gpu, int batch_size);
double batch_tokens_per_sec(const LLM& llm, const GPU& gpu, int batch_size);
double step_power(const GPU& gpu, int batch_size, int max_batch);

struct BatchedConfig {
    RewardWeights reward_weights;
    double dt = 0.05;              // simulation tick length (seconds)
    double max_sim_time = 1e6;     // safety cap
    bool allow_model_switch = true;
    bool power_management = true;
    double sleep_timeout = 5.0;
};

class BatchedSimulator {
public:
    BatchedSimulator(std::vector<GPU> gpus, std::vector<LLM> llm_pool,
                     BatchedConfig config = BatchedConfig());

    std::vector<JobResult> run(Allocator& allocator,
                               std::vector<Prompt> prompts,
                               bool online_update = false);

    double total_idle_energy = 0.0;
    int dropped = 0;

    std::vector<GPU>& gpus() { return gpus_; }

private:
    std::vector<GPU> gpus_;
    std::vector<LLM> llm_pool_;
    BatchedConfig config_;
};

}  // namespace llm_alloc
