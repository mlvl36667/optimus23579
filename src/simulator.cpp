#include "llm_alloc/simulator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llm_alloc {

// ---------------------------------------------------------------------------
// RewardWeights profiles
// ---------------------------------------------------------------------------
RewardWeights RewardWeights::profile(const std::string& name) {
    RewardWeights w;
    if (name == "energy") {
        w.latency = 0.05; w.energy = 1.0; w.quality_loss = 0.3; w.min_quality = 0.6;
    } else if (name == "balanced") {
        w.latency = 1.0; w.energy = 0.02; w.quality_loss = 10.0;
    } else if (name == "quality") {
        w.latency = 0.2; w.energy = 0.002; w.quality_loss = 30.0;
    } else if (name == "latency") {
        w.latency = 5.0; w.energy = 0.005; w.quality_loss = 1.0;
    } else {
        throw std::invalid_argument(
            "unknown objective '" + name +
            "'. choices: energy, balanced, quality, latency");
    }
    return w;
}

// ---------------------------------------------------------------------------
// Physics helpers
// ---------------------------------------------------------------------------
int compute_output_tokens(const Prompt& prompt, const LLM& llm) {
    double base = prompt.input_tokens * 0.6;
    double capability_factor = 0.5 + 0.5 * llm.capability;
    double difficulty_factor = 0.5 + 0.5 * prompt.difficulty;
    return std::max(16, static_cast<int>(base * capability_factor * difficulty_factor));
}

double compute_processing_time(const Prompt& prompt, const LLM& llm, const GPU& gpu) {
    int output_tokens = compute_output_tokens(prompt, llm);
    double vram_bonus = 1.0 + 0.15 * (size_value(gpu.vram) - size_value(llm.size));
    double effective_speed = llm.base_speed * vram_bonus;
    return output_tokens / effective_speed;
}

double compute_energy(double processing_time, const LLM& llm, const GPU& gpu) {
    double power = llm.energy_per_token * llm.base_speed * gpu.power_multiplier();
    return power * processing_time;
}

double compute_quality(const Prompt& prompt, const LLM& llm) {
    double gap = prompt.difficulty - llm.capability;
    if (gap <= 0) return 1.0;
    return std::max(0.0, 1.0 - 1.5 * std::pow(gap, 0.8));
}

// ---------------------------------------------------------------------------
// Simulator
// ---------------------------------------------------------------------------
Simulator::Simulator(std::vector<GPU> gpus, std::vector<LLM> llm_pool,
                     SimulatorConfig config)
    : gpus_(std::move(gpus)), llm_pool_(std::move(llm_pool)),
      config_(std::move(config)) {}

void Simulator::reset_gpus() {
    for (GPU& gpu : gpus_) {
        gpu.busy_until = 0.0;
        gpu.total_busy_time = 0.0;
        gpu.jobs_completed = 0;
    }
}

std::vector<JobResult> Simulator::run(Allocator& allocator,
                                      std::vector<Prompt> prompts,
                                      bool online_update) {
    reset_gpus();
    total_idle_energy = 0.0;
    std::vector<JobResult> results;
    int dropped_count = 0;

    for (const Prompt& prompt : prompts) {
        auto decision = allocator.allocate(prompt, gpus_, llm_pool_,
                                           prompt.arrival_time, nullptr);
        if (!decision.has_value() || decision->gpu == nullptr) {
            dropped_count++;
            continue;
        }
        GPU& gpu = *decision->gpu;
        const LLM& llm = decision->llm;

        if (!llm.fits_on(gpu)) {
            dropped_count++;
            continue;
        }

        JobResult result = process(prompt, gpu, llm);
        results.push_back(result);

        if (online_update && allocator.is_learner()) {
            allocator.update(result.reward);
        }
    }

    dropped = dropped_count;
    total_idle_energy = compute_idle_energy(results);
    return results;
}

double Simulator::compute_idle_energy(const std::vector<JobResult>& results) const {
    if (!config_.include_idle_energy || results.empty()) return 0.0;
    double horizon = 0.0;
    for (const JobResult& r : results) horizon = std::max(horizon, r.end_time);
    double total = 0.0;
    for (const GPU& gpu : gpus_) {
        double idle_time = std::max(0.0, horizon - gpu.total_busy_time);
        total += gpu.idle_power() * idle_time;
    }
    return total;
}

JobResult Simulator::process(const Prompt& prompt, GPU& gpu, const LLM& llm) {
    const RewardWeights& w = config_.reward_weights;

    double start_time = std::max(gpu.busy_until, prompt.arrival_time);
    double proc_time = compute_processing_time(prompt, llm, gpu);
    double end_time = start_time + proc_time;

    gpu.busy_until = end_time;
    gpu.total_busy_time += proc_time;
    gpu.jobs_completed += 1;

    double latency = end_time - prompt.arrival_time;
    double energy = compute_energy(proc_time, llm, gpu);
    double quality = compute_quality(prompt, llm);
    double quality_loss = 1.0 - quality;

    double quality_penalty = 0.0;
    if (w.min_quality > 0 && quality < w.min_quality) {
        quality_penalty = w.min_quality_penalty * (w.min_quality - quality);
    }
    double reward = -(w.latency * latency + w.energy * energy +
                      w.quality_loss * quality_loss + quality_penalty);

    JobResult r;
    r.prompt = prompt;
    r.gpu = gpu;
    r.llm = llm;
    r.start_time = start_time;
    r.end_time = end_time;
    r.latency = latency;
    r.processing_time = proc_time;
    r.energy = energy;
    r.quality = quality;
    r.quality_loss = quality_loss;
    r.reward = reward;
    return r;
}

}  // namespace llm_alloc
