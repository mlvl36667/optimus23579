// Metrics aggregation for simulation results.
#pragma once

#include <map>
#include <string>
#include <vector>

#include "llm_alloc/simulator.hpp"

namespace llm_alloc {

struct AggregatedMetrics {
    std::string allocator_name;
    int n_jobs = 0;
    double total_reward = 0.0;
    double mean_reward = 0.0;
    double mean_latency = 0.0;
    double p95_latency = 0.0;
    double total_energy = 0.0;      // active inference energy
    double idle_energy = 0.0;       // GPU idle energy
    double system_energy = 0.0;     // total_energy + idle_energy
    double mean_energy = 0.0;
    double mean_quality = 0.0;
    double mean_quality_loss = 0.0;
    std::map<int, double> gpu_utilisation;  // gpu_id -> fraction of sim time busy
    std::map<int, int> gpu_tiers;           // gpu_id -> VRAM tier (1=S,2=M,3=L)
    int dropped = 0;

    std::string summary_line() const;
};

// all_gpus is optional (pass nullptr to omit); when provided, idle GPUs appear
// in gpu_utilisation with 0.0 and gpu_tiers is populated.
AggregatedMetrics aggregate(const std::string& allocator_name,
                            const std::vector<JobResult>& results,
                            double idle_energy = 0.0,
                            int dropped = 0,
                            const std::vector<GPU>* all_gpus = nullptr);

}  // namespace llm_alloc
