#include "llm_alloc/metrics.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>

namespace llm_alloc {

std::string AggregatedMetrics::summary_line() const {
    double util_sum = 0.0;
    for (const auto& kv : gpu_utilisation) util_sum += kv.second;
    double util_avg = util_sum / std::max<size_t>(1, gpu_utilisation.size());

    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%-20s | reward %8.3f | latency %7.2fs (p95 %7.2fs) | "
        "energy %9.1fJ (%.0f active + %.0f idle) | quality %.3f | GPU-util %.1f%%",
        allocator_name.c_str(), mean_reward, mean_latency, p95_latency,
        system_energy, total_energy, idle_energy, mean_quality, util_avg * 100.0);
    return std::string(buf);
}

AggregatedMetrics aggregate(const std::string& allocator_name,
                            const std::vector<JobResult>& results,
                            double idle_energy, int dropped,
                            const std::vector<GPU>* all_gpus) {
    AggregatedMetrics m;
    m.allocator_name = allocator_name;
    m.dropped = dropped;

    if (results.empty()) {
        m.idle_energy = 0.0;
        return m;
    }

    std::vector<double> latencies;
    latencies.reserve(results.size());
    double sim_end = 0.0;
    for (const JobResult& r : results) {
        latencies.push_back(r.latency);
        sim_end = std::max(sim_end, r.end_time);
    }
    std::vector<double> latencies_sorted = latencies;
    std::sort(latencies_sorted.begin(), latencies_sorted.end());
    size_t p95_idx = static_cast<size_t>(0.95 * latencies_sorted.size());
    if (p95_idx >= latencies_sorted.size()) p95_idx = latencies_sorted.size() - 1;

    // Per-GPU utilisation
    std::map<int, double> gpu_busy;
    for (const JobResult& r : results) {
        gpu_busy[r.gpu.id] += r.processing_time;
        m.gpu_tiers[r.gpu.id] = size_value(r.gpu.vram);
    }
    if (all_gpus != nullptr) {
        for (const GPU& g : *all_gpus) {
            gpu_busy.emplace(g.id, 0.0);  // seed idle GPUs with 0.0
            m.gpu_tiers[g.id] = size_value(g.vram);
        }
    }
    if (sim_end > 0) {
        for (const auto& kv : gpu_busy) {
            m.gpu_utilisation[kv.first] = std::min(1.0, kv.second / sim_end);
        }
    }

    double total_r = 0.0, total_e = 0.0, total_lat = 0.0, total_q = 0.0, total_ql = 0.0;
    for (const JobResult& r : results) {
        total_r += r.reward;
        total_e += r.energy;
        total_lat += r.latency;
        total_q += r.quality;
        total_ql += r.quality_loss;
    }
    double n = static_cast<double>(results.size());

    m.n_jobs = static_cast<int>(results.size());
    m.total_reward = total_r;
    m.mean_reward = total_r / n;
    m.mean_latency = total_lat / n;
    m.p95_latency = latencies_sorted[p95_idx];
    m.total_energy = total_e;
    m.idle_energy = idle_energy;
    m.system_energy = total_e + idle_energy;
    m.mean_energy = total_e / n;
    m.mean_quality = total_q / n;
    m.mean_quality_loss = total_ql / n;
    return m;
}

}  // namespace llm_alloc
