#include "llm_alloc/batched_simulator.hpp"

#include <algorithm>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace llm_alloc {

// ---------------------------------------------------------------------------
// Throughput / energy physics
// ---------------------------------------------------------------------------
double decode_step_duration(const LLM& llm, const GPU& gpu, int batch_size) {
    double base_step = 1.0 / llm.base_speed;
    double vram_bonus = 1.0 + 0.15 * (size_value(gpu.vram) - size_value(llm.size));
    double batch_penalty = 1.0 + 0.12 * std::sqrt(static_cast<double>(std::max(1, batch_size) - 1));
    return base_step * batch_penalty / vram_bonus;
}

double batch_tokens_per_sec(const LLM& llm, const GPU& gpu, int batch_size) {
    if (batch_size <= 0) return 0.0;
    return batch_size / decode_step_duration(llm, gpu, batch_size);
}

double step_power(const GPU& gpu, int batch_size, int max_batch) {
    if (batch_size <= 0) return gpu.idle_power();
    double utilisation = std::min(1.0, static_cast<double>(batch_size) / std::max(1, max_batch));
    double dynamic = (gpu.active_power() - gpu.idle_power()) * utilisation;
    return gpu.idle_power() + dynamic;
}

// ---------------------------------------------------------------------------
// Runtime structures (implementation detail)
// ---------------------------------------------------------------------------
namespace {

struct Sequence {
    Prompt prompt;
    LLM llm;
    int target_tokens = 0;
    double generated = 0.0;
    double admitted_time = 0.0;
    double overhead_energy = 0.0;
    double active_energy = 0.0;

    bool done() const { return generated >= target_tokens; }
};

struct GPUState {
    GPU* gpu = nullptr;
    std::optional<LLM> loaded_llm;
    std::vector<Sequence> active;
    std::vector<Sequence> waiting;
    long busy_ticks = 0;
    double energy = 0.0;

    // power management
    std::string power_state = "idle";
    double idle_since = 0.0;
    double wake_ready_at = 0.0;
    double wake_energy_pending = 0.0;

    int capacity() const {
        if (!loaded_llm.has_value()) return 0;
        return gpu->kv_cache_slots(*loaded_llm);
    }
};

EnvObservation observe(std::map<int, GPUState>& states, double t) {
    EnvObservation obs;
    obs.time = t;
    for (auto& kv : states) {
        GPUState& st = kv.second;
        GPUObservation go;
        go.gpu_id = kv.first;
        go.vram = st.gpu->vram;
        go.power_state = st.power_state;
        if (st.loaded_llm.has_value()) go.loaded_llm_size = st.loaded_llm->size;
        go.active_count = static_cast<int>(st.active.size());
        go.capacity = st.capacity();
        go.waiting_count = static_cast<int>(st.waiting.size());
        obs.gpus.emplace(kv.first, go);
    }
    return obs;
}

}  // namespace

// ---------------------------------------------------------------------------
// BatchedSimulator
// ---------------------------------------------------------------------------
BatchedSimulator::BatchedSimulator(std::vector<GPU> gpus, std::vector<LLM> llm_pool,
                                   BatchedConfig config)
    : gpus_(std::move(gpus)), llm_pool_(std::move(llm_pool)),
      config_(std::move(config)) {}

std::vector<JobResult> BatchedSimulator::run(Allocator& allocator,
                                             std::vector<Prompt> prompts,
                                             bool online_update) {
    const BatchedConfig& cfg = config_;

    std::map<int, GPUState> states;
    for (GPU& g : gpus_) {
        GPUState st;
        st.gpu = &g;
        states.emplace(g.id, std::move(st));
    }

    std::stable_sort(prompts.begin(), prompts.end(),
              [](const Prompt& a, const Prompt& b) { return a.arrival_time < b.arrival_time; });
    size_t pending_idx = 0;

    std::vector<JobResult> results;
    dropped = 0;
    total_idle_energy = 0.0;
    active_energy = idle_energy = sleep_energy = wake_energy = accounting_error = 0.0;

    double t = 0.0;
    arrival_transitions.clear();
    if (prompts.empty()) return results;
    std::shared_ptr<DecisionSnapshot> previous_decision;
    bool previous_arrival = false;
    double previous_time = 0.0;
    double previous_energy = 0.0;
    double interval_service_cost = 0.0;

    auto fleet_energy = [&]() { return active_energy + idle_energy + sleep_energy + wake_energy; };
    auto emit_transition = [&](const std::shared_ptr<DecisionSnapshot>& successor, bool terminal) {
        if (!previous_arrival) return;
        double energy = fleet_energy() - previous_energy;
        double reward = -(cfg.reward_weights.energy * energy + interval_service_cost);
        arrival_transitions.push_back({energy, interval_service_cost, reward, t - previous_time, terminal});
        if (online_update && allocator.is_learner())
            allocator.observe_arrival_transition(previous_decision, reward, successor, terminal, t - previous_time);
        previous_energy = fleet_energy();
        interval_service_cost = 0.0;
    };

    auto all_empty = [&]() {
        for (auto& kv : states) {
            if (!kv.second.active.empty() || !kv.second.waiting.empty()) return false;
        }
        return true;
    };

    // helper: begin waking a sleeping GPU
    auto begin_wake = [&](GPUState& st) {
        st.power_state = "waking";
        st.wake_ready_at = t + st.gpu->wake_up_latency();
        st.loaded_llm.reset();
        st.energy += st.gpu->wake_up_energy();
        wake_energy += st.gpu->wake_up_energy();
    };

    // helper: admit waiting sequences into active batch up to capacity
    auto admit = [&](GPUState& st) {
        if (st.waiting.empty()) return;
        if (!st.loaded_llm.has_value() && st.active.empty()) {
            st.loaded_llm = st.waiting.front().llm;
        } else if (cfg.allow_model_switch && st.active.empty()) {
            st.loaded_llm = st.waiting.front().llm;
        }
        int cap = st.capacity();
        size_t i = 0;
        while (i < st.waiting.size() && static_cast<int>(st.active.size()) < cap) {
            if (st.waiting[i].llm.name == st.loaded_llm->name) {
                st.active.push_back(std::move(st.waiting[i]));
                st.waiting.erase(st.waiting.begin() + i);
            } else {
                ++i;
            }
        }
    };

    // helper: tick power management
    auto tick_power = [&](GPUState& st) {
        if (!st.active.empty()) {
            st.power_state = "active";
            return;  // energy already charged by decode step
        }
        if (!cfg.power_management) {
            st.energy += st.gpu->idle_power() * cfg.dt;
            idle_energy += st.gpu->idle_power() * cfg.dt;
            return;
        }
        if (st.power_state == "waking") {
            st.energy += st.gpu->idle_power() * cfg.dt;
            idle_energy += st.gpu->idle_power() * cfg.dt;
            return;
        }
        if (st.power_state == "sleep") {
            st.energy += st.gpu->sleep_power() * cfg.dt;
            sleep_energy += st.gpu->sleep_power() * cfg.dt;
            return;
        }
        if (st.power_state != "idle") {
            st.power_state = "idle";
            st.idle_since = t;
        }
        st.energy += st.gpu->idle_power() * cfg.dt;
        idle_energy += st.gpu->idle_power() * cfg.dt;
        if (t - st.idle_since >= cfg.sleep_timeout) {
            st.power_state = "sleep";
            st.loaded_llm.reset();
        }
    };

    // helper: make a JobResult from a finished sequence
    auto make_result = [&](const Sequence& seq, GPU& gpu, double end_time) -> JobResult {
        const RewardWeights& w = cfg.reward_weights;
        const Prompt& prompt = seq.prompt;
        const LLM& llm = seq.llm;

        double latency = end_time - prompt.arrival_time;
        double processing_time = end_time - seq.admitted_time;
        double energy = seq.active_energy + seq.overhead_energy;
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
        r.start_time = seq.admitted_time;
        r.end_time = end_time;
        r.latency = latency;
        r.processing_time = processing_time;
        r.energy = energy;
        r.quality = quality;
        r.quality_loss = quality_loss;
        r.reward = reward;
        return r;
    };

    // helper: run one decode step for a GPU's active batch
    auto decode_step = [&](GPUState& st) {
        if (st.active.empty() || !st.loaded_llm.has_value()) return;
        int batch_size = static_cast<int>(st.active.size());
        int max_batch = st.capacity();
        const LLM& llm = *st.loaded_llm;
        GPU& gpu = *st.gpu;

        double step_dur = decode_step_duration(llm, gpu, batch_size);
        double tokens_this_tick = cfg.dt / step_dur;
        if (tokens_this_tick <= 0) return;

        double power = step_power(gpu, batch_size, max_batch);
        st.energy += power * cfg.dt;
        active_energy += power * cfg.dt;
        st.busy_ticks += 1;

        for (Sequence& seq : st.active) {
            seq.generated += tokens_this_tick;
            seq.active_energy += power * cfg.dt / batch_size;
        }
    };

    auto finish_sequences = [&](GPUState& st) {
        GPU& gpu = *st.gpu;
        double end_time = t + cfg.dt;
        // Collect finished sequences, build results, then erase.
        for (auto it = st.active.begin(); it != st.active.end();) {
            if (it->done()) {
                gpu.jobs_completed += 1;
                JobResult result = make_result(*it, gpu, end_time);
                results.push_back(result);
                const auto& weights = cfg.reward_weights;
                interval_service_cost += weights.latency * result.latency +
                    weights.quality_loss * result.quality_loss + weights.min_quality_penalty *
                    std::max(0.0, weights.min_quality - result.quality);
                it = st.active.erase(it);
            } else {
                ++it;
            }
        }
    };

    while (true) {
        // 1) Admit arrivals that have arrived by time t
        while (pending_idx < prompts.size() && prompts[pending_idx].arrival_time <= t) {
            const Prompt& prompt = prompts[pending_idx];
            pending_idx++;
            EnvObservation env_state = observe(states, t);
            env_state.remaining_arrivals = prompts.size() - pending_idx + 1;
            auto successor = allocator.arrival_state(prompt, gpus_, t, &env_state);
            emit_transition(successor, false);
            auto decision = allocator.allocate(prompt, gpus_, llm_pool_, t, &env_state);
            previous_decision = allocator.pop_last_decision();
            previous_arrival = true;
            previous_time = t;
            if (!decision.has_value() || decision->gpu == nullptr ||
                !decision->llm.fits_on(*decision->gpu)) {
                dropped++;
                continue;
            }
            GPUState& st = states.at(decision->gpu->id);
            int target = compute_output_tokens(prompt, decision->llm);
            Sequence seq;
            seq.prompt = prompt;
            seq.llm = decision->llm;
            seq.target_tokens = target;
            seq.admitted_time = t;
            st.waiting.push_back(std::move(seq));
            if (cfg.power_management && st.power_state == "sleep") {
                begin_wake(st);
            }
        }

        // 2) Transition waking GPUs to idle once cold start completes
        if (cfg.power_management) {
            for (auto& kv : states) {
                GPUState& st = kv.second;
                if (st.power_state == "waking" && t >= st.wake_ready_at) {
                    st.power_state = "idle";
                    st.idle_since = t;
                }
            }
        }

        // 3) Admission control
        for (auto& kv : states) {
            GPUState& st = kv.second;
            if (cfg.power_management && st.power_state == "waking") continue;
            admit(st);
        }

        // 4) Decode one step on every active GPU
        for (auto& kv : states) decode_step(kv.second);

        // 5) Power-state bookkeeping + non-active energy
        double overhead_before = idle_energy + sleep_energy;
        for (auto& kv : states) tick_power(kv.second);

        // 5b) Attribute this tick's overhead energy across in-flight sequences
        double overhead = idle_energy + sleep_energy - overhead_before;
        std::vector<Sequence*> in_flight;
        for (auto& kv : states) {
            for (Sequence& s : kv.second.active) in_flight.push_back(&s);
        }
        if (!in_flight.empty() && overhead > 0) {
            double share = overhead / in_flight.size();
            for (Sequence* s : in_flight) s->overhead_energy += share;
        }
        for (auto& entry : states) finish_sequences(entry.second);

        // 6) Online learner feedback
        bool episode_over = (pending_idx >= prompts.size()) && all_empty();

        // 7) Advance time / termination
        t += cfg.dt;
        if (episode_over) break;
        if (t > cfg.max_sim_time)
            throw std::runtime_error("Batched simulation time limit reached before fleet drain; no terminal transition emitted");
    }
    emit_transition(nullptr, true);

    // Energy accounting
    double total_gpu_energy = 0.0;
    for (auto& kv : states) total_gpu_energy += kv.second.energy;
    double job_energy = 0.0;
    for (const JobResult& r : results) job_energy += r.energy;
    total_idle_energy = std::max(0.0, total_gpu_energy - job_energy);
    accounting_error = total_gpu_energy - (active_energy + idle_energy + sleep_energy + wake_energy);
    if (!results.empty()) {
        double residual_share = total_idle_energy / results.size();
        for (auto& result : results) {
            result.energy += residual_share;
            result.reward -= cfg.reward_weights.energy * residual_share;
        }
        total_idle_energy = 0.0;
    }
    return results;
}

}  // namespace llm_alloc
