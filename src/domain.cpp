#include "llm_alloc/domain.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace llm_alloc {

const char* size_name(Size s) {
    switch (s) {
        case Size::SMALL: return "SMALL";
        case Size::MEDIUM: return "MEDIUM";
        case Size::LARGE: return "LARGE";
    }
    return "?";
}

// ---------------------------------------------------------------------------
// GPU energy/latency properties (keyed by VRAM tier)
//
// CALIBRATION (see tools/measure_gpu.py, tools/gpu_measurements.json):
//   SMALL  <- NVIDIA RTX 3070 Ti (8 GB)  : idle 74 W, active 249 W, 43 TFLOP/s
//   LARGE  <- NVIDIA RTX 3090   (24 GB)  : idle 128 W, active 349 W, 64 TFLOP/s
//   MEDIUM <- interpolated between the two (no physical card for this tier)
// Values marked "measured" come from nvidia-smi power sampling under a
// sustained FP16 GEMM load on this host; values marked "estimated" (sleep
// power, wake-up latency/energy, KV-cache slots) could not be measured with
// this setup and remain stylised.
// ---------------------------------------------------------------------------
double GPU::power_multiplier() const {
    // Ratio of measured active board power to the SMALL tier (249 W):
    //   3070 Ti 249/249 = 1.00 ; 3090 349/249 = 1.40 ; MEDIUM interpolated.
    switch (vram) {
        case Size::SMALL: return 1.0;
        case Size::MEDIUM: return 1.2;
        case Size::LARGE: return 1.4;
    }
    return 1.0;
}

double GPU::idle_power() const {
    // measured (p50 idle, nvidia-smi); MEDIUM interpolated.
    switch (vram) {
        case Size::SMALL: return 74.0;   // RTX 3070 Ti
        case Size::MEDIUM: return 100.0; // interpolated
        case Size::LARGE: return 128.0;  // RTX 3090
    }
    return 74.0;
}

double GPU::sleep_power() const {
    // estimated (not measured): a suspended GPU with VRAM cleared draws little.
    switch (vram) {
        case Size::SMALL: return 2.0;
        case Size::MEDIUM: return 4.0;
        case Size::LARGE: return 8.0;
    }
    return 2.0;
}

double GPU::wake_up_latency() const {
    // estimated (not measured): time to resume + reload weights into VRAM.
    switch (vram) {
        case Size::SMALL: return 2.0;
        case Size::MEDIUM: return 5.0;
        case Size::LARGE: return 12.0;
    }
    return 2.0;
}

double GPU::wake_up_energy() const {
    // estimated (not measured): one-off energy to wake + reload.
    switch (vram) {
        case Size::SMALL: return 150.0;
        case Size::MEDIUM: return 600.0;
        case Size::LARGE: return 2000.0;
    }
    return 150.0;
}

double GPU::active_power() const {
    // measured (peak board power under sustained GEMM); MEDIUM interpolated.
    switch (vram) {
        case Size::SMALL: return 249.0;  // RTX 3070 Ti
        case Size::MEDIUM: return 300.0; // interpolated
        case Size::LARGE: return 349.0;  // RTX 3090
    }
    return 249.0;
}

int GPU::kv_cache_slots(const LLM& llm) const {
    int total_slots = 0;
    switch (vram) {
        case Size::SMALL: total_slots = 8; break;
        case Size::MEDIUM: total_slots = 32; break;
        case Size::LARGE: total_slots = 96; break;
    }
    int model_footprint = 0;
    switch (llm.size) {
        case Size::SMALL: model_footprint = 1; break;
        case Size::MEDIUM: model_footprint = 4; break;
        case Size::LARGE: model_footprint = 12; break;
    }
    return std::max(1, total_slots - model_footprint);
}

// ---------------------------------------------------------------------------
// Default model pool
// ---------------------------------------------------------------------------
std::vector<LLM> default_llm_pool() {
    return {
        LLM("SmallLM",  Size::SMALL,  0.35, 120.0, 0.005),
        LLM("MediumLM", Size::MEDIUM, 0.65, 60.0,  0.015),
        LLM("LargeLM",  Size::LARGE,  0.95, 25.0,  0.045),
    };
}

// ---------------------------------------------------------------------------
// GPU pool
// ---------------------------------------------------------------------------
std::vector<GPU> create_gpu_pool(int n_small, int n_medium, int n_large) {
    std::vector<GPU> gpus;
    int idx = 0;
    struct Tier { int count; Size size; };
    for (const Tier& t : {Tier{n_small, Size::SMALL},
                          Tier{n_medium, Size::MEDIUM},
                          Tier{n_large, Size::LARGE}}) {
        for (int i = 0; i < t.count; ++i) {
            gpus.emplace_back(idx++, t.size);
        }
    }
    return gpus;
}

// ---------------------------------------------------------------------------
// PromptGenerator
// ---------------------------------------------------------------------------
PromptGenerator::PromptGenerator(double mean_difficulty, double std_difficulty,
                                 double mean_interarrival, int mean_input_tokens,
                                 std::optional<uint64_t> rng_seed)
    : mean_difficulty_(mean_difficulty),
      std_difficulty_(std_difficulty),
      mean_interarrival_(mean_interarrival),
      mean_input_tokens_(mean_input_tokens) {
    if (rng_seed.has_value()) {
        rng_.seed(*rng_seed);
    } else {
        std::random_device rd;
        rng_.seed((static_cast<uint64_t>(rd()) << 32) ^ rd());
    }
}

std::vector<Prompt> PromptGenerator::generate(int n) {
    std::vector<Prompt> prompts;
    prompts.reserve(static_cast<size_t>(std::max(0, n)));

    std::normal_distribution<double> diff_dist(mean_difficulty_, std_difficulty_);
    std::exponential_distribution<double> interarrival_dist(1.0 / mean_interarrival_);
    std::normal_distribution<double> token_dist(static_cast<double>(mean_input_tokens_),
                                                mean_input_tokens_ * 0.3);

    for (int i = 0; i < n; ++i) {
        // Difficulty: Gaussian clipped to [0.1, 1.0]
        double diff = diff_dist(rng_);
        diff = std::max(0.1, std::min(1.0, diff));

        // Inter-arrival: exponential (Poisson process)
        double dt = interarrival_dist(rng_);
        clock_ += dt;

        int tokens = std::max(16, static_cast<int>(token_dist(rng_)));

        Prompt p;
        p.id = next_id_++;
        p.arrival_time = clock_;
        // round to 4 decimals (matches Python round(diff, 4))
        p.difficulty = std::round(diff * 10000.0) / 10000.0;
        p.input_tokens = tokens;
        prompts.push_back(p);
    }
    return prompts;
}

}  // namespace llm_alloc
