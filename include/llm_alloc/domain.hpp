// Domain models: GPU, LLM, Prompt — plus pool/generator helpers.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace llm_alloc {

// ---------------------------------------------------------------------------
// Size tier (shared by GPU VRAM and LLM requirements)
// Integer value is used for compatibility checks (LLM <= GPU).
// ---------------------------------------------------------------------------
enum class Size : int {
    SMALL = 1,
    MEDIUM = 2,
    LARGE = 3,
};

inline int size_value(Size s) { return static_cast<int>(s); }
const char* size_name(Size s);

// ---------------------------------------------------------------------------
// LLM — forward declared for GPU::kv_cache_slots
// ---------------------------------------------------------------------------
struct LLM;

// ---------------------------------------------------------------------------
// GPU — one GPU in the cluster
// ---------------------------------------------------------------------------
struct GPU {
    int id = 0;
    Size vram = Size::SMALL;

    // Runtime bookkeeping (mutable)
    double busy_until = 0.0;        // sim timestamp when current job finishes
    double total_busy_time = 0.0;   // accumulated busy time (for utilisation)
    int jobs_completed = 0;

    GPU() = default;
    GPU(int id_, Size vram_) : id(id_), vram(vram_) {}

    double power_multiplier() const;  // larger GPU -> higher base power
    double idle_power() const;        // idle power draw (W)
    double sleep_power() const;       // sleep power draw (W)
    double wake_up_latency() const;   // seconds to wake from sleep
    double wake_up_energy() const;    // one-off energy (J) to wake
    double active_power() const;      // peak active power (W)

    // Max concurrent sequences (batch size) that fit in VRAM for the model.
    int kv_cache_slots(const LLM& llm) const;
};

// ---------------------------------------------------------------------------
// LLM — a model from the pool
// ---------------------------------------------------------------------------
struct LLM {
    std::string name;
    Size size = Size::SMALL;
    double capability = 0.0;       // 0-1 (higher = handles harder prompts)
    double base_speed = 0.0;       // tokens/sec on a matching GPU
    double energy_per_token = 0.0; // energy per token on a matching GPU

    LLM() = default;
    LLM(std::string name_, Size size_, double capability_,
        double base_speed_, double energy_per_token_)
        : name(std::move(name_)), size(size_), capability(capability_),
          base_speed(base_speed_), energy_per_token(energy_per_token_) {}

    bool fits_on(const GPU& gpu) const { return size_value(size) <= size_value(gpu.vram); }
};

std::vector<LLM> default_llm_pool();

// ---------------------------------------------------------------------------
// Prompt — an incoming inference request
// ---------------------------------------------------------------------------
struct Prompt {
    int id = 0;
    double arrival_time = 0.0;
    double difficulty = 0.0;       // 0.1 - 1.0 (Gaussian, clipped)
    int input_tokens = 0;
    int output_tokens = 0;         // filled after allocation
    // assigned_gpu / assigned_llm are not needed by the simulators (they carry
    // the decision separately), so they are omitted here.
};

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------
std::vector<GPU> create_gpu_pool(int n_small = 2, int n_medium = 2, int n_large = 1);

// Generate prompts with Gaussian difficulty and Poisson (exponential) arrival.
class PromptGenerator {
public:
    PromptGenerator(double mean_difficulty = 0.5,
                    double std_difficulty = 0.2,
                    double mean_interarrival = 1.0,
                    int mean_input_tokens = 256,
                    std::optional<uint64_t> rng_seed = std::nullopt);

    std::vector<Prompt> generate(int n = 1);

private:
    double mean_difficulty_;
    double std_difficulty_;
    double mean_interarrival_;
    int mean_input_tokens_;
    std::mt19937_64 rng_;
    int next_id_ = 0;
    double clock_ = 0.0;
};

}  // namespace llm_alloc
