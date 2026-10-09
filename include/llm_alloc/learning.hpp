// Learning allocators: epsilon-greedy bandit, tabular Q-learning, DQN (LibTorch).
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <tuple>
#include <vector>

#include <torch/torch.h>

#include "llm_alloc/simulator.hpp"
#include "llm_alloc/batched_simulator.hpp"

namespace llm_alloc {

// ---------------------------------------------------------------------------
// Shared helpers (exposed for sweep/tests)
// ---------------------------------------------------------------------------
int discretise(double value, double lo, double hi, int n_bins);
double system_load(const std::vector<GPU>& gpus, double current_time);
double max_queue_wait(const std::vector<GPU>& gpus, double current_time);
int queue_depth(const std::vector<GPU>& gpus, double current_time);
std::vector<double> per_tier_state(const std::vector<GPU>& gpus, double current_time);
std::vector<double> env_power_features(const EnvObservation* env_state);
std::vector<double> env_tier_batch_features(const EnvObservation* env_state);

// Action = (LLM size, gpu-pick strategy)
using Action = std::pair<Size, std::string>;

GPU* pick_gpu(std::vector<GPU>& gpus, const LLM& llm, const std::string& strategy,
              const EnvObservation* env_state);
std::vector<Action> build_actions(const std::vector<LLM>& llm_pool);
std::optional<AllocationDecision> action_to_decision(
    const Action& action, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    const EnvObservation* env_state);

extern const std::vector<std::string> kGpuStrategies;

// ---------------------------------------------------------------------------
// DecisionSnapshot — opaque learner-provided (state, action) snapshot.
// A single struct holds both tabular (Q-learning) and vector (DQN) forms;
// each learner uses only its relevant fields.
// ---------------------------------------------------------------------------
struct DecisionSnapshot {
    // Q-learning: discrete state (d_bin, l_bin, b_bin) + action
    std::tuple<int, int, int> q_state{0, 0, 0};
    Action q_action{Size::SMALL, "least_loaded"};
    // DQN: state vector + action index
    torch::Tensor dqn_state;
    int dqn_action_idx = 0;
    bool is_dqn = false;
};

// ---------------------------------------------------------------------------
// 1. Epsilon-Greedy Contextual Bandit
// ---------------------------------------------------------------------------
class EpsilonGreedyAllocator : public Allocator {
public:
    EpsilonGreedyAllocator(double epsilon = 0.15, double decay = 0.999,
                           int n_difficulty_bins = 5,
                           std::optional<uint64_t> seed = std::nullopt);

    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "epsilon_greedy"; }
    bool is_learner() const override { return true; }
    void reset_for_episode() override;
    void update(double reward) override;
    double epsilon() const override { return epsilon_; }

    // test accessors
    size_t q_size() const { return q_.size(); }
    size_t counts_size() const { return counts_.size(); }

private:
    double epsilon_;
    double initial_epsilon_;
    double decay_;
    int n_bins_;
    std::mt19937_64 rng_;
    std::map<std::pair<int, int>, double> q_;      // (bin, size_value) -> value
    std::map<std::pair<int, int>, int> counts_;
    std::optional<std::pair<int, int>> last_key_;
};

// ---------------------------------------------------------------------------
// 2. Tabular Q-Learning
// ---------------------------------------------------------------------------
class QLearningAllocator : public Allocator {
public:
    QLearningAllocator(double alpha = 0.1, double gamma = 0.95,
                       double epsilon = 0.2, double epsilon_decay = 0.998,
                       double epsilon_min = 0.01, int n_difficulty_bins = 5,
                       int n_load_bins = 4, int n_batch_bins = 3,
                       double beta_d = 0.0,
                       std::optional<uint64_t> seed = std::nullopt);

    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "q_learning"; }
    bool is_learner() const override { return true; }
    void reset_for_episode() override;
    void update(double reward) override;
    void end_episode() override;
    double epsilon() const override { return epsilon_; }

    bool supports_observe_transition() const override { return true; }
    std::shared_ptr<DecisionSnapshot> pop_last_decision() override;
    void observe_transition(const std::shared_ptr<DecisionSnapshot>& decision,
                            double reward, const EnvObservation* next_state,
                            bool done, std::optional<double> holding_time) override;

    int state_space_size() const { return n_diff_bins_ * n_load_bins_ * n_batch_bins_; }
    int action_space_size() const { return actions_ ? static_cast<int>(actions_->size()) : 0; }
    size_t q_size() const { return q_.size(); }
    double effective_gamma(std::optional<double> holding_time) const;
    void set_epsilon(double e) { epsilon_ = e; }

    using State = std::tuple<int, int, int>;

private:
    State get_state(const Prompt& prompt, const std::vector<GPU>& gpus,
                    double current_time, const EnvObservation* env_state) const;
    double max_q(const State& state) const;

    double alpha_, gamma_, beta_d_;
    double epsilon_, initial_epsilon_, epsilon_decay_, epsilon_min_;
    int n_diff_bins_, n_load_bins_, n_batch_bins_;
    std::mt19937_64 rng_;

    std::map<State, std::map<Action, double>> q_;
    std::optional<State> prev_state_;
    std::optional<Action> prev_action_;
    std::optional<double> prev_reward_;
    std::optional<std::vector<Action>> actions_;
    std::shared_ptr<DecisionSnapshot> pending_decision_;
};

// ---------------------------------------------------------------------------
// 3. DQN Allocator (LibTorch)
// ---------------------------------------------------------------------------
struct QNetworkImpl : torch::nn::Module {
    QNetworkImpl(int state_dim, int n_actions, int hidden = 64);
    torch::Tensor forward(torch::Tensor x);
    torch::nn::Sequential net{nullptr};
};
TORCH_MODULE(QNetwork);

class DQNAllocator : public Allocator {
public:
    DQNAllocator(double epsilon = 0.3, double epsilon_decay = 0.995,
                 double epsilon_min = 0.01, double gamma = 0.95,
                 double lr = 1e-3, int batch_size = 64, int replay_size = 5000,
                 int target_update_freq = 50, int hidden = 64,
                 double beta_d = 0.0, std::optional<uint64_t> seed = std::nullopt);

    std::optional<AllocationDecision> allocate(
        const Prompt&, std::vector<GPU>&, const std::vector<LLM>&, double,
        const EnvObservation*) override;
    std::string name() const override { return "dqn"; }
    bool is_learner() const override { return true; }
    void reset_for_episode() override;
    void update(double reward) override;
    void end_episode() override;
    double epsilon() const override { return epsilon_; }

    bool supports_observe_transition() const override { return true; }
    std::shared_ptr<DecisionSnapshot> pop_last_decision() override;
    void observe_transition(const std::shared_ptr<DecisionSnapshot>& decision,
                            double reward, const EnvObservation* next_state,
                            bool done, std::optional<double> holding_time) override;

    int state_dim() const { return state_dim_; }
    size_t replay_size() const { return replay_.size(); }
    double effective_gamma(std::optional<double> holding_time) const;

    torch::Tensor state_vector(const Prompt& prompt, const std::vector<GPU>& gpus,
                               double current_time, const EnvObservation* env_state) const;

private:
    struct ReplayItem {
        torch::Tensor s;
        int a;
        double r;
        torch::Tensor s_next;
        bool done;
        double gamma;
    };

    void init_networks(int n_actions);
    torch::Tensor next_vec_from_obs(const torch::Tensor& prev_vec,
                                    const EnvObservation* next_state) const;
    void store_transition(torch::Tensor s, int a, double r, torch::Tensor s_next,
                          bool done, double gamma);
    void train_step();

    double epsilon_, initial_epsilon_, epsilon_decay_, epsilon_min_;
    double gamma_, beta_d_, lr_;
    int batch_size_, target_update_freq_, hidden_;
    std::mt19937_64 rng_;

    std::optional<std::vector<Action>> actions_;
    int state_dim_ = 18;
    std::vector<ReplayItem> replay_;
    int replay_max_;
    long step_count_ = 0;

    QNetwork q_net_{nullptr};
    QNetwork target_net_{nullptr};
    std::shared_ptr<torch::optim::Adam> optimizer_;

    std::optional<torch::Tensor> prev_state_vec_;
    std::optional<int> prev_action_idx_;
    std::optional<double> prev_reward_;
    std::shared_ptr<DecisionSnapshot> pending_decision_;
};

inline bool torch_available() { return true; }

// ---------------------------------------------------------------------------
// 4. Multi-episode training loop
// ---------------------------------------------------------------------------
struct TrainingCurve {
    std::vector<int> episode;
    std::vector<double> mean_reward;
    std::vector<double> total_energy;
    std::vector<double> system_energy;
    std::vector<double> mean_quality;
    std::vector<double> epsilon;
};

TrainingCurve train_learner(Allocator& allocator, int n_episodes = 50,
                            int n_prompts = 200, uint64_t seed = 42,
                            int n_small_gpus = 2, int n_medium_gpus = 2,
                            int n_large_gpus = 1,
                            const std::string& objective = "energy",
                            bool verbose = true,
                            const std::string& engine = "static");

}  // namespace llm_alloc
