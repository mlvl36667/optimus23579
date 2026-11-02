#include "llm_alloc/learning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <stdexcept>

#include "llm_alloc/domain.hpp"
#include "llm_alloc/metrics.hpp"

namespace llm_alloc {

// ======================================================================
// Shared helpers
// ======================================================================
int discretise(double value, double lo, double hi, int n_bins) {
    double ratio = (value - lo) / std::max(hi - lo, 1e-9);
    int idx = static_cast<int>(ratio * n_bins);
    return std::max(0, std::min(n_bins - 1, idx));
}

double system_load(const std::vector<GPU>& gpus, double current_time) {
    if (gpus.empty()) return 0.0;
    int busy = 0;
    for (const GPU& g : gpus) if (g.busy_until > current_time) busy++;
    return static_cast<double>(busy) / gpus.size();
}

double max_queue_wait(const std::vector<GPU>& gpus, double current_time) {
    double m = 0.0;
    bool any = false;
    for (const GPU& g : gpus) {
        double w = g.busy_until - current_time;
        if (!any || w > m) { m = w; any = true; }
    }
    return any ? m : 0.0;
}

int queue_depth(const std::vector<GPU>& gpus, double current_time) {
    int d = 0;
    for (const GPU& g : gpus) if (g.busy_until > current_time) d++;
    return d;
}

std::vector<double> per_tier_state(const std::vector<GPU>& gpus, double current_time) {
    std::vector<double> feats;
    for (Size tier : {Size::SMALL, Size::MEDIUM, Size::LARGE}) {
        std::vector<const GPU*> tier_gpus;
        for (const GPU& g : gpus) if (g.vram == tier) tier_gpus.push_back(&g);
        if (tier_gpus.empty()) {
            feats.push_back(0.0);
            feats.push_back(0.0);
            continue;
        }
        int busy = 0;
        double soonest = 1e18;
        for (const GPU* g : tier_gpus) {
            if (g->busy_until > current_time) busy++;
            soonest = std::min(soonest, std::max(0.0, g->busy_until - current_time));
        }
        double busy_frac = static_cast<double>(busy) / tier_gpus.size();
        feats.push_back(busy_frac);
        feats.push_back(std::min(soonest / 30.0, 1.0));
    }
    return feats;
}

std::vector<double> env_power_features(const EnvObservation* env_state) {
    if (env_state == nullptr || env_state->gpus.empty()) {
        return {0.0, 0.0, 0.0, 0.0};
    }
    size_t n = env_state->gpus.size();
    double sleep_c = 0, active_c = 0, fill = 0, wait = 0;
    for (const auto& kv : env_state->gpus) {
        const GPUObservation& o = kv.second;
        if (o.power_state == "sleep") sleep_c += 1;
        if (o.power_state == "active") active_c += 1;
        fill += o.batch_fill();
        wait += std::min(o.waiting_count / 8.0, 1.0);
    }
    return {sleep_c / n, active_c / n, fill / n, wait / n};
}

std::vector<double> env_tier_batch_features(const EnvObservation* env_state) {
    if (env_state == nullptr || env_state->gpus.empty()) {
        return {0.0, 0.0, 0.0};
    }
    std::vector<double> feats;
    for (Size tier : {Size::SMALL, Size::MEDIUM, Size::LARGE}) {
        std::vector<const GPUObservation*> tier_obs;
        for (const auto& kv : env_state->gpus) {
            if (kv.second.vram == tier) tier_obs.push_back(&kv.second);
        }
        if (tier_obs.empty()) {
            feats.push_back(0.0);
        } else {
            double sum = 0.0;
            for (const GPUObservation* o : tier_obs) sum += o->batch_fill();
            feats.push_back(sum / tier_obs.size());
        }
    }
    return feats;
}

// ======================================================================
// Action space
// ======================================================================
const std::vector<std::string> kGpuStrategies = {
    "least_loaded", "match_vram", "wake_sleeping",
    "gpu_small", "gpu_medium", "gpu_large",
};

namespace {
std::optional<Size> tier_strategy_to_size(const std::string& s) {
    if (s == "gpu_small") return Size::SMALL;
    if (s == "gpu_medium") return Size::MEDIUM;
    if (s == "gpu_large") return Size::LARGE;
    return std::nullopt;
}

std::mt19937_64 make_rng(std::optional<uint64_t> seed) {
    if (seed.has_value()) return std::mt19937_64(*seed);
    std::random_device rd;
    return std::mt19937_64((static_cast<uint64_t>(rd()) << 32) ^ rd());
}
}  // namespace

GPU* pick_gpu(std::vector<GPU>& gpus, const LLM& llm, const std::string& strategy,
              const EnvObservation* env_state) {
    std::vector<GPU*> ok;
    for (GPU& g : gpus) if (llm.fits_on(g)) ok.push_back(&g);
    if (ok.empty()) return nullptr;

    auto least_loaded = [&]() -> GPU* {
        return *std::min_element(ok.begin(), ok.end(),
            [](const GPU* a, const GPU* b) { return a->busy_until < b->busy_until; });
    };

    if (strategy == "least_loaded") {
        return least_loaded();
    } else if (strategy == "match_vram") {
        std::vector<GPU*> exact;
        for (GPU* g : ok) if (g->vram == llm.size) exact.push_back(g);
        std::vector<GPU*>& pool = exact.empty() ? ok : exact;
        return *std::min_element(pool.begin(), pool.end(),
            [](const GPU* a, const GPU* b) { return a->busy_until < b->busy_until; });
    } else if (strategy == "wake_sleeping") {
        if (env_state != nullptr && !env_state->gpus.empty()) {
            std::vector<GPU*> sleeping;
            for (GPU* g : ok) {
                const GPUObservation* obs = env_state->get(g->id);
                if (obs != nullptr && obs->power_state == "sleep") sleeping.push_back(g);
            }
            if (!sleeping.empty()) {
                return *std::max_element(sleeping.begin(), sleeping.end(),
                    [](const GPU* a, const GPU* b) {
                        return size_value(a->vram) < size_value(b->vram);
                    });
            }
        }
        return least_loaded();
    } else if (auto target = tier_strategy_to_size(strategy)) {
        std::vector<GPU*> tier_gpus;
        for (GPU* g : ok) if (g->vram == *target) tier_gpus.push_back(g);
        if (!tier_gpus.empty()) {
            return *std::min_element(tier_gpus.begin(), tier_gpus.end(),
                [](const GPU* a, const GPU* b) { return a->busy_until < b->busy_until; });
        }
        return least_loaded();
    } else {  // "largest"
        return *std::max_element(ok.begin(), ok.end(),
            [](const GPU* a, const GPU* b) {
                if (size_value(a->vram) != size_value(b->vram))
                    return size_value(a->vram) < size_value(b->vram);
                return a->busy_until > b->busy_until;  // -busy_until
            });
    }
}

std::vector<Action> build_actions(const std::vector<LLM>& llm_pool) {
    std::set<int> size_set;
    for (const LLM& m : llm_pool) size_set.insert(size_value(m.size));
    std::vector<Action> actions;
    for (int sv : size_set) {
        Size s = static_cast<Size>(sv);
        for (const std::string& strat : kGpuStrategies) {
            actions.emplace_back(s, strat);
        }
    }
    return actions;
}

std::optional<AllocationDecision> action_to_decision(
    const Action& action, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    const EnvObservation* env_state) {
    Size llm_size = action.first;
    const std::string& gpu_strat = action.second;
    const LLM* chosen = nullptr;
    for (const LLM& m : llm_pool) if (m.size == llm_size) { chosen = &m; break; }
    if (!chosen) return std::nullopt;
    GPU* gpu = pick_gpu(gpus, *chosen, gpu_strat, env_state);
    if (gpu == nullptr) return std::nullopt;
    return AllocationDecision(gpu, *chosen);
}

// ======================================================================
// 1. Epsilon-Greedy Contextual Bandit
// ======================================================================
EpsilonGreedyAllocator::EpsilonGreedyAllocator(double epsilon, double decay,
                                               int n_difficulty_bins,
                                               std::optional<uint64_t> seed)
    : epsilon_(epsilon), initial_epsilon_(epsilon), decay_(decay),
      n_bins_(n_difficulty_bins), rng_(make_rng(seed)) {}

void EpsilonGreedyAllocator::reset_for_episode() { epsilon_ = initial_epsilon_; }

std::optional<AllocationDecision> EpsilonGreedyAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double, const EnvObservation*) {
    int d_bin = discretise(prompt.difficulty, 0.1, 1.0, n_bins_);
    std::set<int> size_set;
    for (const LLM& m : llm_pool) size_set.insert(size_value(m.size));
    std::vector<int> available_sizes(size_set.begin(), size_set.end());

    int chosen_size;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (!evaluation_mode && uni(rng_) < epsilon_) {
        std::uniform_int_distribution<size_t> pick(0, available_sizes.size() - 1);
        chosen_size = available_sizes[pick(rng_)];
    } else {
        chosen_size = available_sizes[0];
        auto value = [&](int size) {
            auto found = q_.find({d_bin, size});
            return found == q_.end() ? 0.0 : found->second;
        };
        double best = value(chosen_size);
        for (int sv : available_sizes) {
            double v = value(sv);
            if (v > best) { best = v; chosen_size = sv; }
        }
    }
    last_key_ = std::make_pair(d_bin, chosen_size);

    const LLM* chosen = nullptr;
    for (const LLM& m : llm_pool) if (size_value(m.size) == chosen_size) { chosen = &m; break; }
    if (!chosen) return std::nullopt;

    std::vector<GPU*> ok;
    for (GPU& g : gpus) if (chosen->fits_on(g)) ok.push_back(&g);
    if (ok.empty()) return std::nullopt;
    std::stable_sort(ok.begin(), ok.end(),
        [](const GPU* a, const GPU* b) { return a->busy_until < b->busy_until; });
    return AllocationDecision(ok[0], *chosen);
}

void EpsilonGreedyAllocator::update(double reward) {
    if (evaluation_mode) return;
    if (!last_key_.has_value()) return;
    auto key = *last_key_;
    counts_[key] += 1;
    int n = counts_[key];
    q_[key] += (reward - q_[key]) / n;
    epsilon_ *= decay_;
}

// ======================================================================
// 2. Tabular Q-Learning
// ======================================================================
QLearningAllocator::QLearningAllocator(double alpha, double gamma, double epsilon,
                                       double epsilon_decay, double epsilon_min,
                                       int n_difficulty_bins, int n_load_bins,
                                       int n_batch_bins, double beta_d,
                                       std::optional<uint64_t> seed)
    : alpha_(alpha), gamma_(gamma), beta_d_(beta_d),
      epsilon_(epsilon), initial_epsilon_(epsilon),
      epsilon_decay_(epsilon_decay), epsilon_min_(epsilon_min),
      n_diff_bins_(n_difficulty_bins), n_load_bins_(n_load_bins),
      n_batch_bins_(n_batch_bins), rng_(make_rng(seed)) {}

void QLearningAllocator::reset_for_episode() {
    prev_state_.reset();
    prev_action_.reset();
    prev_reward_.reset();
    pending_decision_.reset();
}

namespace {
std::pair<int, int> tabular_batch_bins(const EnvObservation& observation,
                                     int load_bins, int batch_bins) {
    int occupied = 0;
    bool waiting = false;
    double peak_fill = 0.0;
    for (const auto& entry : observation.gpus) {
        const auto& gpu = entry.second;
        if (gpu.active_count > 0 || gpu.waiting_count > 0) ++occupied;
        waiting = waiting || gpu.waiting_count > 0;
        double fill = static_cast<double>(std::max(0, gpu.active_count)) / std::max(1, gpu.capacity);
        peak_fill = std::max(peak_fill, std::min(1.0, fill));
    }
    double load = static_cast<double>(occupied) / std::max<size_t>(1, observation.gpus.size());
    int load_bin = waiting ? std::max(0, load_bins - 1) :
        static_cast<int>(std::ceil(load * std::max(0, load_bins - 1)));
    int batch_bin = static_cast<int>(std::ceil(peak_fill * std::max(0, batch_bins - 1)));
    return {load_bin, batch_bin};
}
}

QLearningAllocator::State QLearningAllocator::get_state(
    const Prompt& prompt, const std::vector<GPU>& gpus, double current_time,
    const EnvObservation* env_state) const {
    int d_bin = discretise(prompt.difficulty, 0.1, 1.0, n_diff_bins_);
    int horizon_offset = 0;
    if (horizon_steps_ > 0) {
        if (!env_state || !env_state->remaining_arrivals || *env_state->remaining_arrivals == 0 ||
            *env_state->remaining_arrivals > static_cast<size_t>(horizon_steps_))
            throw std::invalid_argument("Finite-horizon state requires a valid remaining-arrival count");
        horizon_offset = static_cast<int>(*env_state->remaining_arrivals - 1) * n_load_bins_;
    }
    if (env_state != nullptr) {
        auto bins = tabular_batch_bins(*env_state, n_load_bins_, n_batch_bins_);
        return State{d_bin, bins.first + horizon_offset, bins.second};
    }
    double load = system_load(gpus, current_time);
    int l_bin = discretise(load, 0.0, 1.0, n_load_bins_);
    auto pf = env_power_features(env_state);
    int b_bin = discretise(pf[2], 0.0, 1.0, n_batch_bins_);
    return State{d_bin, l_bin, b_bin};
}

double QLearningAllocator::max_q(const State& state) const {
    if (!actions_) return 0.0;
    auto it = q_.find(state);
    if (it == q_.end() || it->second.empty()) return 0.0;
    double best = -1e300;
    for (const Action& a : *actions_) {
        auto ait = it->second.find(a);
        double v = (ait == it->second.end()) ? 0.0 : ait->second;
        best = std::max(best, v);
    }
    return best;
}

std::optional<AllocationDecision> QLearningAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double current_time, const EnvObservation* env_state) {
    if (!actions_) actions_ = build_actions(llm_pool);

    State state = get_state(prompt, gpus, current_time, env_state);

    // Legacy static-engine TD bootstrap on observing s'.
    if (!evaluation_mode && prev_state_.has_value() && prev_reward_.has_value()) {
        const State& s = *prev_state_;
        const Action& a = *prev_action_;
        double r = *prev_reward_;
        apply_td(s, a, r, max_q(state), gamma_, false);
    }

    // epsilon-greedy selection
    Action action;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (!evaluation_mode && uni(rng_) < epsilon_) {
        std::uniform_int_distribution<size_t> pick(0, actions_->size() - 1);
        action = (*actions_)[pick(rng_)];
    } else {
        auto value = [&](const Action& candidate) {
            auto found = q_.find(state);
            if (found == q_.end()) return 0.0;
            auto entry = found->second.find(candidate);
            return entry == found->second.end() ? 0.0 : entry->second;
        };
        action = (*actions_)[0];
        double best = value(action);
        for (const Action& a : *actions_) {
            double v = value(a);
            if (v > best) { best = v; action = a; }
        }
    }

    if (!evaluation_mode) ++visits_[state][action];
    prev_state_ = state;
    prev_action_ = action;
    pending_decision_ = std::make_shared<DecisionSnapshot>();
    pending_decision_->q_state = state;
    pending_decision_->q_action = action;
    pending_decision_->is_dqn = false;

    return action_to_decision(action, gpus, llm_pool, env_state);
}

std::shared_ptr<DecisionSnapshot> QLearningAllocator::pop_last_decision() {
    auto d = pending_decision_;
    pending_decision_.reset();
    return d;
}

double QLearningAllocator::effective_gamma(std::optional<double> holding_time) const {
    if (beta_d_ <= 0.0 || !holding_time.has_value()) return gamma_;
    return std::exp(-beta_d_ * std::max(0.0, *holding_time));
}

std::shared_ptr<DecisionSnapshot> QLearningAllocator::arrival_state(
    const Prompt& prompt, const std::vector<GPU>& gpus, double time, const EnvObservation* observation) {
    auto snapshot = std::make_shared<DecisionSnapshot>();
    snapshot->q_state = get_state(prompt, gpus, time, observation);
    return snapshot;
}

void QLearningAllocator::observe_arrival_transition(
    const std::shared_ptr<DecisionSnapshot>& decision, double reward,
    const std::shared_ptr<DecisionSnapshot>& successor, bool done, double elapsed) {
    if (evaluation_mode || !decision) return;
    if (!done && !successor) throw std::invalid_argument("Nonterminal Q transition requires a successor state");
    if (horizon_steps_ > 0) {
        int remaining_after = std::get<1>(decision->q_state) / n_load_bins_;
        if ((done && remaining_after != 0) ||
            (!done && std::get<1>(successor->q_state) / n_load_bins_ != remaining_after - 1))
            throw std::logic_error("Finite-horizon transition must advance exactly one arrival, then drain");
    }
    apply_td(decision->q_state, decision->q_action, reward,
             done ? 0.0 : max_q(successor->q_state), effective_gamma(elapsed), done);
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
}

void QLearningAllocator::observe_transition(
    const std::shared_ptr<DecisionSnapshot>& decision, double reward,
    const EnvObservation* next_state, bool done, std::optional<double> holding_time) {
    if (evaluation_mode || !decision) return;
    if (horizon_steps_ > 0) throw std::logic_error("Finite-horizon Q-learning requires arrival transitions");
    const State& s = decision->q_state;
    const Action& a = decision->q_action;
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);

    double gamma_k = effective_gamma(holding_time);
    double bootstrap = 0.0;
    if (!done) {
        auto bins = next_state ? tabular_batch_bins(*next_state, n_load_bins_, n_batch_bins_) :
            std::make_pair(std::get<1>(s), std::get<2>(s));
        State s_next{std::get<0>(s), bins.first, bins.second};
        bootstrap = max_q(s_next);
    }
    apply_td(s, a, reward, bootstrap, gamma_k, done);
}

void QLearningAllocator::update(double reward) {
    if (evaluation_mode) return;
    prev_reward_ = reward;
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
}

void QLearningAllocator::write_visits(std::ostream& output) const {
    output << "difficulty_bin,load_bin,batch_bin,model_size,gpu_rule,visits,updates,q_value";
    if (horizon_steps_ > 0) output << ",remaining_arrivals";
    output << '\n';
    if (!actions_) return;
    for (int difficulty = 0; difficulty < n_diff_bins_; ++difficulty) {
        for (int load = 0; load < n_load_bins_ * std::max(1, horizon_steps_); ++load) {
            for (int batch = 0; batch < n_batch_bins_; ++batch) {
                State state{difficulty, load, batch};
                for (const auto& action : *actions_) {
                    auto lookup = [&](const auto& table) -> double {
                        auto found = table.find(state);
                        if (found == table.end()) return 0.0;
                        auto entry = found->second.find(action);
                        return entry == found->second.end() ? 0.0 : static_cast<double>(entry->second);
                    };
                    output << difficulty << ',' << load % n_load_bins_ << ',' << batch << ','
                           << size_name(action.first) << ',' << action.second << ','
                           << lookup(visits_) << ',' << lookup(updates_) << ',' << lookup(q_);
                    if (horizon_steps_ > 0) output << ',' << load / n_load_bins_ + 1;
                    output << '\n';
                }
            }
        }
    }
}

double QLearningAllocator::q_value(const State& state, const Action& action) const {
    auto found = q_.find(state);
    if (found == q_.end()) return 0.0;
    auto entry = found->second.find(action);
    return entry == found->second.end() ? 0.0 : entry->second;
}

void QLearningAllocator::set_finite_horizon(int steps) {
    if (steps <= 0 || steps > 1000000)
        throw std::invalid_argument("Finite horizon must be between 1 and 1000000 arrivals");
    if (!q_.empty() || !visits_.empty())
        throw std::logic_error("Set the horizon before training, using a fresh Q table");
    horizon_steps_ = steps;
}

void QLearningAllocator::set_td_trace(std::ostream* output) {
    td_trace_ = output;
    if (td_trace_) {
        *td_trace_ << "update,difficulty_bin,load_bin,batch_bin,model_size,gpu_rule,reward,bootstrap,discount,target,td_error,q_before,q_after,alpha,epsilon,terminal";
        if (horizon_steps_ > 0) *td_trace_ << ",remaining_arrivals";
        *td_trace_ << '\n';
    }
}

void QLearningAllocator::Moments::add(double value) {
    if (count == 0) minimum = maximum = value;
    minimum = std::min(minimum, value);
    maximum = std::max(maximum, value);
    ++count;
    double delta = value - mean;
    mean += delta / count;
    squared_deviations += delta * (value - mean);
}

double QLearningAllocator::Moments::stddev() const {
    return count > 1 ? std::sqrt(std::max(0.0, squared_deviations / (count - 1))) : 0.0;
}

void QLearningAllocator::write_reward_stats(std::ostream& output) const {
    output.precision(17);
    output << "difficulty_bin,load_bin,batch_bin,model_size,gpu_rule,terminal,window_start,window_end,count,reward_mean,reward_stddev,reward_min,reward_max,target_mean,target_stddev,td_mean,td_stddev,abs_td_mean";
    if (horizon_steps_ > 0) output << ",remaining_arrivals";
    output << '\n';
    for (const auto& entry : reward_stats_) {
        const auto& state = std::get<0>(entry.first);
        const auto& action = std::get<1>(entry.first);
        auto window = std::get<3>(entry.first);
        const auto& stats = entry.second;
        output << std::get<0>(state) << ',' << std::get<1>(state) % n_load_bins_ << ',' << std::get<2>(state)
            << ',' << size_name(action.first) << ',' << action.second << ',' << std::get<2>(entry.first)
            << ',' << window * 5000 + 1 << ',' << std::min((window + 1) * 5000, td_update_count_)
            << ',' << stats.reward.count << ',' << stats.reward.mean << ',' << stats.reward.stddev()
            << ',' << stats.reward.minimum << ',' << stats.reward.maximum << ',' << stats.target.mean
            << ',' << stats.target.stddev() << ',' << stats.td_error.mean << ',' << stats.td_error.stddev()
            << ',' << stats.absolute_td_error.mean;
        if (horizon_steps_ > 0) output << ',' << std::get<1>(state) / n_load_bins_ + 1;
        output << '\n';
    }
}

void QLearningAllocator::apply_td(const State& state, const Action& action,
    double reward, double bootstrap, double discount, bool terminal) {
    double& value = q_[state][action];
    double before = value;
    double target = reward + (terminal ? 0.0 : discount * bootstrap);
    double error = target - before;
    auto& stats = reward_stats_[std::make_tuple(state, action, terminal, td_update_count_ / 5000)];
    stats.reward.add(reward);
    stats.target.add(target);
    stats.td_error.add(error);
    stats.absolute_td_error.add(std::abs(error));
    value += alpha_ * error;
    ++updates_[state][action];
    ++td_update_count_;
    if (td_trace_) {
        td_trace_->precision(17);
        *td_trace_ << td_update_count_ << ',' << std::get<0>(state) << ','
            << std::get<1>(state) % n_load_bins_ << ',' << std::get<2>(state) << ','
            << size_name(action.first) << ',' << action.second << ',' << reward << ','
            << bootstrap << ',' << discount << ',' << target << ',' << error << ','
            << before << ',' << value << ',' << alpha_ << ',' << epsilon_ << ',' << terminal;
        if (horizon_steps_ > 0) *td_trace_ << ',' << std::get<1>(state) / n_load_bins_ + 1;
        *td_trace_ << '\n';
    }
}

void QLearningAllocator::end_episode() {
    if (evaluation_mode) return;
    if (prev_state_.has_value() && prev_reward_.has_value()) {
        const State& s = *prev_state_;
        const Action& a = *prev_action_;
        double r = *prev_reward_;
        apply_td(s, a, r, 0.0, gamma_, true);
    }
    prev_state_.reset();
    prev_action_.reset();
    prev_reward_.reset();
    pending_decision_.reset();
}

// ======================================================================
// 3. DQN Allocator (LibTorch)
// ======================================================================
QNetworkImpl::QNetworkImpl(int state_dim, int n_actions, int hidden) {
    net = torch::nn::Sequential(
        torch::nn::Linear(state_dim, hidden),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden, hidden),
        torch::nn::ReLU(),
        torch::nn::Linear(hidden, n_actions));
    register_module("net", net);
}

torch::Tensor QNetworkImpl::forward(torch::Tensor x) { return net->forward(x); }

DQNAllocator::DQNAllocator(double epsilon, double epsilon_decay, double epsilon_min,
                           double gamma, double lr, int batch_size, int replay_size,
                           int target_update_freq, int hidden, double beta_d,
                           std::optional<uint64_t> seed)
    : epsilon_(epsilon), initial_epsilon_(epsilon), epsilon_decay_(epsilon_decay),
      epsilon_min_(epsilon_min), gamma_(gamma), beta_d_(beta_d), lr_(lr),
      batch_size_(batch_size), target_update_freq_(target_update_freq),
      hidden_(hidden), rng_(make_rng(seed)), replay_max_(replay_size) {
    if (seed.has_value()) torch::manual_seed(static_cast<int64_t>(*seed));
}

void DQNAllocator::init_networks(int n_actions) {
    q_net_ = QNetwork(state_dim_, n_actions, hidden_);
    target_net_ = QNetwork(state_dim_, n_actions, hidden_);
    // copy weights q -> target
    torch::NoGradGuard no_grad;
    auto src = q_net_->parameters();
    auto dst = target_net_->parameters();
    for (size_t i = 0; i < src.size(); ++i) dst[i].copy_(src[i]);
    optimizer_ = std::make_shared<torch::optim::Adam>(
        q_net_->parameters(), torch::optim::AdamOptions(lr_));
}

torch::Tensor DQNAllocator::state_vector(const Prompt& prompt,
                                         const std::vector<GPU>& gpus,
                                         double current_time,
                                         const EnvObservation* env_state) const {
    int n_gpus = std::max<size_t>(1, gpus.size());
    double load = system_load(gpus, current_time);
    double max_wait = max_queue_wait(gpus, current_time);
    double max_wait_norm = std::min(max_wait / 30.0, 1.0);
    double queue_depth_norm = static_cast<double>(queue_depth(gpus, current_time)) / n_gpus;
    double input_tokens_norm = std::min(prompt.input_tokens / 1024.0, 1.0);
    auto tier_feats = per_tier_state(gpus, current_time);        // 6
    auto power_feats = env_power_features(env_state);            // 4
    auto tier_batch_feats = env_tier_batch_features(env_state);  // 3

    std::vector<float> vec;
    vec.reserve(18);
    vec.push_back(static_cast<float>(prompt.difficulty));
    vec.push_back(static_cast<float>(input_tokens_norm));
    vec.push_back(static_cast<float>(load));
    vec.push_back(static_cast<float>(max_wait_norm));
    vec.push_back(static_cast<float>(queue_depth_norm));
    for (double v : tier_feats) vec.push_back(static_cast<float>(v));
    for (double v : power_feats) vec.push_back(static_cast<float>(v));
    for (double v : tier_batch_feats) vec.push_back(static_cast<float>(v));
    return torch::tensor(vec, torch::kFloat32);
}

void DQNAllocator::reset_for_episode() {
    prev_state_vec_.reset();
    prev_action_idx_.reset();
    prev_reward_.reset();
    pending_decision_.reset();
    epsilon_ = std::max(epsilon_min_, epsilon_);
}

std::optional<AllocationDecision> DQNAllocator::allocate(
    const Prompt& prompt, std::vector<GPU>& gpus, const std::vector<LLM>& llm_pool,
    double current_time, const EnvObservation* env_state) {
    if (!actions_) {
        actions_ = build_actions(llm_pool);
        init_networks(static_cast<int>(actions_->size()));
    }

    torch::Tensor state_vec = state_vector(prompt, gpus, current_time, env_state);

    // legacy static-engine transition store
    if (!evaluation_mode && prev_state_vec_.has_value() && prev_reward_.has_value()) {
        store_transition(*prev_state_vec_, *prev_action_idx_, *prev_reward_,
                         state_vec, false, gamma_);
        train_step();
    }

    int action_idx;
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    if (!evaluation_mode && uni(rng_) < epsilon_) {
        std::uniform_int_distribution<int> pick(0, static_cast<int>(actions_->size()) - 1);
        action_idx = pick(rng_);
    } else {
        torch::NoGradGuard no_grad;
        torch::Tensor q_vals = q_net_->forward(state_vec.unsqueeze(0));
        action_idx = static_cast<int>(q_vals.argmax(1).item<int64_t>());
    }

    prev_state_vec_ = state_vec;
    prev_action_idx_ = action_idx;
    pending_decision_ = std::make_shared<DecisionSnapshot>();
    pending_decision_->dqn_state = state_vec;
    pending_decision_->dqn_action_idx = action_idx;
    pending_decision_->is_dqn = true;

    return action_to_decision((*actions_)[action_idx], gpus, llm_pool, env_state);
}

std::shared_ptr<DecisionSnapshot> DQNAllocator::pop_last_decision() {
    auto d = pending_decision_;
    pending_decision_.reset();
    return d;
}

double DQNAllocator::effective_gamma(std::optional<double> holding_time) const {
    if (beta_d_ <= 0.0 || !holding_time.has_value()) return gamma_;
    return std::exp(-beta_d_ * std::max(0.0, *holding_time));
}

std::shared_ptr<DecisionSnapshot> DQNAllocator::arrival_state(
    const Prompt& prompt, const std::vector<GPU>& gpus, double time, const EnvObservation* observation) {
    auto snapshot = std::make_shared<DecisionSnapshot>();
    snapshot->dqn_state = state_vector(prompt, gpus, time, observation);
    snapshot->is_dqn = true;
    return snapshot;
}

void DQNAllocator::observe_arrival_transition(
    const std::shared_ptr<DecisionSnapshot>& decision, double reward,
    const std::shared_ptr<DecisionSnapshot>& successor, bool done, double elapsed) {
    if (evaluation_mode || !decision) return;
    auto next = done ? torch::zeros({state_dim_}, torch::kFloat32) : successor->dqn_state;
    store_transition(decision->dqn_state, decision->dqn_action_idx, reward, next,
                     done, effective_gamma(elapsed));
    train_step();
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
}

torch::Tensor DQNAllocator::next_vec_from_obs(const torch::Tensor& prev_vec,
                                              const EnvObservation* next_state) const {
    torch::Tensor next_vec = prev_vec.clone();
    auto power_feats = env_power_features(next_state);          // 4
    auto tier_batch_feats = env_tier_batch_features(next_state);  // 3
    std::vector<double> dyn;
    dyn.insert(dyn.end(), power_feats.begin(), power_feats.end());
    dyn.insert(dyn.end(), tier_batch_feats.begin(), tier_batch_feats.end());
    auto acc = next_vec.accessor<float, 1>();
    for (size_t i = 0; i < dyn.size(); ++i) acc[11 + static_cast<int>(i)] = static_cast<float>(dyn[i]);
    return next_vec;
}

void DQNAllocator::observe_transition(
    const std::shared_ptr<DecisionSnapshot>& decision, double reward,
    const EnvObservation* next_state, bool done, std::optional<double> holding_time) {
    if (evaluation_mode || !decision) return;
    torch::Tensor state_vec = decision->dqn_state;
    int action_idx = decision->dqn_action_idx;
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);

    torch::Tensor next_vec;
    if (done) {
        next_vec = torch::zeros({state_dim_}, torch::kFloat32);
    } else {
        next_vec = next_vec_from_obs(state_vec, next_state);
    }
    double gamma_k = effective_gamma(holding_time);
    store_transition(state_vec, action_idx, reward, next_vec, done, gamma_k);
    train_step();
}

void DQNAllocator::update(double reward) {
    if (evaluation_mode) return;
    prev_reward_ = reward;
    epsilon_ = std::max(epsilon_min_, epsilon_ * epsilon_decay_);
}

void DQNAllocator::end_episode() {
    if (evaluation_mode) return;
    if (prev_state_vec_.has_value() && prev_reward_.has_value()) {
        torch::Tensor dummy_next = torch::zeros({state_dim_}, torch::kFloat32);
        store_transition(*prev_state_vec_, *prev_action_idx_, *prev_reward_,
                         dummy_next, true, gamma_);
        train_step();
    }
    prev_state_vec_.reset();
    prev_action_idx_.reset();
    prev_reward_.reset();
    pending_decision_.reset();
}

void DQNAllocator::store_transition(torch::Tensor s, int a, double r,
                                    torch::Tensor s_next, bool done, double gamma) {
    if (static_cast<int>(replay_.size()) >= replay_max_) {
        replay_.erase(replay_.begin());
    }
    replay_.push_back(ReplayItem{std::move(s), a, r, std::move(s_next), done, gamma});
}

void DQNAllocator::train_step() {
    if (static_cast<int>(replay_.size()) < batch_size_) return;

    // seeded sampling without replacement
    std::vector<size_t> idx(replay_.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    for (int i = 0; i < batch_size_; ++i) {
        std::uniform_int_distribution<size_t> pick(i, idx.size() - 1);
        std::swap(idx[i], idx[pick(rng_)]);
    }

    std::vector<torch::Tensor> states, next_states;
    std::vector<int64_t> actions;
    std::vector<float> rewards, dones, gammas;
    states.reserve(batch_size_);
    next_states.reserve(batch_size_);
    for (int i = 0; i < batch_size_; ++i) {
        const ReplayItem& it = replay_[idx[i]];
        states.push_back(it.s);
        next_states.push_back(it.s_next);
        actions.push_back(it.a);
        rewards.push_back(static_cast<float>(it.r));
        dones.push_back(it.done ? 1.0f : 0.0f);
        gammas.push_back(static_cast<float>(it.gamma));
    }

    torch::Tensor s = torch::stack(states);
    torch::Tensor s_next = torch::stack(next_states);
    torch::Tensor a = torch::tensor(actions, torch::kLong);
    torch::Tensor r = torch::tensor(rewards, torch::kFloat32);
    torch::Tensor d = torch::tensor(dones, torch::kFloat32);
    torch::Tensor g = torch::tensor(gammas, torch::kFloat32);

    torch::Tensor q_values = q_net_->forward(s);
    torch::Tensor q_selected = q_values.gather(1, a.unsqueeze(1)).squeeze(1);

    torch::Tensor target;
    {
        torch::NoGradGuard no_grad;
        torch::Tensor next_online = q_net_->forward(s_next);
        torch::Tensor next_actions = next_online.argmax(1, /*keepdim=*/true);
        torch::Tensor q_next = target_net_->forward(s_next).gather(1, next_actions).squeeze(1);
        target = r + g * q_next * (1 - d);
    }

    torch::Tensor loss = torch::nn::functional::mse_loss(q_selected, target);
    optimizer_->zero_grad();
    loss.backward();
    torch::nn::utils::clip_grad_norm_(q_net_->parameters(), 10.0);
    optimizer_->step();

    step_count_ += 1;
    if (step_count_ % target_update_freq_ == 0) {
        torch::NoGradGuard no_grad;
        auto src = q_net_->parameters();
        auto dst = target_net_->parameters();
        for (size_t i = 0; i < src.size(); ++i) dst[i].copy_(src[i]);
    }
}

// ======================================================================
// 4. Multi-episode training loop
// ======================================================================
TrainingCurve train_learner(Allocator& allocator, int n_episodes, int n_prompts,
                            uint64_t seed, int n_small_gpus, int n_medium_gpus,
                            int n_large_gpus, const std::string& objective,
                            bool verbose, const std::string& engine) {
    RewardWeights weights = RewardWeights::profile(objective);
    std::vector<LLM> llm_pool = default_llm_pool();

    TrainingCurve curve;

    for (int ep = 0; ep < n_episodes; ++ep) {
        PromptGenerator gen(0.5, 0.2, 1.0, 256, seed + ep);
        std::vector<Prompt> prompts = gen.generate(n_prompts);

        std::vector<GPU> gpus = create_gpu_pool(n_small_gpus, n_medium_gpus, n_large_gpus);

        allocator.reset_for_episode();

        bool is_learner = allocator.is_learner();
        std::vector<JobResult> results;
        double idle_energy = 0.0;
        int dropped = 0;

        if (engine == "batched") {
            BatchedSimulator sim(gpus, llm_pool, BatchedConfig{weights});
            results = sim.run(allocator, prompts, is_learner);
            idle_energy = sim.total_idle_energy;
            dropped = sim.dropped;
        } else {
            SimulatorConfig scfg;
            scfg.n_prompts = n_prompts;
            scfg.reward_weights = weights;
            Simulator sim(gpus, llm_pool, scfg);
            results = sim.run(allocator, prompts, is_learner);
            idle_energy = sim.total_idle_energy;
            dropped = sim.dropped;
        }

        allocator.end_episode();

        AggregatedMetrics m = aggregate(allocator.name(), results, idle_energy, dropped);
        double eps_val = allocator.epsilon();

        curve.episode.push_back(ep);
        curve.mean_reward.push_back(m.mean_reward);
        curve.total_energy.push_back(m.total_energy);
        curve.system_energy.push_back(m.system_energy);
        curve.mean_quality.push_back(m.mean_quality);
        curve.epsilon.push_back(eps_val);

        if (verbose && (ep % std::max(1, n_episodes / 10) == 0 || ep == n_episodes - 1)) {
            std::printf("  ep %4d/%d | reward %8.3f | sys_energy %8.0fJ | "
                        "quality %.3f | eps=%.4f\n",
                        ep, n_episodes, m.mean_reward, m.system_energy,
                        m.mean_quality, eps_val);
        }
    }

    return curve;
}

}  // namespace llm_alloc
