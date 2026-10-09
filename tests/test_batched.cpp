#include "test_harness.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;

// --------------- domain: KV-cache --------------------------------------
TEST(kv_bigger_gpu_more_slots) {
    auto pool = default_llm_pool();
    const LLM& small_llm = pool[0];
    int s = GPU(0, Size::SMALL).kv_cache_slots(small_llm);
    int m = GPU(1, Size::MEDIUM).kv_cache_slots(small_llm);
    int l = GPU(2, Size::LARGE).kv_cache_slots(small_llm);
    CHECK(s < m && m < l);
}

TEST(kv_bigger_model_fewer_slots) {
    GPU g(0, Size::LARGE);
    auto pool = default_llm_pool();
    CHECK(g.kv_cache_slots(pool[0]) > g.kv_cache_slots(pool[2]));
}

TEST(kv_slots_at_least_one) {
    GPU g(0, Size::SMALL);
    for (const auto& llm : default_llm_pool()) {
        if (llm.fits_on(g)) CHECK(g.kv_cache_slots(llm) >= 1);
    }
}

// --------------- throughput physics ------------------------------------
TEST(tp_throughput_increases_with_batch) {
    auto pool = default_llm_pool();
    GPU g(0, Size::LARGE);
    const LLM& llm = pool[2];
    double prev = 0.0;
    for (int b : {1, 2, 4, 8, 16, 32}) {
        double tps = batch_tokens_per_sec(llm, g, b);
        CHECK(tps > prev);
        prev = tps;
    }
}

TEST(tp_throughput_sublinear) {
    auto pool = default_llm_pool();
    GPU g(0, Size::LARGE);
    const LLM& llm = pool[2];
    double t8 = batch_tokens_per_sec(llm, g, 8);
    double t16 = batch_tokens_per_sec(llm, g, 16);
    CHECK(t16 < 2 * t8);
}

TEST(tp_energy_per_token_drops_with_batch) {
    auto pool = default_llm_pool();
    GPU g(0, Size::LARGE);
    const LLM& llm = pool[2];
    int cap = g.kv_cache_slots(llm);
    double dt = 0.05;
    auto e_per_token = [&](int b) {
        double power = step_power(g, b, cap);
        double step = decode_step_duration(llm, g, b);
        double tokens_tick = (dt / step) * b;
        return (power * dt) / tokens_tick;
    };
    CHECK(e_per_token(8) < e_per_token(1));
    CHECK(e_per_token(32) < e_per_token(8));
}

TEST(tp_step_power_bounds) {
    GPU g(0, Size::LARGE);
    int cap = g.kv_cache_slots(default_llm_pool()[2]);
    CHECK(step_power(g, 0, cap) == g.idle_power());
    CHECK(step_power(g, cap, cap) <= g.active_power() + 1e-6);
}

// --------------- simulator behaviour -----------------------------------
TEST(bsim_runs_and_completes_all) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    PromptGenerator gen(0.5, 0.2, 0.3, 256, 42);
    auto prompts = gen.generate(80);
    LeastLoadedAllocator a;
    auto results = sim.run(a, prompts);
    CHECK((int)results.size() + sim.dropped == 80);
    for (const auto& r : results) {
        CHECK(r.latency >= 0);
        CHECK(r.energy >= 0);
        CHECK(r.quality >= 0 && r.quality <= 1);
    }
}

TEST(bsim_continuous_admission_batches) {
    GPU gpu(0, Size::LARGE);
    auto full = default_llm_pool();
    std::vector<LLM> pool = {full[2]};  // only LargeLM
    BatchedSimulator sim({gpu}, pool, BatchedConfig());
    std::vector<Prompt> prompts;
    for (int i = 0; i < 10; ++i) {
        Prompt p; p.id = i; p.arrival_time = 0.0; p.difficulty = 0.9; p.input_tokens = 128;
        prompts.push_back(p);
    }
    LeastLoadedAllocator a;
    auto results = sim.run(a, prompts);
    CHECK(results.size() == 10);
    double makespan = 0.0;
    for (const auto& r : results) makespan = std::max(makespan, r.end_time);
    int solo_tokens = compute_output_tokens(prompts[0], pool[0]);
    double solo_time = solo_tokens / pool[0].base_speed;
    CHECK(makespan < 10 * solo_time);
}

TEST(bsim_dropped_when_no_compatible_gpu) {
    auto gpus = create_gpu_pool(2, 0, 0);  // only small GPUs
    auto pool = default_llm_pool();
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    Prompt hard; hard.difficulty = 0.95; hard.input_tokens = 128;
    DifficultyAllocator a(1);
    auto results = sim.run(a, {hard});
    CHECK(sim.dropped == 1);
    CHECK(results.empty());
}

TEST(bsim_metrics_compatible) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    BatchedConfig cfg{RewardWeights::profile("energy")};
    BatchedSimulator sim(gpus, pool, cfg);
    PromptGenerator gen(0.5, 0.2, 0.3, 256, 7);
    auto prompts = gen.generate(60);
    LeastLoadedAllocator a;
    auto results = sim.run(a, prompts);
    auto m = aggregate("least_loaded", results, sim.total_idle_energy, sim.dropped);
    CHECK(m.n_jobs == (int)results.size());
    CHECK(m.system_energy >= m.total_energy);
    CHECK(m.mean_quality > 0);
}

TEST(bsim_batching_more_efficient_than_spread) {
    auto pool = default_llm_pool();
    BatchedConfig cfg{RewardWeights::profile("energy")};

    auto gpus1 = create_gpu_pool(1, 1, 1);
    BatchedSimulator sim1(gpus1, pool, cfg);
    std::vector<Prompt> bursty;
    for (int i = 0; i < 40; ++i) {
        Prompt p; p.id = i; p.arrival_time = i * 0.02; p.difficulty = 0.5; p.input_tokens = 128;
        bursty.push_back(p);
    }
    LeastLoadedAllocator a1;
    auto res1 = sim1.run(a1, bursty);

    auto gpus2 = create_gpu_pool(1, 1, 1);
    BatchedSimulator sim2(gpus2, pool, cfg);
    std::vector<Prompt> spread;
    for (int i = 0; i < 40; ++i) {
        Prompt p; p.id = i; p.arrival_time = i * 20.0; p.difficulty = 0.5; p.input_tokens = 128;
        spread.push_back(p);
    }
    LeastLoadedAllocator a2;
    auto res2 = sim2.run(a2, spread);

    double e_bursty = 0.0, e_spread = 0.0;
    for (const auto& r : res1) e_bursty += r.energy;
    for (const auto& r : res2) e_spread += r.energy;
    e_bursty /= std::max<size_t>(1, res1.size());
    e_spread /= std::max<size_t>(1, res2.size());
    CHECK(e_bursty < e_spread);
}

TEST(bsim_online_update_populates_q) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    PromptGenerator gen(0.5, 0.2, 0.3, 256, 42);
    auto prompts = gen.generate(40);
    QLearningAllocator a(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 42);
    sim.run(a, prompts, true);
    CHECK(a.q_size() > 0);
}

// --------------- power management --------------------------------------
TEST(pm_sleep_power_below_idle) {
    for (Size s : {Size::SMALL, Size::MEDIUM, Size::LARGE}) {
        GPU g(0, s);
        CHECK(g.sleep_power() < g.idle_power());
    }
}

TEST(pm_unused_gpus_sleep_and_save_energy) {
    auto pool = default_llm_pool();
    std::vector<Prompt> prompts;
    for (int i = 0; i < 60; ++i) {
        Prompt p; p.id = i; p.arrival_time = i * 0.3; p.difficulty = 0.4; p.input_tokens = 150;
        prompts.push_back(p);
    }
    auto gpus_pm = create_gpu_pool(4, 4, 2);
    BatchedConfig cfg_pm; cfg_pm.power_management = true; cfg_pm.sleep_timeout = 3.0;
    BatchedSimulator sim_pm(gpus_pm, pool, cfg_pm);
    LeastLoadedAllocator a1;
    auto res_pm = sim_pm.run(a1, prompts);
    double e_pm = sim_pm.total_idle_energy;
    for (const auto& r : res_pm) e_pm += r.energy;

    auto gpus_no = create_gpu_pool(4, 4, 2);
    BatchedConfig cfg_no; cfg_no.power_management = false;
    BatchedSimulator sim_no(gpus_no, pool, cfg_no);
    LeastLoadedAllocator a2;
    auto res_no = sim_no.run(a2, prompts);
    double e_no = sim_no.total_idle_energy;
    for (const auto& r : res_no) e_no += r.energy;

    CHECK(e_pm < e_no);
}

TEST(pm_wake_up_adds_latency) {
    GPU gpu(0, Size::LARGE);
    auto full = default_llm_pool();
    std::vector<LLM> pool = {full[2]};
    BatchedConfig cfg; cfg.power_management = true; cfg.sleep_timeout = 1.0;
    BatchedSimulator sim({gpu}, pool, cfg);
    std::vector<Prompt> prompts;
    { Prompt p; p.id = 0; p.arrival_time = 0.0; p.difficulty = 0.9; p.input_tokens = 64; prompts.push_back(p); }
    { Prompt p; p.id = 1; p.arrival_time = 50.0; p.difficulty = 0.9; p.input_tokens = 64; prompts.push_back(p); }
    LeastLoadedAllocator a;
    auto results = sim.run(a, prompts);
    CHECK(results.size() == 2);
    const JobResult* second = nullptr;
    for (const auto& r : results) if (r.prompt.id == 1) second = &r;
    CHECK(second != nullptr);
    CHECK(second->latency >= gpu.wake_up_latency() * 0.5);
}

// --------------- objective correctness ---------------------------------
TEST(obj_quality_objective_rewards_quality) {
    auto pool = default_llm_pool();
    BatchedConfig cfg{RewardWeights::profile("quality")};
    cfg.power_management = true; cfg.sleep_timeout = 5.0;
    std::vector<Prompt> prompts;
    for (int i = 0; i < 150; ++i) {
        Prompt p; p.id = i; p.arrival_time = i * 0.3; p.difficulty = 0.5; p.input_tokens = 200;
        prompts.push_back(p);
    }
    auto eval = [&](std::unique_ptr<Allocator> a) {
        auto gpus = create_gpu_pool(4, 4, 2);
        BatchedSimulator sim(gpus, pool, cfg);
        auto res = sim.run(*a, prompts);
        return aggregate(a->name(), res, sim.total_idle_energy, sim.dropped);
    };
    auto large = eval(std::make_unique<AlwaysLargeAllocator>(1));
    auto small = eval(std::make_unique<AlwaysSmallAllocator>(1));
    CHECK(large.mean_quality > small.mean_quality);
    CHECK(large.mean_reward > small.mean_reward);
}

// --------------- wake_sleeping action ----------------------------------
static EnvObservation mk_env(const std::vector<GPU>& gpus,
                             const std::vector<int>& sleeping_ids) {
    EnvObservation env;
    env.time = 0.0;
    for (const auto& g : gpus) {
        GPUObservation o;
        o.gpu_id = g.id;
        o.vram = g.vram;
        bool asleep = std::find(sleeping_ids.begin(), sleeping_ids.end(), g.id) != sleeping_ids.end();
        o.power_state = asleep ? "sleep" : "idle";
        o.active_count = 0;
        o.capacity = g.kv_cache_slots(default_llm_pool()[0]);
        o.waiting_count = 0;
        env.gpus.emplace(g.id, o);
    }
    return env;
}

TEST(ws_registered_in_action_space) {
    bool found = std::find(kGpuStrategies.begin(), kGpuStrategies.end(),
                           "wake_sleeping") != kGpuStrategies.end();
    CHECK(found);
    CHECK(build_actions(default_llm_pool()).size() == 18);
}

TEST(ws_picks_sleeping_gpu) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    const LLM& small = pool[0];
    auto env = mk_env(gpus, {1});
    GPU* picked = pick_gpu(gpus, small, "wake_sleeping", &env);
    CHECK(picked != nullptr);
    CHECK(picked->id == 1);
}

TEST(ws_fallback_least_loaded_when_none_sleeping) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    const LLM& small = pool[0];
    for (auto& g : gpus) g.busy_until = 100.0;
    gpus[0].busy_until = 1.0;
    auto env = mk_env(gpus, {});
    GPU* picked = pick_gpu(gpus, small, "wake_sleeping", &env);
    CHECK(picked != nullptr && picked->id == gpus[0].id);
}

TEST(ws_fallback_without_env_state) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    const LLM& small = pool[0];
    for (auto& g : gpus) g.busy_until = 100.0;
    gpus[2].busy_until = 1.0;
    GPU* picked = pick_gpu(gpus, small, "wake_sleeping", nullptr);
    CHECK(picked != nullptr && picked->id == gpus[2].id);
}

// --------------- tier targeting ----------------------------------------
TEST(tier_targets_requested_tier) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    const LLM& small = pool[0];
    GPU* ps = pick_gpu(gpus, small, "gpu_small", nullptr);
    GPU* pm = pick_gpu(gpus, small, "gpu_medium", nullptr);
    GPU* pl = pick_gpu(gpus, small, "gpu_large", nullptr);
    CHECK(ps && ps->vram == Size::SMALL);
    CHECK(pm && pm->vram == Size::MEDIUM);
    CHECK(pl && pl->vram == Size::LARGE);
}

TEST(tier_graceful_fallback_incompatible) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    const LLM& large = pool[2];
    GPU* picked = pick_gpu(gpus, large, "gpu_small", nullptr);
    CHECK(picked != nullptr);
    CHECK(picked->vram == Size::LARGE);  // fallback respects fits_on
}

TEST(tier_action_to_decision_valid_for_all) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto pool = default_llm_pool();
    for (const auto& action : build_actions(pool)) {
        auto d = action_to_decision(action, gpus, pool, nullptr);
        CHECK(d.has_value());
        CHECK(d->llm.fits_on(*d->gpu));
    }
}

// --------------- SMDP sojourn discount ---------------------------------
TEST(smdp_effective_gamma_default_fixed) {
    QLearningAllocator ql(0.1, 0.9, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 0);
    CHECK_CLOSE(ql.effective_gamma(0.5), 0.9, 1e-12);
    CHECK_CLOSE(ql.effective_gamma(10.0), 0.9, 1e-12);
    CHECK_CLOSE(ql.effective_gamma(std::nullopt), 0.9, 1e-12);
}

TEST(smdp_effective_gamma_sojourn_dependent) {
    QLearningAllocator ql(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.2, 0);
    CHECK_CLOSE(ql.effective_gamma(1.0), std::exp(-0.2), 1e-9);
    CHECK_CLOSE(ql.effective_gamma(5.0), std::exp(-1.0), 1e-9);
    CHECK(ql.effective_gamma(5.0) < ql.effective_gamma(1.0));
    CHECK_CLOSE(ql.effective_gamma(std::nullopt), 0.95, 1e-12);
}

TEST(smdp_qlearning_trains_with_beta_d) {
    auto pool = default_llm_pool();
    auto gpus = create_gpu_pool(2, 2, 1);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 4);
    auto prompts = gen.generate(40);
    QLearningAllocator ql(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.1, 4);
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    sim.run(ql, prompts, true);
    CHECK(ql.q_size() > 0);
}

TEST(smdp_dqn_effective_gamma_and_training) {
    DQNAllocator dqn(0.3, 0.995, 0.01, 0.95, 1e-3, 8, 200, 50, 64, 0.2, 0);
    CHECK_CLOSE(dqn.effective_gamma(2.0), std::exp(-0.4), 1e-6);
    CHECK_CLOSE(dqn.effective_gamma(std::nullopt), 0.95, 1e-12);
    auto pool = default_llm_pool();
    auto gpus = create_gpu_pool(2, 2, 1);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 5);
    auto prompts = gen.generate(60);
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    sim.run(dqn, prompts, true);
    CHECK(dqn.replay_size() > 0);
}

// --------------- learner integration -----------------------------------
TEST(integ_qlearning_learns_on_batched) {
    auto pool = default_llm_pool();
    auto gpus = create_gpu_pool(2, 2, 1);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 11);
    auto prompts = gen.generate(60);
    QLearningAllocator a(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 42);
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    sim.run(a, prompts, true);
    CHECK(a.q_size() > 0);
}

TEST(integ_dqn_learns_on_batched) {
    auto pool = default_llm_pool();
    auto gpus = create_gpu_pool(2, 2, 1);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 11);
    auto prompts = gen.generate(80);
    DQNAllocator a(0.3, 0.995, 0.01, 0.95, 1e-3, 16, 500, 50, 64, 0.0, 42);
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    sim.run(a, prompts, true);
    CHECK(a.replay_size() > 0);
}

// --------------- backward compat ---------------------------------------
TEST(compat_heuristic_online_update_noop) {
    auto pool = default_llm_pool();
    auto gpus = create_gpu_pool(1, 1, 1);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 3);
    auto prompts = gen.generate(15);
    BatchedSimulator sim(gpus, pool, BatchedConfig());
    DifficultyAllocator a(1);
    auto results = sim.run(a, prompts, true);
    CHECK(!results.empty());
}
