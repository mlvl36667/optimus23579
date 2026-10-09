#include "test_harness.hpp"

#include <algorithm>
#include <cmath>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/domain.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/simulator.hpp"

using namespace llm_alloc;

// --------------- domain ------------------------------------------------
TEST(domain_size_ordering) {
    CHECK(size_value(Size::SMALL) < size_value(Size::MEDIUM));
    CHECK(size_value(Size::MEDIUM) < size_value(Size::LARGE));
}

TEST(domain_llm_fits_on_gpu) {
    GPU small_gpu(0, Size::SMALL);
    GPU large_gpu(1, Size::LARGE);
    LLM large_llm("L", Size::LARGE, 0.9, 25, 0.04);
    LLM small_llm("S", Size::SMALL, 0.3, 100, 0.005);
    CHECK(!large_llm.fits_on(small_gpu));
    CHECK(large_llm.fits_on(large_gpu));
    CHECK(small_llm.fits_on(small_gpu));
    CHECK(small_llm.fits_on(large_gpu));
}

TEST(domain_gpu_pool_counts) {
    auto gpus = create_gpu_pool(3, 2, 1);
    CHECK(gpus.size() == 6);
    int small = 0, large = 0;
    for (const auto& g : gpus) {
        if (g.vram == Size::SMALL) small++;
        if (g.vram == Size::LARGE) large++;
    }
    CHECK(small == 3);
    CHECK(large == 1);
}

TEST(domain_prompt_generator_deterministic) {
    PromptGenerator g1(0.5, 0.2, 1.0, 256, 7);
    PromptGenerator g2(0.5, 0.2, 1.0, 256, 7);
    auto p1 = g1.generate(10);
    auto p2 = g2.generate(10);
    for (size_t i = 0; i < p1.size(); ++i) {
        CHECK_CLOSE(p1[i].difficulty, p2[i].difficulty, 1e-12);
        CHECK_CLOSE(p1[i].arrival_time, p2[i].arrival_time, 1e-12);
    }
}

TEST(domain_prompt_difficulty_bounds) {
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 0);
    auto prompts = gen.generate(500);
    for (const auto& p : prompts) {
        CHECK(p.difficulty >= 0.1 && p.difficulty <= 1.0);
    }
}

TEST(domain_gpu_idle_power) {
    for (Size s : {Size::SMALL, Size::MEDIUM, Size::LARGE}) {
        GPU gpu(0, s);
        CHECK(gpu.idle_power() > 0);
    }
}

// --------------- allocators --------------------------------------------
TEST(alloc_random_returns_valid) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Prompt p; p.id = 0; p.arrival_time = 0; p.difficulty = 0.5; p.input_tokens = 128;
    RandomAllocator a(1);
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.fits_on(*d->gpu));
}

TEST(alloc_always_small) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    AlwaysSmallAllocator a(1);
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.size == Size::SMALL);
}

TEST(alloc_always_large) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    AlwaysLargeAllocator a(1);
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.size == Size::LARGE);
}

TEST(alloc_difficulty_easy_and_hard) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    DifficultyAllocator a(1);
    Prompt easy; easy.difficulty = 0.15; easy.input_tokens = 128;
    auto d1 = a.allocate(easy, gpus, pool, 0.0, nullptr);
    CHECK(d1.has_value() && d1->llm.size == Size::SMALL);
    Prompt hard; hard.difficulty = 0.9; hard.input_tokens = 128;
    auto d2 = a.allocate(hard, gpus, pool, 0.0, nullptr);
    CHECK(d2.has_value() && d2->llm.size == Size::LARGE);
}

TEST(alloc_least_loaded_picks_free_gpu) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    gpus[0].busy_until = 999.0;
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    LeastLoadedAllocator a;
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->gpu->id != gpus[0].id);
}

TEST(alloc_energy_aware_returns_valid) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    EnergyAwareAllocator a;
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.fits_on(*d->gpu));
}

// --------------- simulator ---------------------------------------------
TEST(sim_run_produces_results) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    SimulatorConfig cfg; cfg.n_prompts = 20;
    Simulator sim(gpus, pool, cfg);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 42);
    auto prompts = gen.generate(20);
    RandomAllocator a(42);
    auto results = sim.run(a, prompts);
    CHECK(!results.empty());
    for (const auto& r : results) {
        CHECK(r.latency >= 0);
        CHECK(r.energy >= 0);
        CHECK(r.quality >= 0 && r.quality <= 1);
    }
}

TEST(sim_quality_degrades_hard_prompt_small_model) {
    GPU gpu(0, Size::SMALL);
    auto pool = default_llm_pool();
    std::vector<LLM> one = {pool[0]};  // SmallLM
    Simulator sim({gpu}, one);
    Prompt hard; hard.difficulty = 0.95; hard.input_tokens = 128;
    AlwaysSmallAllocator a(1);
    auto results = sim.run(a, {hard});
    CHECK(results.size() == 1);
    CHECK(results[0].quality < 0.5);
}

TEST(sim_online_update_populates_q) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Simulator sim(gpus, pool);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 42);
    auto prompts = gen.generate(10);
    QLearningAllocator a(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 42);
    sim.run(a, prompts, true);
    a.end_episode();
    CHECK(a.q_size() > 0);
}

// --------------- learning: epsilon-greedy ------------------------------
TEST(eps_greedy_runs_and_counts) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    EpsilonGreedyAllocator a(0.15, 0.999, 5, 42);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    a.update(-1.5);
    CHECK(a.counts_size() > 0);
}

TEST(eps_greedy_decays) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    EpsilonGreedyAllocator a(0.5, 0.9, 5, 1);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    a.allocate(p, gpus, pool, 0.0, nullptr);
    a.update(-1.0);
    CHECK(a.epsilon() < 0.5);
}

// --------------- learning: Q-learning ----------------------------------
TEST(ql_allocate_returns_valid) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    QLearningAllocator a(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 42);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.fits_on(*d->gpu));
}

TEST(ql_state_action_space) {
    QLearningAllocator a(0.1, 0.95, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, 1);
    CHECK(a.state_space_size() == 60);  // 5*4*3
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(a.action_space_size() == 18);  // 3 sizes * 6 strategies
}

TEST(ql_training_runs) {
    QLearningAllocator a(0.1, 0.95, 0.3, 0.998, 0.05, 5, 4, 3, 0.0, 42);
    auto curve = train_learner(a, 10, 50, 42, 2, 2, 1, "energy", false, "static");
    CHECK(curve.episode.size() == 10);
    CHECK(a.q_size() > 0);
}

// --------------- learning: DQN -----------------------------------------
TEST(dqn_allocate_returns_valid) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    DQNAllocator a(0.3, 0.995, 0.01, 0.95, 1e-3, 64, 5000, 50, 64, 0.0, 42);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 128;
    auto d = a.allocate(p, gpus, pool, 0.0, nullptr);
    CHECK(d.has_value());
    CHECK(d->llm.fits_on(*d->gpu));
}

TEST(dqn_state_dim_is_18) {
    DQNAllocator a(0.3, 0.995, 0.01, 0.95, 1e-3, 64, 5000, 50, 64, 0.0, 42);
    CHECK(a.state_dim() == 18);
    auto gpus = create_gpu_pool(2, 2, 1);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 256;
    auto v = a.state_vector(p, gpus, 0.0, nullptr);
    CHECK(v.size(0) == 18);
}

TEST(dqn_state_dim_gpu_count_independent) {
    DQNAllocator a(0.3, 0.995, 0.01, 0.95, 1e-3, 64, 5000, 50, 64, 0.0, 42);
    Prompt p; p.difficulty = 0.5; p.input_tokens = 256;
    auto vs = create_gpu_pool(1, 1, 1);
    auto vl = create_gpu_pool(10, 6, 4);
    auto v_small = a.state_vector(p, vs, 0.0, nullptr);
    auto v_large = a.state_vector(p, vl, 0.0, nullptr);
    CHECK(v_small.size(0) == 18);
    CHECK(v_large.size(0) == 18);
}

TEST(dqn_training_runs) {
    DQNAllocator a(0.3, 0.995, 0.01, 0.95, 1e-3, 16, 200, 50, 64, 0.0, 42);
    auto curve = train_learner(a, 10, 50, 42, 2, 2, 1, "energy", false, "static");
    CHECK(curve.episode.size() == 10);
}

// --------------- metrics -----------------------------------------------
TEST(metrics_aggregate_empty) {
    auto m = aggregate("test", {});
    CHECK(m.n_jobs == 0);
    CHECK(m.system_energy == 0);
}

TEST(metrics_aggregate_consistent) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    Simulator sim(gpus, pool);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 7);
    auto prompts = gen.generate(50);
    DifficultyAllocator a(7);
    auto results = sim.run(a, prompts);
    auto m = aggregate("difficulty", results, sim.total_idle_energy);
    CHECK(m.n_jobs == (int)results.size());
    CHECK(m.mean_quality > 0);
    CHECK(m.idle_energy >= 0);
    CHECK(m.system_energy >= m.total_energy);
}

// --------------- energy objective ----------------------------------------
TEST(obj_reward_profiles_exist) {
    for (const char* name : {"energy", "balanced", "quality", "latency"}) {
        auto w = RewardWeights::profile(name);
        CHECK(w.energy > 0);
    }
}

TEST(obj_energy_profile_has_min_quality) {
    auto w = RewardWeights::profile("energy");
    CHECK(w.min_quality > 0);
}

TEST(obj_idle_energy_computed) {
    auto gpus = create_gpu_pool(1, 1, 1);
    auto pool = default_llm_pool();
    SimulatorConfig cfg; cfg.reward_weights = RewardWeights::profile("energy");
    Simulator sim(gpus, pool, cfg);
    PromptGenerator gen(0.5, 0.2, 1.0, 256, 42);
    auto prompts = gen.generate(20);
    AlwaysSmallAllocator a(42);
    sim.run(a, prompts);
    CHECK(sim.total_idle_energy > 0);
}

TEST(obj_min_quality_penalty_applied) {
    GPU gpu(0, Size::SMALL);
    auto pool = default_llm_pool();
    std::vector<LLM> one = {pool[0]};
    SimulatorConfig cfg; cfg.reward_weights = RewardWeights::profile("energy");
    Simulator sim({gpu}, one, cfg);
    Prompt hard; hard.difficulty = 0.95; hard.input_tokens = 128;
    AlwaysSmallAllocator a(1);
    auto results = sim.run(a, {hard});
    CHECK(results.size() == 1);
    CHECK(results[0].reward < -5);
}

// --------------- state helpers -------------------------------------------
TEST(helpers_per_tier_state_length) {
    auto gpus = create_gpu_pool(2, 2, 1);
    auto feats = per_tier_state(gpus, 0.0);
    CHECK(feats.size() == 6);
}

TEST(helpers_per_tier_state_missing_tier_zero) {
    auto gpus = create_gpu_pool(2, 0, 0);
    auto feats = per_tier_state(gpus, 0.0);
    CHECK(feats[2] == 0.0 && feats[3] == 0.0);
    CHECK(feats[4] == 0.0 && feats[5] == 0.0);
}

TEST(helpers_queue_depth) {
    auto gpus = create_gpu_pool(2, 1, 1);
    gpus[0].busy_until = 100.0;
    gpus[1].busy_until = 50.0;
    CHECK(queue_depth(gpus, 10.0) == 2);
}

// --------------- reward weight profile throws --------------------------
TEST(obj_unknown_profile_throws) {
    bool threw = false;
    try { RewardWeights::profile("nope"); }
    catch (const std::exception&) { threw = true; }
    CHECK(threw);
}
