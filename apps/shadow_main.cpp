#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <torch/torch.h>

#include "llm_alloc/allocators.hpp"
#include "llm_alloc/batched_simulator.hpp"
#include "llm_alloc/learning.hpp"
#include "llm_alloc/metrics.hpp"
#include "llm_alloc/shadow_price.hpp"

using namespace llm_alloc;
using json = nlohmann::json;

namespace {
struct Options {
    double budget = 120.0, initial_price = 0.1, dual_lr = 0.0005;
    int episodes = 600, dual_period = 20, polish = 200, prompts = 200, eval_streams = 5;
    uint64_t seed = 42;
    std::string output = "experiments/shadow/result.json";
};

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        std::string name = argv[index];
        if (index + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
        std::string value = argv[++index];
        if (name == "--output") { options.output = value; continue; }
        size_t consumed = 0;
        double number = std::stod(value, &consumed);
        if (consumed != value.size() || !std::isfinite(number)) throw std::invalid_argument("Invalid value for " + name);
        if (name == "--budget-j") options.budget = number;
        else if (name == "--initial-price") options.initial_price = number;
        else if (name == "--dual-lr") options.dual_lr = number;
        else {
            if (number < 0 || number > 1000000000 || std::floor(number) != number)
                throw std::invalid_argument("Expected nonnegative integer for " + name);
            int integer = static_cast<int>(number);
            if (name == "--episodes") options.episodes = integer;
            else if (name == "--dual-period") options.dual_period = integer;
            else if (name == "--polish-episodes") options.polish = integer;
            else if (name == "--n-prompts") options.prompts = integer;
            else if (name == "--eval-streams") options.eval_streams = integer;
            else if (name == "--seed") options.seed = integer;
            else throw std::invalid_argument("Unknown option " + name);
        }
    }
    if (options.episodes <= 0 || options.dual_period <= 0 || options.polish < 0 ||
        options.prompts <= 0 || options.eval_streams <= 0)
        throw std::invalid_argument("Episodes, period, prompts and evaluation streams must be positive");
    return options;
}

struct Episode {
    double energy, service, quality, latency;
};

Episode run(Allocator& allocator, double price, const std::vector<Prompt>& prompts, bool train) {
    auto weights = RewardWeights::profile("quality");
    weights.energy = price;
    allocator.reset_for_episode();
    allocator.evaluation_mode = !train;
    BatchedSimulator simulator(create_gpu_pool(2, 2, 1), default_llm_pool(), BatchedConfig{weights});
    auto results = simulator.run(allocator, prompts, train);
    if (results.size() != prompts.size() || simulator.dropped != 0)
        throw std::runtime_error("Constrained experiment requires every arrival to complete");
    double service = 0.0;
    for (const auto& result : results)
        service += weights.latency * result.latency + weights.quality_loss * result.quality_loss +
            weights.min_quality_penalty * std::max(0.0, weights.min_quality - result.quality);
    if (train) allocator.end_episode();
    auto metrics = aggregate(allocator.name(), results, simulator.total_idle_energy, simulator.dropped);
    return {metrics.system_energy, service, metrics.mean_quality, metrics.mean_latency};
}
}

int main(int argc, char** argv) {
    if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        std::printf("Usage: llm_shadow [--budget-j 120] [--initial-price 0.1] [--dual-lr 0.0005]\n"
                    "  [--episodes 600] [--dual-period 20] [--polish-episodes 200]\n"
                    "  [--n-prompts 200] [--eval-streams 5] [--seed 42] [--output PATH]\n"
                    "Tabular Q-learning only; undiscounted quality/latency service cost, 2+2+1 GPUs.\n");
        return 0;
    }
    try {
        auto options = parse(argc, argv);
        EnergyShadowPrice controller(options.budget, options.initial_price, options.dual_lr);
        torch::set_num_threads(1);
        QLearningAllocator shadow(0.1, 1.0, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, options.seed);
        QLearningAllocator fixed(0.1, 1.0, 0.2, 0.998, 0.01, 5, 4, 3, 0.0, options.seed);
        json history = json::array();
        double block_energy = 0.0;
        size_t block_arrivals = 0;
        std::printf("Shadow price: budget=%.3f J/arrival, quality profile service cost, gamma=1, seed=%llu\n",
                    options.budget, static_cast<unsigned long long>(options.seed));
        for (int episode = 0; episode < options.episodes + options.polish; ++episode) {
            PromptGenerator generator(0.5, 0.2, 1.0, 256, options.seed + episode);
            auto prompts = generator.generate(options.prompts);
            double used_price = controller.price();
            auto measured = run(shadow, used_price, prompts, true);
            run(fixed, options.initial_price, prompts, true);
            bool adapting = episode < options.episodes;
            if (adapting) {
                block_energy += measured.energy;
                block_arrivals += prompts.size();
                if ((episode + 1) % options.dual_period == 0 || episode + 1 == options.episodes) {
                    controller.update(block_energy, block_arrivals);
                    block_energy = 0.0;
                    block_arrivals = 0;
                }
            }
            history.push_back({{"episode", episode}, {"phase", adapting ? "dual" : "frozen_price_polish"},
                {"price_used", used_price}, {"price_next", controller.price()},
                {"energy_j_per_arrival", measured.energy / options.prompts},
                {"service_cost_per_arrival", measured.service / options.prompts},
                {"violation_j_per_arrival", measured.energy / options.prompts - options.budget},
                {"quality", measured.quality}, {"latency_s", measured.latency}});
            if ((episode + 1) % options.dual_period == 0 || episode + 1 == options.episodes + options.polish) {
                std::printf("ep=%d phase=%s price=%.6f -> %.6f energy=%.3f quality=%.3f\n",
                    episode + 1, adapting ? "dual" : "polish", used_price, controller.price(),
                    measured.energy / options.prompts, measured.quality);
                std::fflush(stdout);
            }
        }
        auto heuristics = all_heuristic_allocators(options.seed);
        std::vector<std::pair<std::string, Allocator*>> policies{{"q_shadow", &shadow}, {"q_fixed_price", &fixed}};
        for (auto& heuristic : heuristics) policies.emplace_back(heuristic->name(), heuristic.get());
        json evaluation = json::array();
        std::printf("Frozen evaluation: policy energy_J/arrival service_cost quality latency_s violation_J/arrival violating_stream_fraction\n");
        for (auto& policy : policies) {
            double energy = 0.0, service = 0.0, quality = 0.0, latency = 0.0;
            int violations = 0;
            json streams = json::array();
            for (int stream = 0; stream < options.eval_streams; ++stream) {
                PromptGenerator generator(0.5, 0.2, 1.0, 256, options.seed + 1000000 + stream);
                auto measured = run(*policy.second, controller.price(), generator.generate(options.prompts), false);
                double per_arrival = measured.energy / options.prompts;
                energy += per_arrival;
                service += measured.service / options.prompts;
                quality += measured.quality;
                latency += measured.latency;
                violations += per_arrival > options.budget;
                streams.push_back({{"stream", stream}, {"energy_j_per_arrival", per_arrival},
                    {"service_cost_per_arrival", measured.service / options.prompts},
                    {"quality", measured.quality}, {"latency_s", measured.latency}});
            }
            double count = options.eval_streams;
            evaluation.push_back({{"policy", policy.first}, {"energy_j_per_arrival", energy / count},
                {"service_cost_per_arrival", service / count}, {"quality", quality / count},
                {"latency_s", latency / count}, {"violation_j_per_arrival", energy / count - options.budget},
                {"mean_feasible", energy / count <= options.budget},
                {"violating_stream_fraction", violations / count}, {"streams", streams}});
            std::printf("%-20s %.3f %.3f %.3f %.3f %+.3f %.2f\n", policy.first.c_str(),
                energy / count, service / count, quality / count, latency / count,
                energy / count - options.budget, violations / count);
        }
        json report{{"budget_j_per_arrival", options.budget}, {"initial_price", options.initial_price},
            {"final_price", controller.price()}, {"dual_lr", options.dual_lr}, {"dual_period", options.dual_period},
            {"episodes", options.episodes}, {"polish_episodes", options.polish}, {"n_prompts", options.prompts},
            {"seed", options.seed}, {"eval_streams", options.eval_streams},
            {"gamma", 1.0}, {"service_profile", "quality"},
            {"history", history}, {"evaluation", evaluation}};
        auto parent = std::filesystem::path(options.output).parent_path();
        if (!parent.empty()) std::filesystem::create_directories(parent);
        std::ofstream output(options.output);
        output << report.dump(2) << '\n';
        output.close();
        if (!output) throw std::runtime_error("Cannot write " + options.output);
        std::printf("Results: %s\n", options.output.c_str());
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
}
