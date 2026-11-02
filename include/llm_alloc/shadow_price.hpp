#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace llm_alloc {

class EnergyShadowPrice {
public:
    EnergyShadowPrice(double budget_j, double initial_price, double learning_rate)
        : budget_(budget_j), price_(initial_price), learning_rate_(learning_rate) {
        if (!std::isfinite(budget_) || budget_ <= 0.0 || !std::isfinite(price_) || price_ < 0.0 ||
            !std::isfinite(learning_rate_) || learning_rate_ <= 0.0)
            throw std::invalid_argument("Invalid shadow-price budget, initial price or learning rate");
    }

    double price() const { return price_; }
    double budget() const { return budget_; }

    void update(double total_energy, size_t arrivals) {
        if (arrivals == 0 || !std::isfinite(total_energy) || total_energy < 0.0)
            throw std::invalid_argument("Shadow-price update requires finite energy and positive arrivals");
        double next = price_ + learning_rate_ * (total_energy / arrivals - budget_);
        if (!std::isfinite(next)) throw std::overflow_error("Shadow price overflow");
        price_ = std::max(0.0, next);
    }

private:
    double budget_, price_, learning_rate_;
};

}
