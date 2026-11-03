#!/usr/bin/env python3
"""Solve the empirical batched Q-learning model exported by llm_run."""

import argparse
import csv
import math
import sys
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="CSV produced by --q-transition-model")
    parser.add_argument("--gamma", type=float, default=0.95,
                        help="discount factor used by value iteration (default: 0.95)")
    parser.add_argument("--quality-floor", type=float,
                        help="use energy/quality sufficient statistics for a Lagrangian constrained model")
    parser.add_argument("--quality-lambda", type=float,
                        help="nonnegative quality multiplier; requires --quality-floor")
    parser.add_argument("--tolerance", type=float, default=1e-10)
    parser.add_argument("--max-iterations", type=int, default=10000)
    parser.add_argument("--output", "-o", help="output CSV (default: stdout)")
    args = parser.parse_args()
    if not 0.0 < args.gamma <= 1.0:
        parser.error("--gamma must be in (0,1]")
    if args.tolerance <= 0 or args.max_iterations <= 0:
        parser.error("tolerance and max-iterations must be positive")
    if (args.quality_floor is None) != (args.quality_lambda is None):
        parser.error("--quality-floor and --quality-lambda must be supplied together")
    if args.quality_floor is not None and (not 0.0 <= args.quality_floor <= 1.0 or args.quality_lambda < 0.0):
        parser.error("quality floor must be in [0,1] and lambda must be nonnegative")

    actions = set()
    transitions = defaultdict(list)
    sample_totals = {}
    state_meta = {}
    num_states = 0
    with open(args.model, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"state_id", "difficulty_bin", "load_bin", "batch_bin",
                    "model_size", "gpu_rule", "next_state_id", "count",
                    "observations", "probability", "mean_reward"}
        if args.quality_lambda is not None:
            required |= {"mean_energy", "mean_quality_sum", "mean_quality_count"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            parser.error("input is not a transition model CSV from llm_run")
        for row in reader:
            state = int(row["state_id"])
            action = (row["model_size"], row["gpu_rule"])
            next_state = int(row["next_state_id"])
            count = int(row["count"])
            observations = int(row["observations"])
            probability = float(row["probability"])
            if args.quality_lambda is None:
                reward = float(row["mean_reward"])
            else:
                energy = float(row["mean_energy"])
                quality_sum = float(row["mean_quality_sum"])
                quality_count = float(row["mean_quality_count"])
                # Maximize negative energy plus the Lagrangian quality surplus.
                reward = -energy + args.quality_lambda * (
                    quality_sum - args.quality_floor * quality_count)
            num_states = max(num_states, state + 1, next_state + 1)
            state_meta[state] = (row["difficulty_bin"], row["load_bin"], row["batch_bin"])
            actions.add(action)
            key = (state, action)
            if key in sample_totals and sample_totals[key] != observations:
                parser.error(f"inconsistent observation totals for state {state}, action {action}")
            sample_totals[key] = observations
            if observations > 0 and count > 0:
                transitions[(state, action)].append((next_state, probability, reward))

    if not transitions or not actions:
        parser.error("transition model contains no observed state/action transitions")

    available = defaultdict(list)
    for state in range(num_states):
        for action in sorted(actions):
            samples = transitions.get((state, action), [])
            if not samples:
                continue
            p_sum = sum(p for _, p, _ in samples)
            # llm_run versions before 17-digit output wrote probabilities with
            # default stream precision; renormalize the small CSV rounding error.
            if not math.isfinite(p_sum) or p_sum <= 0.0 or abs(p_sum - 1.0) > 1e-3:
                parser.error(f"probabilities for state {state}, action {action} sum to {p_sum}, not 1")
            transitions[(state, action)] = [(nxt, p / p_sum, reward)
                                             for nxt, p, reward in samples]
            available[state].append((action, transitions[(state, action)]))

    values = [0.0] * num_states
    q_values = {}
    for iteration in range(1, args.max_iterations + 1):
        delta = 0.0
        updated = values.copy()
        for state in range(num_states):
            choices = available.get(state, ())
            if not choices:
                continue
            qs = []
            for action, samples in choices:
                q = sum(p * (reward + (args.gamma * values[nxt] if nxt >= 0 else 0.0))
                        for nxt, p, reward in samples)
                q_values[(state, action)] = q
                qs.append(q)
            updated[state] = max(qs)
            delta = max(delta, abs(updated[state] - values[state]))
        values = updated
        if delta < args.tolerance:
            break
    else:
        print("warning: value iteration reached max iterations before tolerance", file=sys.stderr)

    out = open(args.output, "w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.writer(out)
        writer.writerow(["state_id", "difficulty_bin", "load_bin", "batch_bin",
                         "model_size", "gpu_rule", "observations", "q_star_model",
                         "v_star_model", "pi_star_model", "identified"])
        for state in range(num_states):
            meta = state_meta.get(state, ("", "", ""))
            choices = available.get(state, ())
            best_action = None
            if choices:
                best_action = max(choices, key=lambda item: q_values[(state, item[0])])[0]
            for action in sorted(actions):
                samples = transitions.get((state, action), [])
                obs = sample_totals.get((state, action), 0)
                writer.writerow([state, *meta, action[0], action[1], obs,
                                 f"{q_values[(state, action)]:.12g}" if (state, action) in q_values else "",
                                 f"{values[state]:.12g}" if choices else "",
                                 action[0] + "/" + action[1] if action == best_action else "",
                                 "yes" if samples else "no"])
    finally:
        if args.output:
            out.close()
    uncovered = sum(not available.get(s) for s in range(num_states))
    missing_pairs = num_states * len(actions) - sum(len(v) for v in available.values())
    print(f"value iteration: {iteration} iterations; {num_states} states, {len(actions)} actions; "
          f"{missing_pairs} unobserved state/action pairs; {uncovered} states without any observed action",
          file=sys.stderr)
    print("q_star_model and pi_star_model are optimal only for this empirical, partially observed model; "
          "unobserved actions are excluded.", file=sys.stderr)
    if args.quality_lambda is not None:
        print(f"model reward: -energy + {args.quality_lambda:g} * (quality - "
              f"{args.quality_floor:g}) per completed prompt", file=sys.stderr)


if __name__ == "__main__":
    main()
