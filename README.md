# optimus23579

A C++20 reinforcement-learning simulator that allocates incoming LLM prompts
across a heterogeneous GPU fleet, jointly optimising latency, energy, and answer
quality. Uses **LibTorch** for the DQN allocator and **nlohmann/json** for JSON
export.

## How it works

The starting point is a small, self-contained model of an LLM inference service.
A **fleet** of GPUs is given, each belonging to one of three VRAM tiers —
`SMALL`, `MEDIUM`, or `LARGE` — and each tier has fixed physical characteristics:
idle, sleep and peak-active power draw, a wake-up latency and a one-off wake-up
energy cost, and a KV-cache budget that determines how many sequences it can
decode at once. Alongside the hardware there is a **pool of models**, again in
three sizes: a small model is fast, cheap, and weak; a large model is slow,
power-hungry, and capable. A model only fits on a GPU whose VRAM tier is at
least as large as the model's. **Prompts** arrive over time as a Poisson process
(exponential inter-arrival times) and carry a difficulty drawn from a clipped
Gaussian; difficulty is what makes model choice matter, because a model's
*capability* must keep up with it or the answer quality degrades.

The decision we study is deceptively simple: for every incoming prompt, pick a
`(GPU, model)` pair. That choice is made by an **allocator**, and the whole
project is about comparing and *learning* good allocators. Every served prompt
is scored by a reward that is the negative weighted sum of three costs — the
end-to-end **latency** (queueing plus generation), the **energy** in joules
(power × time, including each job's attributed share of the fleet's idle/sleep
overhead), and the **quality loss** (how far the model fell short of the
prompt's difficulty), with an optional soft penalty when quality drops below a
floor. Four named **objective profiles** — `energy`, `balanced`, `quality`,
`latency` — simply reweight those three terms so that each objective's intended
cost dominates. This reward is the single number an allocator is judged on.

Two simulation engines turn a decision into those numbers. The **static** engine
is a clean baseline: one job per GPU, a simple queue, processing time and energy
derived directly from the model's speed and the GPU's power. The **batched**
engine is the realistic one: it models continuous batching the way modern
servers (vLLM, TGI) actually work. Each GPU holds an active batch of in-flight
sequences; every tick it emits one token per sequence; finished sequences leave
and waiting ones are admitted mid-flight up to the KV-cache limit. Throughput
rises *sub-linearly* with batch size and energy-per-token *falls*, so batching
is rewarded. On top of this sits a power-state machine — a GPU that has been
idle past a timeout goes to **sleep** (drawing almost nothing but clearing its
VRAM), and must pay a wake-up latency and energy to come back — which is what
makes consolidating work onto a few busy GPUs genuinely cheaper than spreading
it thin.

We solve the allocation problem with two families of methods. The **heuristics**
are fixed rules: random, always-small, always-large, difficulty-thresholded,
least-loaded, and an energy-aware greedy chooser. They need no training and
serve as the yardstick. The **learners** treat the problem as a (semi-)Markov
decision process: they observe a compact state — the prompt's difficulty bin,
the fleet's load, and (on the batched engine) its batch-saturation and
sleep/wake dynamics — and choose a composite action that is a *model size*
crossed with a *GPU-pick strategy* (least-loaded, VRAM-matching, waking a
sleeper, or targeting a specific tier). An **ε-greedy contextual bandit** learns
a per-difficulty value table; a **tabular Q-learning** agent learns state→action
values with TD(0); and a **DQN** (a LibTorch MLP trained with Double-DQN,
experience replay, and gradient clipping) learns the same mapping from a richer
continuous state vector. Because jobs finish at different times, the learners use
proper per-transition credit assignment and an optional SMDP sojourn discount
`γ = exp(−β·σ)` that discounts a transition by how long it was in flight.

What you get out is a comparison. Training a learner over many episodes produces
a convergence curve (reward, energy, quality, and ε over time); evaluating any
allocator on a held-out prompt stream produces aggregate metrics — mean reward,
mean and p95 latency, active/idle/system energy, mean quality, per-GPU and
per-tier utilisation, and the number of dropped prompts. The tooling can
therefore answer questions like: which policy wins under an energy budget versus
a latency budget; how energy-per-request scales with fleet size; how the sleep
timeout trades idle energy against wake-up cost; and how answer quality holds up
as prompts get harder. Results are printed as a table and exported as JSON for
external plotting.

The model is deliberately an **abstraction**, and its limits follow from that.
Power, speed, KV-cache and quality are smooth analytic functions of a few tiers,
not measurements from real hardware or real models, so absolute numbers are only
as meaningful as those constants; the token counts, difficulty, and
quality-degradation curves are stylised. The state and action spaces are kept
small and fleet-size-independent on purpose, which keeps learning tractable but
also caps how fine-grained a policy can be. Everything runs single-node on CPU,
and the arrival process is open-loop (no feedback from congestion to the arrival
rate). To take it further, the natural next steps are to **calibrate** the
physics constants against real serving traces, **enrich** the state and action
spaces (per-GPU observations, explicit model-switching costs, request priorities
or SLAs), add **model-based or policy-gradient** learners alongside the
value-based ones, extend the engine to **multi-node / networked** fleets, and
replace the stylised quality model with one fit to actual model-vs-difficulty
behaviour.

## Calibration against real GPUs

The GPU **power** constants are no longer arbitrary — they are calibrated
against two physical cards in the development host using
[`tools/measure_gpu.py`](tools/measure_gpu.py). That script needs only a
CUDA-capable PyTorch and the `nvidia-smi` binary (no `pynvml`, `transformers`,
or internet): it samples board power with `nvidia-smi` while driving each card
with a sustained FP16 GEMM (a stand-in for the dense matmuls that dominate LLM
decode), and reports idle power, active power, and sustained TFLOP/s.

Measured on this host (raw numbers in
[`tools/gpu_measurements.json`](tools/gpu_measurements.json)):

| Tier → card         | VRAM  | idle W | active W | FP16 TFLOP/s |
|---------------------|-------|--------|----------|--------------|
| `SMALL` ← RTX 3070 Ti | 8 GB  | 74     | 249      | 43           |
| `LARGE` ← RTX 3090    | 24 GB | 128    | 349      | 64           |
| `MEDIUM`              | —     | 100    | 300      | —            |

These feed `GPU::idle_power()`, `GPU::active_power()` and
`GPU::power_multiplier()` in `src/domain.cpp`. The `MEDIUM` tier has no physical
card, so its values are **interpolated** between the two measured cards. The
remaining constants that this setup cannot measure — `sleep_power`,
`wake_up_latency`, `wake_up_energy`, and the KV-cache slot counts — are left as
**estimates** and are commented as such in the source.

To re-measure on a different machine:

```bash
python3 tools/measure_gpu.py --json tools/gpu_measurements.json
# then edit the measured() constants in src/domain.cpp accordingly
```

Note that the CUDA device order (`cuda:0`, `cuda:1`) is by compute capability
and need not match the `nvidia-smi` index order, so the script identifies cards
by **name**, not index.

### Model decode speed and energy (real LLMs via Ollama)

The per-model constants — decode **speed** (`base_speed`, tokens/s) and
**energy-per-token** (`energy_per_token`) in `default_llm_pool()` — are
calibrated from three real quantised Llama models run through a local
[Ollama](https://ollama.com) server using
[`tools/measure_ollama.py`](tools/measure_ollama.py). It reads back Ollama's
server-side decode token count and duration while sampling board power with
`nvidia-smi`, and identifies the active card by GPU utilisation (Ollama ignores
`CUDA_VISIBLE_DEVICES` on this host and always schedules onto the RTX 3090).

Measured on the RTX 3090 (raw numbers in
[`tools/ollama_measurements.json`](tools/ollama_measurements.json)):

| Tier → model          | params | decode tok/s | J/token (on 3090) |
|-----------------------|--------|--------------|-------------------|
| `SMALL` ← llama3.2:1b  | 1B     | 390          | 0.65              |
| `MEDIUM` ← llama3.2:3b | 3B     | 235          | 1.44              |
| `LARGE` ← llama3.1:8b  | 8B     | 129          | 2.70              |

`base_speed` is the measured tokens/s directly. `energy_per_token` is stored as
the measured J/token **divided by the measurement card's `power_multiplier`**
(the 3090 is the `LARGE` tier, ×1.4), because `compute_energy()` multiplies
`energy_per_token · base_speed · power_multiplier` — so a model running on its
matching tier reproduces the measured J/token. A model's `capability` (answer
quality vs. prompt difficulty) is *not* decode physics and remains a stylised
estimate.

All three models were measured on the same card (the 3090), so the numbers
capture how speed/energy scale **with model size**, holding the GPU fixed; the
across-GPU scaling is covered by `measure_gpu.py` above. To re-measure:

```bash
ollama pull llama3.2:1b llama3.2:3b llama3.1:8b   # if not already local
python3 tools/measure_ollama.py \
    --models llama3.2:1b llama3.2:3b llama3.1:latest \
    --json tools/ollama_measurements.json
# then edit default_llm_pool() in src/domain.cpp accordingly
```

## Dependencies

- A C++20 compiler and CMake ≥ 3.18.
- **LibTorch** (CPU build). The project requires C++20 to match current
  PyTorch C++ headers and uses `TORCH_CXX_FLAGS` to match the installed Torch
  ABI. Prefer the CPU-only PyTorch wheel; CUDA-enabled wheels require a matching
  CUDA toolkit even though this project runs on CPU.
- **nlohmann/json** (header-only; the system install at `/usr/include` is used
  for the benchmark/sweep JSON export).

## Build

On Debian/Ubuntu, install the compiler, CMake, Python virtual-environment
support, and JSON development dependency:

```bash
sudo apt install build-essential cmake python3-venv nlohmann-json3-dev
```

If LibTorch is not already installed, install CPU PyTorch in a virtual
environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Configure from the repository root and explicitly bind CMake to the virtual
environment's Torch package. A separate build directory avoids stale cache
entries from a prior CUDA-enabled Torch installation:

```bash
source .venv/bin/activate
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release \
  -DTorch_DIR="$PWD/.venv/lib/python3.10/site-packages/torch/share/cmake/Torch" \
  -DCMAKE_PREFIX_PATH="$PWD/.venv/lib/python3.10/site-packages/torch"
cmake --build build-cpu -j4
```

If using a Python version other than 3.10, replace `python3.10` in the Torch
paths with that version's `pythonX.Y` directory. Resolve configuration errors
before building; a failed configure does not generate build files.

## Run

### Energy constraint with a learned shadow price

`llm_shadow` is an opt-in tabular Q-learning experiment. It minimizes the
undiscounted episode service cost of the `quality` profile (latency and quality
loss, without its fixed energy coefficient), subject to expected full-fleet
energy per arrival being at most `--budget-j`. Gamma is 1 for both learned
policies, matching the undiscounted constraint; the existing runners are unchanged.
All arrivals must finish: dropped or incomplete workloads abort this experiment.

For fixed lambda, training reward is `-(service_cost + lambda * energy)` per
arrival interval, including idle, sleep, wake and final drain. The Lagrangian's
`lambda * budget * number_of_arrivals` term is omitted because the arrival count
is fixed, making that term policy-independent in the undiscounted episode sum.
Every `--dual-period` episodes, projected dual ascent applies
`lambda = max(0, lambda + dual_lr * (block_energy/block_arrivals - budget))`.
Price remains fixed inside each block. A final `--polish-episodes` phase trains
with the last price frozen, and evaluation freezes both price and policy.

The equally trained `q_fixed_price` control keeps `--initial-price` throughout;
seven heuristics are evaluated on the same held-out streams. The report separates
service cost, energy, mean budget feasibility and the fraction of evaluation
streams exceeding the budget; rewards under different prices are not ranked.
History records price used/next and budget violation for every episode.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/llm_shadow --budget-j 120 --episodes 600 --dual-period 20 \
  --polish-episodes 200 --initial-price 0.1 --dual-lr 0.0005 \
  --eval-streams 5 --seed 42 --output experiments/shadow/budget_120_seed_42.json
```

This is approximate primal-dual learning, not a feasibility guarantee. Changing
lambda makes old tabular estimates stale; block updates and frozen-price polish
mitigate but do not eliminate this issue. The observation remains partially
observed; legacy `--horizon-state 0` also omits the finite horizon. Infeasible budgets may cause price
to grow without obtaining feasibility. One seed and a few held-out streams do not
establish statistical reliability. DQN is deliberately not included: supporting
adaptive prices there requires replay reward decomposition/relabeling or a
price-conditioned value function rather than mixing old-price scalar rewards.

Initial seed-42 check (600 adaptive + 200 polish episodes, five held-out streams):

| Budget (J/arrival) | Final lambda | Shadow policy energy (J/arrival) | Mean feasible |
| --- | --- | --- | --- |
| 120 | 3.871 | 316.957 | no |
| 150 | 3.381 | 322.639 | no |
| 180 | 2.954 | 380.942 | no |

Why did the learned policies exceed the budgets? Projected dual ascent penalizes
violations; it does not prevent an allocation or enforce a hard execution budget.
It relies on the inner learner finding a sufficiently good policy at each price.
Here the price increased in response to excess energy, but the learned policies
remained expensive. A feasible consolidation control exists on the evaluated
streams, so these failures cannot be attributed simply to impossible budgets.

Two plausible causes require separate experiments. First, the same Q table is
reused when the price changes every 20 episodes, although earlier fixed-objective
experiments needed hundreds of episodes to learn consolidation. Second, gamma=1
was introduced to align with total episode cost, but the state does not encode
remaining arrivals: one table entry mixes different remaining horizons. Neither
cause is established by the controller arithmetic checks alone.

The next diagnostic should hold lambda fixed with gamma=1 and determine whether
the inner learner can learn an energy-efficient policy before adapting lambda.
`q_fixed_price` in the existing runner is the matched control for this check;
repeat runs at different `--initial-price` values and training durations, and
inspect that policy's results separately from `q_shadow`. Only after this control
works should slower dual updates or a horizon-aware state be evaluated. Hard
per-episode energy enforcement would require a separate execution constraint.

#### Finite-horizon and fresh fixed-price controls

`llm_shadow` now defaults to `--horizon-state 1`. The batched observation supplies
the exact number of remaining arrivals, including the current one. Tabular states
distinguish all remaining counts, so an undiscounted early-decision estimate no
longer shares a table entry with the final-decision estimate. Same-tick arrivals
still decrement the count, and the final decision includes the fleet drain with
no bootstrap. This removes horizon aliasing, not hidden GPU/model-state aliasing.

The existing three-coordinate state key packs the horizon into its load coordinate
internally; CSV diagnostics decode the load and append `remaining_arrivals` when
enabled. `llm_run` and existing sweep experiments retain the original 60 states.
With 200 arrivals, the shadow experiment has 12,000 possible states: this increases
sample requirements and does not guarantee better performance at a fixed budget.

`--mode fixed` disables dual updates entirely. Both learners start with fresh Q
tables and use `--initial-price` from their first training episode. Compare horizon
off/on at the same fixed price to isolate the state change from adaptive-price
history. Use a new output name to preserve earlier results:

```bash
for horizon in 0 1; do
  ./build/llm_shadow --mode fixed --horizon-state "$horizon" \
    --initial-price 3.870829 --budget-j 120 --episodes 2600 \
    --polish-episodes 0 --eval-streams 5 --seed 42 \
    --output "experiments/shadow/fixed_3p87_horizon_${horizon}_seed_42.json" || exit 1
done
```

These are diagnostic controls, not evidence that the adaptive controller converges.
The default adaptive mode retains block updates and frozen-price polish, now with
the optional finite-horizon state. Reports record `mode`, `horizon_state` and
`tabular_states` so old and new experiments can be distinguished.

These runs validate controller direction and reporting but do not demonstrate
successful constrained learning. The consolidation heuristic uses 106.387 J/arrival
on these streams, so this failure is not evidence that the budgets are infeasible.
The shadow prices are training iterates, not validated optimal marginal prices.
The change to gamma=1 and changing tabular targets introduce additional learning
difficulty compared with the earlier constant-discount experiments.

### Q-learning exploration and coverage experiment

The comparison runner's tabular Q-learning uses 5 difficulty, 3 load and 2
batch-fill bins (30 states total). It selects one of three model sizes; GPU
placement is handled by the fixed, batch-aware `least_loaded` scheduler. This
reduces its table from 1,080 to 90 state/action pairs.

For the batched engine, `--q-transition-model PATH` exports an empirical
transition model from Q-learning's training trajectories. The CSV contains
all 30 source states, 3 model-size actions, and 31 possible outcomes (the 30
states plus terminal), including zero-count outcomes. `observations` is the
sample count for the source state/action and `probability` is the observed
frequency. A row with zero observations is unknown, not a measured zero
transition probability. This is an estimate from the chosen training policy;
increase `--episodes` and exploration to improve coverage.

To solve that empirical model with value iteration and export its estimated
`Q*`, `V*`, and greedy policy `pi*`, run:

```bash
python3 tools/value_iteration.py transitions.csv --gamma 0.95 \
  --output optimal_policy.csv
```

The output labels these as model estimates. It computes the greedy policy only
over actions with observed samples; unobserved state/action pairs have blank
Q-values, and states without observed actions are reported as uncovered. Thus
`pi*` is optimal for the sampled model and supported actions, not a guarantee
of the true simulator's global optimum until coverage is adequate.

To evaluate the exported policy in the simulator on the same final prompt
stream as the trained learners, pass its CSV back to `llm_run` with
`--q-policy optimal_policy.csv`. It appears in the normal allocator comparison
as `value_iteration`; unseen states fall back to `SMALL/least_loaded`, and the
number of such fallbacks is reported. Match engine, objective, fleet, and
arrival parameters to the ones used to estimate the transition model.

Use `--objective energy_q93` for the constrained objective: minimize energy
subject to fleet-average quality of at least 0.93. During training, each
learner uses a projected Lagrange multiplier that rises when an episode misses
the quality target and falls when the target is exceeded. Evaluation ranks
feasible policies by system energy and reports quality separately. This
controls the reported mean-quality constraint rather than requiring every
individual prompt to meet 0.93.

The transition export also records energy and completed-prompt quality totals,
so a constrained value-iteration model can be recomputed for a selected
Lagrange multiplier without mixing the learners' changing training rewards:

```bash
python3 tools/value_iteration.py build-cpu/transitions_energy_q93.csv \
  --gamma 0.95 --quality-floor 0.93 --quality-lambda 0 \
  --output optimal_policy_energy_q93.csv
```

DQN checkpoints use `--dqn-save model.pt`; this writes the LibTorch weights
and a matching `model.pt.meta` manifest containing the engine, input-feature
schema, and ordered action list. `--dqn-load model.pt` checks that manifest
before loading. `--dqn-trace decisions.csv` writes each final evaluation
decision with all 18 input features, the chosen model/GPU rule, and all action
Q-values, so the learned policy can be inspected after the process exits.
`--epsilon-trace PATH` and `--q-action-trace PATH` provide the corresponding
decision logs for the bandit and tabular learner. The three logs share prompt
IDs and include each learner's chosen model; the DQN log additionally includes
its GPU rule and all 18 Q-values.

The arrival process defaults to an exponential interarrival time with a mean of
1 second. `--mean-interarrival S` sets this mean for both training and final
evaluation. Smaller values create denser traffic and make high batch-fill states
more likely in the batched engine; `--n-prompts` changes episode length, not
arrival density. For example:

```bash
./build/llm_run --engine batched --episodes 1000 --objective energy \
  --mean-interarrival 0.2 --q-epsilon 0.3 --q-epsilon-decay 0.99999 \
  --q-epsilon-min 0.03 --q-visits experiments/q_visits_dense.csv
```

`--q-alpha X` sets the tabular learning rate (default 0.1; valid range `(0,1]`).
`--q-reward-stats PATH` exports reward, target and TD-error statistics grouped by
state/action, terminal flag and nonoverlapping 5,000-update windows. It includes
counts, sample standard deviations, reward extrema and mean absolute TD error.
One-sample groups report standard deviation zero; this does not establish low
variance. Counts and windows must be considered when comparing groups. High
within-group dispersion can reflect random arrivals, hidden state, or changing
policies; it alone does not prove state aliasing. Raw TD traces retain every update
for narrower-window or conditional analyses.

To isolate training duration from learning rate, compare three settings with the
same exploration schedule: baseline (150 episodes, alpha 0.1), longer (600, 0.1),
and smaller steps (150, 0.02). Do not change both at once. Start with seed 42;
repeat the loop for seeds 43 through 46 before drawing conclusions:

```bash
cmake --build build -j2 && ctest --test-dir build --output-on-failure
mkdir -p experiments/q_stability
set -o pipefail
for setting in baseline longer smaller_alpha; do
  episodes=150
  alpha=0.1
  if [ "$setting" = longer ]; then episodes=600; fi
  if [ "$setting" = smaller_alpha ]; then alpha=0.02; fi
  prefix="experiments/q_stability/${setting}_42"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/llm_run \
    --engine batched --objective energy --episodes "$episodes" \
    --n-prompts 200 --seed 42 --q-alpha "$alpha" \
    --q-epsilon 0.2 --q-epsilon-decay 0.998 --q-epsilon-min 0.01 \
    --q-reward-stats "${prefix}_rewards.csv" --q-td-trace "${prefix}_td.csv" \
    --q-visits "${prefix}_visits.csv" --json "${prefix}.json" --debug \
    | tee "${prefix}.log" || exit 1
done
```

The runner still trains all three learners; longer runs therefore also increase
DQN runtime. Only the Q-learning alpha changes in the smaller-step experiment.

`--q-td-trace PATH` streams every tabular TD update to CSV without retaining
the trace in memory. Rows contain a global update index, state/action, reward,
successor maximum Q (`bootstrap`), discount, target, TD error, Q before/after,
learning rate, exploration and terminal flag. The epsilon column is its value
at the update, not necessarily the exploration used to choose the earlier action.
Terminal targets ignore the bootstrap. Evaluation produces no TD rows. When
inspecting convergence, compare TD-error distributions over successive update
windows rather than expecting individual stochastic errors to decrease monotonically.

All tabular update paths share the same TD calculation. Regression tests check
a hand-calculated bootstrap target, a terminal target with no bootstrap, frozen
evaluation, and learning the cheapest GPU in a deterministic one-step environment
with known tier costs 1, 5 and 9. Passing these tests validates basic arithmetic
and action selection, not convergence in the full partially observed simulator.

Build and run tests locally, then collect a diagnostic trace:

```bash
cmake --build build -j2 && ctest --test-dir build --output-on-failure
mkdir -p experiments/q_td
set -o pipefail
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/llm_run \
  --engine batched --objective energy --episodes 150 --n-prompts 200 \
  --seed 42 --q-epsilon 0.2 --q-epsilon-decay 0.998 --q-epsilon-min 0.01 \
  --q-td-trace experiments/q_td/seed_42_td.csv \
  --q-visits experiments/q_td/seed_42_visits.csv \
  --json experiments/q_td/seed_42.json --debug | tee experiments/q_td/seed_42.log
```

`llm_run --q-visits PATH` exports all tabular state/action combinations, including
unvisited ones, with training selection counts (`visits`), TD update counts
(`updates`) and learned values. Frozen evaluation does not increment these counts.
The 60 states and 18 actions produce 1,080 CSV data rows after initialization.
Coverage of all theoretical states is not expected on every workload; inspect
coverage within reached states as well as the overall fraction.

Use `--q-epsilon`, `--q-epsilon-decay` and `--q-epsilon-min` to control only the
tabular learner. Decay is applied per update, not per episode. For 150 episodes
of 200 arrivals, decay 0.99995 takes initial epsilon 0.2 to about 0.045, versus
the original decay 0.998 which reaches the 0.01 floor after about 1,497 updates.
Keep initial epsilon and the floor fixed to isolate the effect of slower decay.

Build, test, then compare paired seeds at the same workload:

```bash
cmake --build build -j2
ctest --test-dir build --output-on-failure
mkdir -p experiments/q_exploration
set -o pipefail
for seed in 42 43 44 45 46; do
  for schedule in baseline slow; do
    decay=0.998
    if [ "$schedule" = slow ]; then decay=0.99995; fi
    prefix="experiments/q_exploration/${schedule}_${seed}"
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/llm_run \
      --engine batched --objective energy --episodes 150 --n-prompts 200 \
      --seed "$seed" --q-epsilon 0.2 --q-epsilon-min 0.01 \
      --q-epsilon-decay "$decay" --q-visits "${prefix}_visits.csv" \
      --json "${prefix}.json" --debug | tee "${prefix}.log" || exit 1
  done
done
```

All allocators are evaluated, but only Q-learning's exploration settings change.
Compare Q-learning reward, energy, quality and latency against `consolidate_small`
within each seed, then compare baseline versus slow across the five paired seeds.
The exploration-only experiment originally used the old state representation.
After the batched-state fix, rerun into a new output directory: these results
are not directly the same experiment as the archived exploration-only runs.

For batched Q-learning, `load_bin` now uses the fraction of GPUs with active or
waiting sequences, with upward rounding so even one occupied GPU is visible.
Any waiting sequence selects the highest load bin. `batch_bin` uses the maximum
per-GPU active/capacity ratio rather than a fleet-wide average. With the default
three batch bins: 0 means empty, 1 means nonempty up to half full, and 2 means
more than half full. These signals come from the live observation, not stale
`busy_until` values. Static-engine state construction is unchanged. The table
still has 60 states; it does not yet encode model residency or sleep state.

`consolidate_small` is a fixed control: it always uses the smallest model on
the lowest-ID GPU of the smallest compatible tier, regardless of load or sleep
state. It queues work instead of spreading it. If the fleet has no small GPU,
it uses the smallest compatible tier available.

The arrival sweeps compare this control and `always_small` against all three
learners at nine rates from 0.5 to 50 requests/s, retraining at each rate.
Evaluation is frozen and uses a shared held-out stream per rate. The JSON includes
energy, mean/p95 latency, quality, reward and dropped jobs in `measurements`:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ./build/llm_sweep --sweep arrival_energy --outdir sweeps
```

This sweep uses seed 42 only; it is not evidence of statistical significance.

Add `--debug` to `llm_run` to print evaluation reward components, quality-floor
violations, GPU/model placement counts, retained exploration, and energy attributed
to jobs versus residual system energy. Evaluation disables learner exploration
and static-engine bootstrap updates. Profile comparisons rescore the same cached
results without rerunning policies. Batched debug output also reports integrated
active, idle (including waking power), sleep and one-off wake energy and their
accounting error.

The batched engine trains on consecutive arrival decisions (rounded to simulator
ticks), not sequence completions. The successor observation includes the next
prompt and the current fleet. Each transition charges all fleet energy since the
previous decision, plus latency and quality costs of jobs completing in that
interval. The final transition includes draining the fleet and has no bootstrap.
Same-tick arrivals have zero elapsed time. Energy before the first arrival is
charged to the first transition; this initial idle cost is policy-independent.
Time-based discounting uses decision-to-decision elapsed time, or the drain time
for the terminal transition. Constant discounting remains the default.

For reporting and cached profile scoring, residual fleet energy (including wake
costs and gaps with no active jobs) is distributed equally over completed jobs.
This reporting allocation is not used for temporal credit assignment. Undiscounted
transition rewards and reported job rewards therefore sum to the same full-system
objective when all prompts complete. The static engine retains its legacy reward.
The compact fleet observation is still an approximation, not a full Markov state.

```bash
./llm_run                                  # comparison (static engine)
./llm_run --engine batched                 # continuous-batching engine
./llm_run --episodes 80 --objective quality
./llm_run --n-prompts 200 --json out.json  # export metrics to JSON

./llm_benchmark --list                     # list benchmark configs
./llm_benchmark --config medium_energy --outdir benchmarks/
./llm_sweep --list                         # list parameter sweeps
./llm_sweep --sweep arrival_energy --outdir sweeps/

./llm_tests                                # run the test suite
ctest                                      # same, via CTest
```

