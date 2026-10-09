# Dispensator

A C++17 reinforcement-learning simulator that allocates incoming LLM prompts
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

## Dependencies

- A C++17 compiler (tested with g++ 9.4) and CMake ≥ 3.18.
- **LibTorch** 2.4.1 (CPU). The build prefers a CPU-only LibTorch at
  `$HOME/.local/lib/libtorch`; otherwise it falls back to the `torch` pip
  package. Override explicitly with `-DTorch_DIR=/path/to/share/cmake/Torch`.
  The pip wheels use the pre-C++11 string ABI, so the project is compiled with
  `-D_GLIBCXX_USE_CXX11_ABI=0` (carried automatically via `TORCH_CXX_FLAGS`).
- **nlohmann/json** (header-only; the system install at `/usr/include` is used
  for the benchmark/sweep JSON export).

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j4
```

## Run

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

