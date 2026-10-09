# LLM Allocation Simulator — C++ port

A full C++17 port of the Python `llm_alloc` package, using **LibTorch** for the
DQN allocator and **nlohmann/json** for JSON export.

## Modules

| Header / source                | Python original          |
|--------------------------------|--------------------------|
| `domain.{hpp,cpp}`             | `domain.py`              |
| `simulator.{hpp,cpp}`          | `simulator.py`           |
| `batched_simulator.{hpp,cpp}`  | `batched_simulator.py`   |
| `allocators.{hpp,cpp}`         | `allocators.py`          |
| `metrics.{hpp,cpp}`            | `metrics.py`             |
| `learning.{hpp,cpp}`           | `learning.py` (eps-greedy, tabular Q, DQN, train loop) |
| `apps/run_main.cpp`            | `run.py`                 |
| `apps/benchmark_main.cpp`      | `benchmark.py`           |
| `apps/sweep_main.cpp`          | `sweep.py`               |

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

