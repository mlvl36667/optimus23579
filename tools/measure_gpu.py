#!/usr/bin/env python3
"""Measure real GPU physical parameters to calibrate the simulator's constants.

Dependency-light by design: needs only a CUDA-capable PyTorch (already present
as torch 2.x +cuXXX) and the `nvidia-smi` binary. No pynvml / transformers /
internet required — the host this was written for has no outbound network.

What it measures, per physical GPU:
  * idle_power_w   — board power with no compute running (nvidia-smi sampling)
  * active_power_w — board power under a sustained GEMM load (a stand-in for
                     the dense matmuls that dominate LLM decode)
  * throughput     — sustained FP16 GEMM TFLOP/s (relative compute capability)

GPUs are identified by NAME (not index), because the CUDA runtime orders
devices by compute capability while nvidia-smi orders by PCI bus — so cuda:0
is not necessarily "GPU 0" in nvidia-smi.

Usage:
    python3 tools/measure_gpu.py                 # human-readable summary
    python3 tools/measure_gpu.py --json out.json # also dump raw numbers
"""
import argparse
import json
import re
import subprocess
import threading
import time

import torch


def smi_power_by_name():
    """Return {gpu_name: power_draw_w} sampled once from nvidia-smi."""
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=name,power.draw",
         "--format=csv,noheader,nounits"],
        text=True,
    )
    result = {}
    for line in out.strip().splitlines():
        name, power = [x.strip() for x in line.split(",")]
        result[name] = float(power)
    return result


class PowerSampler(threading.Thread):
    """Background thread that polls nvidia-smi and keeps per-GPU power samples."""

    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples = {}  # name -> [w, w, ...]

    def run(self):
        while not self._stop.is_set():
            try:
                for name, w in smi_power_by_name().items():
                    self.samples.setdefault(name, []).append(w)
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._stop.set()
        self.join()

    def stats(self, name):
        xs = sorted(self.samples.get(name, []))
        if not xs:
            return {"mean": 0.0, "p50": 0.0, "max": 0.0, "n": 0}
        return {
            "mean": sum(xs) / len(xs),
            "p50": xs[len(xs) // 2],
            "max": xs[-1],
            "n": len(xs),
        }


def gemm_throughput_tflops(device, size=4096, seconds=6.0, dtype=torch.float16):
    """Run back-to-back square GEMMs for `seconds` and return sustained TFLOP/s.

    A square (n x n) @ (n x n) GEMM costs ~2 * n^3 FLOPs. This is the standard
    proxy for the dense linear layers that dominate transformer decode compute.
    """
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    # Warm up (kernel autotune / clocks spin up).
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize(device)

    flops_per_gemm = 2.0 * size ** 3
    iters = 0
    t0 = time.time()
    while time.time() - t0 < seconds:
        c = a @ b
        iters += 1
        if iters % 50 == 0:
            torch.cuda.synchronize(device)
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    return (flops_per_gemm * iters) / elapsed / 1e12


def torch_index_by_name():
    """Map {gpu_name: torch_device_index}."""
    m = {}
    for i in range(torch.cuda.device_count()):
        m[torch.cuda.get_device_properties(i).name] = i
    return m


def measure():
    assert torch.cuda.is_available(), "CUDA not available in torch"
    names = torch_index_by_name()
    props = {n: torch.cuda.get_device_properties(i) for n, i in names.items()}

    # --- idle power: let the cards settle, sample with no compute ---
    print("[1/3] measuring idle power (3 s, no compute)...", flush=True)
    for i in names.values():
        torch.cuda.synchronize(i)
    time.sleep(1.0)
    idle = PowerSampler()
    idle.start()
    time.sleep(3.0)
    idle.stop()

    # --- active power + throughput: load one card at a time ---
    results = {}
    for name, idx in names.items():
        print(f"[2/3] loading {name} (cuda:{idx}) with sustained GEMM...",
              flush=True)
        sampler = PowerSampler()
        sampler.start()
        tflops = gemm_throughput_tflops(f"cuda:{idx}")
        time.sleep(0.2)
        sampler.stop()
        a = sampler.stats(name)
        results[name] = {
            "torch_index": idx,
            "vram_gib": round(props[name].total_memory / 1024 ** 3, 1),
            "sm_count": props[name].multi_processor_count,
            "idle_power_w": round(idle.stats(name)["p50"], 1),
            "active_power_w": round(a["max"], 1),
            "active_power_mean_w": round(a["mean"], 1),
            "gemm_tflops_fp16": round(tflops, 1),
        }

    print("[3/3] done.\n", flush=True)
    return results


def print_summary(results):
    print("=" * 74)
    print("  MEASURED GPU PARAMETERS (this host)")
    print("=" * 74)
    hdr = f"  {'GPU':<24} {'VRAM':>6} {'SM':>4} {'idle W':>7} {'active W':>9} {'TFLOP/s':>8}"
    print(hdr)
    print("-" * 74)
    for name, r in sorted(results.items(), key=lambda kv: kv[1]["vram_gib"]):
        print(f"  {name:<24} {r['vram_gib']:>5.0f}G {r['sm_count']:>4} "
              f"{r['idle_power_w']:>7.1f} {r['active_power_w']:>9.1f} "
              f"{r['gemm_tflops_fp16']:>8.1f}")
    print("=" * 74)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write raw measurements to this path")
    args = ap.parse_args()
    results = measure()
    print_summary(results)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] raw measurements written to {args.json}")


if __name__ == "__main__":
    main()
