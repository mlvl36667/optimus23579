#!/usr/bin/env python3
"""Measure real LLM decode throughput and energy-per-token via Ollama.

Complements tools/measure_gpu.py (raw GPU power/TFLOP/s). This script drives a
set of real quantised LLMs through Ollama's REST API, reads back the server-
reported token counts and durations, and samples board power with nvidia-smi.
From that it derives, per model:

  * decode_tokens_per_s  — sustained generation speed  -> LLM::base_speed
  * energy_per_token_j   — (active power * decode time) / tokens
                                                        -> LLM::energy_per_token

It measures all models on whatever GPU Ollama actually schedules them on (this
host always picks the RTX 3090), and identifies the active card by the GPU with
the highest utilisation during decode — robust to CUDA_VISIBLE_DEVICES pinning
being ignored by the Ollama runner.

Dependency-light: needs only `requests`, a running `ollama serve`, and
`nvidia-smi`. No internet (models already pulled), no pynvml.

Usage:
    python3 tools/measure_ollama.py --models llama3.2:1b llama3.2:3b llama3.1:latest
    python3 tools/measure_ollama.py --models ... --json tools/ollama_measurements.json
"""
import argparse
import json
import subprocess
import threading
import time
from collections import defaultdict

import requests

OLLAMA_URL = "http://localhost:11434"

PROMPT = ("Write a long, detailed technical explanation of how modern GPUs "
          "schedule work across their streaming multiprocessors, including "
          "warps, occupancy, and memory coalescing.")


def wait_for_server(timeout_s=30):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
            return True
        except Exception:
            time.sleep(0.5)
    return False


class GpuSampler(threading.Thread):
    """Poll nvidia-smi; keep per-GPU power and utilisation traces."""

    def __init__(self, interval_s=0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.power = defaultdict(list)   # name -> [W,...]
        self.util = defaultdict(list)    # name -> [%,...]

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi",
                     "--query-gpu=name,power.draw,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True)
                for row in out.strip().splitlines():
                    name, p, u = [x.strip() for x in row.split(",")]
                    self.power[name].append(float(p))
                    self.util[name].append(float(u))
            except Exception:
                pass
            time.sleep(self.interval_s)

    def stop(self):
        self._stop.set()
        self.join()

    def active_card(self):
        """The GPU name with the highest peak utilisation (= where decode ran)."""
        if not self.util:
            return None
        return max(self.util, key=lambda n: max(self.util[n]))


def generate(model, n_predict):
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "options": {"num_predict": n_predict, "temperature": 0.7, "seed": 42},
    }, timeout=600)
    r.raise_for_status()
    return r.json()


def measure_model(model, n_predict):
    # Warm up: load weights into VRAM (excluded from timing).
    generate(model, 16)

    sampler = GpuSampler()
    sampler.start()
    resp = generate(model, n_predict)
    time.sleep(0.2)
    sampler.stop()

    eval_count = resp.get("eval_count", 0)       # decoded tokens
    eval_dur_ns = resp.get("eval_duration", 0)   # ns spent decoding
    decode_s = eval_dur_ns / 1e9 if eval_dur_ns else 0.0
    tok_s = eval_count / decode_s if decode_s > 0 else 0.0

    card = sampler.active_card()
    peak_w = max(sampler.power[card]) if card else 0.0
    # Attribute decode energy using peak active power over the decode window.
    energy_j = peak_w * decode_s
    e_per_tok = energy_j / eval_count if eval_count else 0.0

    # idle = lowest observed power on the active card (best-effort).
    idle_w = min(sampler.power[card]) if card else 0.0

    return {
        "model": model,
        "active_card": card,
        "decode_tokens": eval_count,
        "decode_seconds": round(decode_s, 3),
        "decode_tokens_per_s": round(tok_s, 1),
        "active_power_peak_w": round(peak_w, 1),
        "active_power_min_w": round(idle_w, 1),
        "energy_per_token_j": round(e_per_tok, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-predict", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3,
                    help="runs per model; the median tok/s run is reported")
    ap.add_argument("--json")
    args = ap.parse_args()

    if not wait_for_server():
        raise SystemExit("ollama server not reachable at " + OLLAMA_URL)

    results = {}
    for model in args.models:
        print(f"[*] measuring {model} ({args.repeats} runs)...", flush=True)
        runs = [measure_model(model, args.n_predict) for _ in range(args.repeats)]
        # pick the median by tok/s to damp noise
        runs.sort(key=lambda r: r["decode_tokens_per_s"])
        best = runs[len(runs) // 2]
        results[model] = best
        print(f"    {best['active_card']}: {best['decode_tokens_per_s']:.1f} tok/s | "
              f"peak {best['active_power_peak_w']:.0f} W | "
              f"{best['energy_per_token_j']:.4f} J/token", flush=True)

    print("\n" + "=" * 74)
    print("  MEASURED LLM DECODE (Ollama)")
    print("=" * 74)
    print(f"  {'model':<18} {'card':<16} {'tok/s':>8} {'peak W':>8} {'J/token':>9}")
    print("-" * 74)
    for model, r in results.items():
        card = (r["active_card"] or "?").replace("NVIDIA GeForce ", "")
        print(f"  {model:<18} {card:<16} {r['decode_tokens_per_s']:>8.1f} "
              f"{r['active_power_peak_w']:>8.0f} {r['energy_per_token_j']:>9.4f}")
    print("=" * 74)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[INFO] written to {args.json}")


if __name__ == "__main__":
    main()
