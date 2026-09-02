# Benchmarks

Stdlib-only Python harness for reproducible Adaptive AI Inference Runtime evaluation.

## Prerequisites

- Built `runtime-server` (llama-enabled for real presets):

```bash
cmake -S . -B build-llama -DAIRUNTIME_ENABLE_LLAMA=ON
cmake --build build-llama
```

- External GGUF (not bundled), for example:

`$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf`

Expected SHA-256:

`57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`

Formal llama presets (`real-batching-baseline`, `real-batching-adaptive`, `overload`) **abort** if the GGUF file bytes do not match this checksum (no warning-and-continue).

Measured batch metrics (`mean_batch_size`, `multi_request_batch_rate`, …) are isolated via `/metrics` deltas after cold+warmup minus after the measured campaign. `max_batch_size_observed` is process-lifetime only and is not primary measured evidence.

## Commands

```bash
python3 benchmarks/run.py --help
python3 benchmarks/analyze.py --help

# Functional smoke (not final evidence)
python3 benchmarks/run.py \
  --preset real-batching-baseline \
  --server-bin ./build-llama/runtime-server \
  --model-path "$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf" \
  --smoke \
  --output-dir /tmp/airuntime-bench-out

python3 benchmarks/analyze.py /tmp/airuntime-bench-out
```

## Presets

- `real-batching-baseline` — `max_batch_size=1`, wait 0
- `real-batching-adaptive` — `max_batch_size=4`, wait 3 ms
- `overload` — tight queues + high concurrency (records HTTP 503)

## Final evidence policy

Use `--require-clean` only on a clean committed SHA.

Do not commit raw dumps under `results/`. Temporary outputs go under `/tmp` or another ignored path.

See [docs/benchmarks.md](../docs/benchmarks.md).
