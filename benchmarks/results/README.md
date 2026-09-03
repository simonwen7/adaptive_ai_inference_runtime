# Benchmark results

Only curated summary JSON is tracked here. Raw per-request result files remain outside the repository (under `/tmp` during campaign runs).

## Final evidence

**`m6_final_summary.json`** — curated machine-readable summary of the M6 formal benchmark campaign.

- Source SHA: `dfc00d73033af62f1888bac014d5586086c5b30d`
- 21 formal runs (9 baseline + 9 adaptive + 3 overload), all `--require-clean`
- Model: `Qwen3.5-0.8B-Q4_0.gguf` (SHA-256 `57d1997…`)
- Hardware: Apple M4 Pro, 24 GB, macOS 15.7.4, Metal

Results are hardware-specific. No statistical significance is claimed (3 runs per cell, median aggregation).

## Reproduction

```bash
cmake -S . -B build-llama -DAIRUNTIME_ENABLE_LLAMA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-llama

python3 benchmarks/run.py --preset real-batching-baseline --require-clean \
  --server-bin ./build-llama/runtime-server \
  --model-path "$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf" \
  --output-dir /tmp/airuntime-bench-out

python3 benchmarks/run.py --preset real-batching-adaptive --require-clean \
  --server-bin ./build-llama/runtime-server \
  --model-path "$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf" \
  --output-dir /tmp/airuntime-bench-out

python3 benchmarks/run.py --preset overload --require-clean \
  --server-bin ./build-llama/runtime-server \
  --model-path "$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf" \
  --output-dir /tmp/airuntime-bench-out

python3 benchmarks/analyze.py /tmp/airuntime-bench-out/*.json \
  --summary-json /tmp/airuntime-bench-out/summary.json
```

Do not commit raw per-request dumps, server logs, or smoke outputs here.
