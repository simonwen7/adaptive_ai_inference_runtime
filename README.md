# Adaptive AI Inference Runtime

A C++20 inference-serving runtime that demonstrates production control-plane concerns — request scheduling, dynamic batching, multi-worker routing, bounded model residency, overload backpressure, cooperative cancellation, and HTTP serving — with an optional real llama.cpp GGUF backend on Apple Metal.

## What it demonstrates

- **Dynamic static batching** — `BatchBuilder` groups concurrent requests into same-model batches dispatched through llama.cpp `infer_batch`
- **Multi-worker routing** with pluggable policies (round-robin, least-loaded, residency-aware)
- **Workload-aware scheduling** and cost-aware eviction under memory constraints
- **Bounded queues and backpressure** — overload is rejected with HTTP 503, not unbounded queueing
- **Cooperative cancellation and deadlines** — first-terminal-wins request lifecycle
- **HTTP API** — `POST /v1/infer`, lifecycle NDJSON stream, health/models/runtime/metrics
- **Reproducible benchmark harness** — stdlib Python, `--require-clean`, frozen model/hardware identity

## Measured result

On an Apple M4 Pro (24 GB, Metal) serving Qwen3.5-0.8B Q4_0 via llama.cpp:

> At concurrency 4, adaptive batching increased median HTTP serving throughput from 2.17 to 4.35 req/s (+101%) while reducing median p95 end-to-end latency from 2849.9 ms to 2105.4 ms (−26.1%).

At concurrency 4, median end-to-end serving output-token throughput increased from 67.84 to 136.35 tokens/s (+101%). The real llama.cpp backend formed true 4-request batches on every measured batch execution.

Under the frozen c=32 overload scenario (capacity deliberately limited), bounded queues rejected 61/64 requests with HTTP 503 in each of three runs, with zero transport errors. No global HTTP freeze or timeout collapse occurred.

Full methodology, per-run data, caveats, and the c=2 negative result are in [docs/benchmarks.md](docs/benchmarks.md). Machine-readable results: [`benchmarks/results/m6_final_summary.json`](benchmarks/results/m6_final_summary.json).

## Architecture

See [docs/architecture.md](docs/architecture.md).

## Build

Default (synthetic backend only):

```bash
cmake -S . -B build
cmake --build build
```

With llama.cpp backend:

```bash
cmake -S . -B build-llama -DAIRUNTIME_ENABLE_LLAMA=ON
cmake --build build-llama
```

macOS + Homebrew Boost:

```bash
cmake -S . -B build -DBoost_ROOT=$(brew --prefix boost)
```

## Run

Synthetic:

```bash
./build/runtime-server --host 127.0.0.1 --port 8080
```

Real GGUF:

```bash
./build-llama/runtime-server \
  --backend llama --workers 1 \
  --model-id qwen-small \
  --model-path /path/to/model.gguf \
  --ctx-size 2048
```

Policy knobs: `--max-batch-size` (4) · `--max-batch-wait-ms` (0) · `--scheduler fifo|workload` (workload) · `--scheduler-capacity` (32) · `--worker-queue-capacity` (8)

## Test

```bash
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'
```

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md) for harness usage and [docs/benchmarks.md](docs/benchmarks.md) for results.

## Optional model (external)

- `ggml-org/Qwen3.5-0.8B-GGUF` / `Qwen3.5-0.8B-Q4_0.gguf`
- Not bundled; `*.gguf` is gitignored

## Dependencies

- C++20, Boost.system, nlohmann/json (FetchContent), GoogleTest (FetchContent)
- Optional llama.cpp @ `9cc33944f9b7a44243618d5522adae357d7fdc90` when `-DAIRUNTIME_ENABLE_LLAMA=ON`

## Scope and limitations

- No continuous batching, KV cache reuse, or speculative decoding
- No token streaming (stream endpoint is lifecycle status only)
- No Metal hard preemption
- Benchmark results are hardware-specific (Apple M4 Pro / Metal); not generalizable to other platforms

## License

MIT License. See [LICENSE](LICENSE).
