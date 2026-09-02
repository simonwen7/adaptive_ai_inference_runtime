# Adaptive AI Inference Runtime

A C++20 multi-model inference **runtime** for production-oriented serving concerns: request scheduling, dynamic batching, multi-worker routing, memory-constrained model residency, overload/backpressure, cooperative cancellation, and a thin HTTP API — with an optional real llama.cpp GGUF backend on CPU/Metal.

Built to study and demonstrate **control-plane** behavior of an inference server, not to ship a new model architecture.

## What works today

- Multi-worker runtime with bounded queues and admission
- Dynamic static batching (`BatchBuilder`)
- Adaptive policies: workload-aware scheduling, residency-aware routing, cost-aware eviction
- Reliability: deadlines, cancellation, first-terminal-wins
- HTTP: `POST /v1/infer`, lifecycle NDJSON stream, health/models/runtime/metrics
- Optional **library-level** llama.cpp GGUF path (greedy raw-prompt generation, real multi-sequence `infer_batch`)
- Apple Metal auto-offload when supported; explicit CPU via `--gpu-layers 0`
- Reproducible **benchmark harness** (methodology + tooling; final numbers from a clean SHA)

Docs: [Architecture](docs/architecture.md) · [Benchmarks](docs/benchmarks.md)

## Build

Default (synthetic only; does **not** fetch llama.cpp):

```bash
cmake -S . -B build
cmake --build build
```

Optional real backend:

```bash
cmake -S . -B build-llama -DAIRUNTIME_ENABLE_LLAMA=ON
cmake --build build-llama
```

macOS + Homebrew Boost:

```bash
cmake -S . -B build -DBoost_ROOT=$(brew --prefix boost)
```

## Run

Synthetic (default):

```bash
./build/runtime-server --host 127.0.0.1 --port 8080
```

Real GGUF (llama-enabled build):

```bash
./build-llama/runtime-server \
  --backend llama --workers 1 \
  --model-id qwen-small \
  --model-path /path/to/model.gguf \
  --ctx-size 2048
```

Useful policy knobs (defaults preserve prior behavior):

`--max-batch-size` (default 4) · `--max-batch-wait-ms` (default 0) · `--scheduler fifo|workload` (default workload) · `--scheduler-capacity` (32) · `--worker-queue-capacity` (8)

## Test

```bash
ctest --test-dir build --output-on-failure
python3 -m unittest discover -s benchmarks/tests -p 'test_*.py'
```

## Benchmarks

See [benchmarks/README.md](benchmarks/README.md). Harness is stdlib Python only. Final portfolio results require a clean committed SHA and `--require-clean`. This README does **not** claim specific speedups yet.

## Optional model (external)

- `ggml-org/Qwen3.5-0.8B-GGUF` / `Qwen3.5-0.8B-Q4_0.gguf`
- Example path: `$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf`
- Not bundled; `*.gguf` is gitignored

## Dependencies

- C++20, Boost.system, nlohmann/json (FetchContent), GoogleTest (FetchContent)
- Optional llama.cpp @ `9cc33944f9b7a44243618d5522adae357d7fdc90` when `-DAIRUNTIME_ENABLE_LLAMA=ON`

## Honest limits

- No token streaming (stream endpoint is lifecycle status only)
- No continuous batching / prefix cache / speculative decoding
- No Metal hard preemption
- No benchmark performance claims in CI

## License

MIT License. See [LICENSE](LICENSE).
