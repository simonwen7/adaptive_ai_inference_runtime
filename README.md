# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally.

## Current Status

**Milestone 4 — Reliability, Backpressure & Thin HTTP Serving complete**

Implemented:

- Request lifecycle extensions: `Cancelled` and `TimedOut` terminal states
- First-terminal-wins semantics with cooperative cancellation and steady-clock deadlines
- Generic request observers (no HTTP types in core)
- Deadline-aware Worker handoff (`wait_push_until`) and lazy dead-entry pruning
- Batch filtering of cancelled/timed-out requests before backend execution
- Thin Boost.Beast/Asio HTTP serving library (`airuntime_serving`)
- Non-streaming `POST /v1/infer` and lifecycle NDJSON `POST /v1/infer/stream`
- Endpoints: `GET /health`, `GET /v1/models`, `GET /v1/runtime`, `GET /metrics`
- Client disconnect maps to request cancellation
- Bounded overload maps to HTTP 503 (`QueueFull`)

Honest M4 limits:

- Running backend work is **not** hard-preempted in M4
- Streaming is **lifecycle/status** streaming, **not** token streaming
- Real token streaming waits for a real backend milestone

Not implemented yet: llama.cpp, GGUF, real models, KV cache, continuous batching, final benchmarks, TLS/auth, Prometheus.

## Dependencies

- C++20
- Boost (system) — system-installed (`find_package(Boost 1.83 REQUIRED COMPONENTS system)`)
- nlohmann/json v3.11.3 — pinned via CMake FetchContent
- GoogleTest — FetchContent
- No spdlog in M4 (minimal `std::cerr` diagnostics only)

## Build

```bash
cmake -S . -B build
cmake --build build
```

On macOS with Homebrew Boost:

```bash
cmake -S . -B build -DBoost_ROOT=$(brew --prefix boost)
cmake --build build
```

## Run

```bash
./build/runtime-server --host 127.0.0.1 --port 8080
```

Example:

```bash
curl -s http://127.0.0.1:8080/health
curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"model-a","prompt":"hello","timeout_ms":30000}'
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Backpressure

- Global scheduler capacity full → admission `QueueFull` → HTTP 503
- Worker lane saturation → deadline-aware bounded wait (`wait_push_until`)
- No global max-in-flight or per-model admission in M4

## Roadmap

- Synthetic execution
- FIFO scheduling
- Multi-worker routing
- Memory-constrained model residency
- Dynamic batching
- Adaptive routing
- Adaptive eviction
- Adaptive scheduling
- Backpressure and failures
- HTTP serving
- llama.cpp backend
- Benchmarks

## License

MIT License. See [LICENSE](LICENSE).
