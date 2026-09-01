# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally.

## Current Status

**Milestone 2 — Multi-Worker Routing & Memory-Constrained Model Residency complete**

Implemented:

- Concurrent multi-worker execution (per-worker bounded lane + thread)
- Global FIFO scheduling with RoundRobin and LeastLoaded routers
- Immutable `ModelRegistry` and extended `ModelSpec` memory footprints
- Per-worker `ModelManager` + `ResourceManager` with memory budgets
- Deterministic LRU eviction and residency hit/miss/reload metrics
- Hardened admission against duplicate `RequestPtr` enqueue

Not implemented yet: dynamic batching, adaptive routing/eviction/scheduling, HTTP, streaming, llama.cpp, or benchmarks.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Run

```bash
./build/runtime-server
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

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
