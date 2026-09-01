# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally.

## Current Status

**Milestone 3 — Dynamic Batching & Adaptive Runtime Policies complete**

Implemented:

- Worker-local dynamic batching with contiguous same-model head-of-line policy
- Real `IModelBackend::infer_batch` API with ordered per-request results
- Deterministic synthetic batch-cost model (shared prefill amortization)
- `WorkloadAwareScheduler` with largest-group preference and bypass starvation protection
- `ResidencyAwareRouter` with resident / no-eviction / eviction-required tiers
- `CostAwareEvictionPolicy` with exact lexicographic victim ordering
- Baseline policies preserved: `FifoScheduler`, `RoundRobinRouter`, `LeastLoadedRouter`, `LruEvictionPolicy`

Not implemented yet: HTTP, streaming, timeouts/cancellation, llama.cpp, real models, or final benchmarks.

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
