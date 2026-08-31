# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally.

## Current Status

**Milestone 1 — Runtime Core & Synthetic Execution complete**

Implemented:

- Explicit `InferenceRequest` lifecycle with invalid-transition protection
- Bounded thread-safe FIFO queue and `FifoScheduler`
- Admission controller with queue-full / stopped rejection
- Single-worker asynchronous dispatch via a Runtime-owned `std::jthread`
- Generic `IModelBackend` with deterministic `SyntheticModelBackend`
- Synthetic model load/unload and inference (no real neural models)

Not implemented yet: multi-worker routing, memory residency, dynamic batching, adaptive policies, HTTP, streaming, or llama.cpp.

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
