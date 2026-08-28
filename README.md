# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally; the current codebase establishes the project foundation and does not yet provide inference functionality.

## Current Status

Milestone 0 — Repository Foundation complete

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
