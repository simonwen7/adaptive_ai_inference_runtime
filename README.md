# Adaptive AI Inference Runtime

A modern C++ multi-model inference runtime focused on production-oriented concerns: request scheduling, dynamic batching, worker routing, model residency, memory pressure, overload handling, failure handling, and performance evaluation.

This repository is under active development. The runtime architecture is being built incrementally.

## Current Status

**Milestone 5 — Real llama.cpp GGUF backend (optional)**

Implemented:

- Optional library-level llama.cpp integration via CMake FetchContent
- Pinned upstream: [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) @ `9cc33944f9b7a44243618d5522adae357d7fdc90`
- Separate `airuntime_llama_backend` (core/serving remain llama-free)
- Process-scoped `LlamaBackendRuntime` RAII (`ggml_backend_load_all` / `llama_backend_init` / `llama_backend_free`)
- PIMPL `LlamaCppBackend` implementing `IModelBackend` with real GGUF load/unload
- Raw-prompt greedy text generation (no chat templates)
- Real static multi-sequence `infer_batch` (not serial `infer` loops)
- Context reuse with mandatory KV/memory clear between batches
- Apple Metal auto offload when `llama_supports_gpu_offload()`; explicit CPU via `--gpu-layers 0`
- `ErrorCode::ContextLengthExceeded` → HTTP 400 (no silent truncation)
- M4 cooperative cancellation unchanged (no Metal hard preemption / abort-callback cancel)

Default builds keep the synthetic backend and do **not** fetch llama.cpp.

Honest M5 limits:

- No hard Metal preemption / abort-callback cancellation
- No token streaming (`/v1/infer/stream` remains lifecycle NDJSON only)
- No continuous batching / prefix cache / speculative decoding
- No benchmark claims yet

## Dependencies

- C++20
- Boost (system) — system-installed (`find_package(Boost 1.83 REQUIRED COMPONENTS system)`)
- nlohmann/json v3.11.3 — pinned via CMake FetchContent
- GoogleTest — FetchContent
- Optional: llama.cpp @ `9cc33944f9b7a44243618d5522adae357d7fdc90` when `-DAIRUNTIME_ENABLE_LLAMA=ON`

## Build

Default (synthetic only, fast; does not fetch llama.cpp):

```bash
cmake -S . -B build
cmake --build build
```

Optional real llama.cpp backend:

```bash
cmake -S . -B build-llama -DAIRUNTIME_ENABLE_LLAMA=ON
cmake --build build-llama
```

On Apple Silicon, Metal is enabled for the pinned ggml build (`GGML_METAL=ON`). On non-Apple CI, Metal is off.

On macOS with Homebrew Boost:

```bash
cmake -S . -B build -DBoost_ROOT=$(brew --prefix boost)
cmake --build build
```

## Run

Synthetic (default):

```bash
./build/runtime-server --host 127.0.0.1 --port 8080
```

Real GGUF backend (requires llama-enabled build):

```bash
./build-llama/runtime-server \
  --backend llama \
  --workers 1 \
  --model-id qwen-small \
  --model-path /path/to/model.gguf \
  --ctx-size 2048
```

`--ctx-size` is **context tokens per sequence**, not total llama `n_ctx`.

CPU fallback: `--gpu-layers 0`.

Optional local example model (not bundled; download yourself if desired):

- Hugging Face: `ggml-org/Qwen3.5-0.8B-GGUF`
- File: `Qwen3.5-0.8B-Q4_0.gguf`

Example path used in local acceptance:

`$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf`

```bash
curl -s http://127.0.0.1:8080/health
curl -s -X POST http://127.0.0.1:8080/v1/infer \
  -H 'Content-Type: application/json' \
  -d '{"model_id":"qwen-small","prompt":"hello","timeout_ms":30000}'
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

Real-model llama tests (skipped unless set):

```bash
export AIRUNTIME_TEST_GGUF="$HOME/Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf"
ctest --test-dir build-llama --output-on-failure
```

## Backpressure

- Global scheduler capacity full → admission `QueueFull` → HTTP 503
- Worker lane saturation → deadline-aware bounded wait (`wait_push_until`)
- Context length exceeded → HTTP 400

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
