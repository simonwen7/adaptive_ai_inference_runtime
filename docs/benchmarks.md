# Benchmark methodology

This document describes how Adaptive AI Inference Runtime is evaluated. **Phase 2 ships the harness and methodology.** Final portfolio numbers must be generated later from a **clean committed SHA** with `--require-clean`.

## What is measured

### HTTP end-to-end latency

Time from immediately before opening/sending `POST /v1/infer` until the complete terminal response body is received.

Includes: connection setup, HTTP, queueing, batching, inference, response serialization.

### Successful request throughput (RPS)

`HTTP 200` completions / measured wall-clock (warmup excluded).

### Successful output-token throughput

`sum(output_tokens for HTTP 200)` / measured wall-clock.

Labeled **end-to-end serving** output-token throughput — **not** raw llama decode TPS.

### Not measured / not claimed

- TTFT
- TPOT
- Semantic answer quality
- CI latency/throughput thresholds

## Environment stamps

Every run records:

- `project_git_sha`, `git_dirty`
- llama.cpp pin: `9cc33944f9b7a44243618d5522adae357d7fdc90`
- model basename, size, SHA-256
- OS / arch / SoC / memory / Python (no username/home/serial)

Expected local model (external): `Qwen3.5-0.8B-Q4_0.gguf`  
SHA-256: `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf`

Results are **hardware-specific** (e.g. Apple Silicon Metal). They are not universal llama.cpp speed claims.

## Warmup and runs

For real batching evidence (final campaign):

1. Spawn server (`--port 0`), parse Listening port, poll `/health`
2. 1 cold request (setup failure aborts)
3. 5 warmup requests (discarded; failure aborts)
4. 60 measured requests
5. Restart server for every independent run

Final matrix:

| Batch mode | max_batch_size | max_batch_wait_ms | Concurrency | Runs |
|------------|----------------|-------------------|-------------|------|
| baseline | 1 | 0 | 1, 2, 4 | 3 each |
| adaptive | 4 | 3 | 1, 2, 4 | 3 each |

Fixed: llama backend, workers=1, ctx=2048, `max_output_tokens=32`, greedy M5 sampler, mixed 16-prompt workload.

Across 3 runs: report **median** of run-level RPS, tokens/s, p50, p95, mean batch size, and multi-request batch rate.

Percentiles: **nearest-rank** p50 and p95.

## Batch metrics

Runtime `/metrics` batch counters are **cumulative** for the process lifetime.

The harness isolates the **measured window** by snapshotting:

1. `metrics_before` — immediately after cold + warmup
2. `metrics_after` — immediately after the measured campaign

Measured batch evidence uses counter deltas (`after − before`):

- `mean_batch_size = Δrequests_executed_via_batches / Δbatches_executed`
- `multi_request_batch_rate = Δmulti_request_batches / Δbatches_executed`  
  (fraction of **measured batches** that contained more than one request)
- `multi_request_batches` / `batches_executed` / `requests_executed_via_batches` in `run_summary` are these measured-window deltas

If a cumulative counter decreases between snapshots, the run fails (`BenchmarkError`).

`max_batch_size_observed` is a **process-lifetime max gauge** (includes cold/warmup). It is retained only inside raw `metrics_before` / `metrics_after` snapshots and is **not** used as measured-window primary evidence.

## Overload evaluation

Frozen overload preset (final campaign):

- `max_batch_size=1`, wait 0
- `scheduler_capacity=4`, `worker_queue_capacity=2`
- concurrency 32, 64 measured requests
- 3 runs

Record HTTP 200, HTTP 503, 503 rate, successful RPS/p50/p95.  
503 is expected policy behavior — not a harness failure.

## Operational caveats

- Restart the server for every independent run (avoids cross-run counter carry)
- Within a run, measured batch metrics use after−before deltas so cold/warmup do not contaminate evidence
- Mismatch vs the frozen GGUF SHA-256 aborts the run before cold/warmup/measurement
- Thermal/background load on Apple Silicon can add variance
- Keep machine plugged in; reduce competing workloads; use short controlled runs

## Final evidence tables (reserved)

_Populate only after clean-SHA campaign:_

### A. Real GGUF serving

| Concurrency | Batch mode | RPS | tokens/s | p50 ms | p95 ms | mean batch |
|-------------|------------|-----|----------|--------|--------|------------|
| … | … | … | … | … | … | … |

### D. Overload

| Metric | Value |
|--------|-------|
| … | … |
