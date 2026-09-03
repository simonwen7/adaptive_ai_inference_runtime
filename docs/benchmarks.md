# Benchmarks

Measured performance of the Adaptive AI Inference Runtime, from the M6 formal evidence campaign.

## Source identity

| Field | Value |
|-------|-------|
| Project SHA | `dfc00d73033af62f1888bac014d5586086c5b30d` |
| llama.cpp SHA | `9cc33944f9b7a44243618d5522adae357d7fdc90` |
| Model | `Qwen3.5-0.8B-Q4_0.gguf` |
| Model SHA-256 | `57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf` |
| Hardware | Apple M4 Pro, 12 cores, 24 GB unified, arm64 |
| OS | macOS 15.7.4, Metal |
| `git_dirty` | `false` for all runs |

## Methodology

**Latency**: HTTP end-to-end — from immediately before sending `POST /v1/infer` until the complete terminal response body is received. Includes connection setup, queueing, batching, inference, and response serialization. TTFT and TPOT are not measured.

**Throughput**: Successful HTTP 200 completions / measured wall-clock. Output-token throughput is **end-to-end serving** throughput — not raw llama decode TPS.

**Percentiles**: Nearest-rank p50 and p95 within each run.

**Aggregation**: Median of 3 independent run-level aggregates. Request-level samples are not pooled across runs.

**Per run**: 1 cold request, 5 warmup requests (discarded), 60 measured requests. Server restarted between runs.

**Batch metrics**: Runtime `/metrics` counters are cumulative. The harness uses `metrics_after − metrics_before` deltas for the measured window only:
- `mean_batch_size = Δrequests_executed_via_batches / Δbatches_executed`
- `multi_request_batch_rate = Δmulti_request_batches / Δbatches_executed`

`max_batch_size_observed` is a process-lifetime gauge and is not used as measured-window evidence.

## Batching configuration

| Parameter | Baseline | Adaptive | Equal? |
|-----------|----------|----------|--------|
| backend | llama | llama | ✓ |
| workers | 1 | 1 | ✓ |
| ctx_size | 2048 | 2048 | ✓ |
| scheduler | workload | workload | ✓ |
| scheduler_capacity | 32 | 32 | ✓ |
| worker_queue_capacity | 8 | 8 | ✓ |
| max_output_tokens | 32 | 32 | ✓ |
| **max_batch_size** | **1** | **4** | treatment |
| **max_batch_wait_ms** | **0** | **3** | treatment |

## Median results

| Concurrency | Mode | RPS | Output tok/s | p50 ms | p95 ms | Mean batch | Multi-batch rate |
|-------------|------|-----|-------------|--------|--------|------------|-----------------|
| 1 | baseline | 2.09 | 65.56 | 198.6 | 1332.1 | 1.00 | 0% |
| 1 | adaptive | 2.36 | 73.82 | 353.4 | 967.7 | 1.00 | 0% |
| 2 | baseline | 4.20 | 131.59 | 380.6 | 736.4 | 1.00 | 0% |
| 2 | adaptive | 3.15 | 98.85 | 555.6 | 1181.9 | 2.00 | 100% |
| 4 | baseline | 2.17 | 67.84 | 1741.9 | 2849.9 | 1.00 | 0% |
| 4 | adaptive | 4.35 | 136.35 | 898.8 | 2105.4 | 4.00 | 100% |

## Adaptive vs baseline changes

| Concurrency | ΔRPS | ΔOutput tok/s | Δp50 | Δp95 |
|-------------|------|---------------|------|------|
| 1 | +12.6% | +12.6% | +78.0% | −27.4% |
| 2 | −24.9% | −24.9% | +46.0% | +60.5% |
| 4 | +101.0% | +101.0% | −48.4% | −26.1% |

Formula: `(adaptive_median − baseline_median) / baseline_median × 100`. For latency columns, negative means lower (better) latency.

## Per-run detail

### c=1 baseline

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 0.76 | 23.79 | 898.8 | 3730.2 |
| 1 | 2.09 | 65.56 | 198.6 | 1332.1 |
| 2 | 5.32 | 166.72 | 189.5 | 196.5 |

### c=1 adaptive

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 2.55 | 79.80 | 353.4 | 683.0 |
| 1 | 2.36 | 73.82 | 330.5 | 967.7 |
| 2 | 2.13 | 66.62 | 388.7 | 1030.7 |

### c=2 baseline

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 4.20 | 131.59 | 380.6 | 736.4 |
| 1 | 5.31 | 166.45 | 374.4 | 400.9 |
| 2 | 2.87 | 89.93 | 580.8 | 1360.1 |

### c=2 adaptive

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 3.15 | 98.85 | 555.6 | 1181.9 |
| 1 | 3.44 | 107.67 | 542.8 | 753.7 |
| 2 | 3.06 | 95.87 | 571.0 | 1419.4 |

### c=4 baseline

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 2.30 | 72.18 | 1608.6 | 2610.0 |
| 1 | 2.17 | 67.84 | 1765.4 | 2849.9 |
| 2 | 2.12 | 66.34 | 1741.9 | 2894.3 |

### c=4 adaptive

| Run | RPS | tok/s | p50 ms | p95 ms |
|-----|-----|-------|--------|--------|
| 0 | 4.12 | 129.19 | 907.4 | 2105.4 |
| 1 | 4.35 | 136.35 | 898.8 | 1932.8 |
| 2 | 4.62 | 144.65 | 744.0 | 2569.9 |

## Variance

| Cell | RPS min/med/max | p95 min/med/max ms |
|------|-----------------|-------------------|
| c=1 baseline | 0.76 / 2.09 / 5.32 | 196.5 / 1332.1 / 3730.2 |
| c=1 adaptive | 2.13 / 2.36 / 2.55 | 683.0 / 967.7 / 1030.7 |
| c=2 baseline | 2.87 / 4.20 / 5.31 | 400.9 / 736.4 / 1360.1 |
| c=2 adaptive | 3.06 / 3.15 / 3.44 | 753.7 / 1181.9 / 1419.4 |
| c=4 baseline | 2.12 / 2.17 / 2.30 | 2610.0 / 2849.9 / 2894.3 |
| c=4 adaptive | 4.12 / 4.35 / 4.62 | 1932.8 / 2105.4 / 2569.9 |

## Batch utilization

At c=1, no multi-request batching occurred (mean batch 1.00, 0% multi-request rate) — expected with only one concurrent client.

At c=2, adaptive mode formed real two-request batches on every measured batch execution (mean batch 2.00, 100% multi-request batch rate). However, median RPS decreased (−24.9%) and median p95 increased (+60.5%) vs baseline. Batch formation does not automatically produce an end-to-end throughput win at every concurrency point.

At c=4, adaptive mode formed real four-request batches consistently (mean batch 4.00, 100% multi-request batch rate), yielding +101% median RPS and −26.1% median p95.

## Overload / backpressure

Frozen overload configuration:

| Parameter | Value |
|-----------|-------|
| workers | 1 |
| max_batch_size | 1 |
| max_batch_wait_ms | 0 |
| scheduler_capacity | 4 |
| worker_queue_capacity | 2 |
| concurrency | 32 |
| measured_requests | 64 |
| max_output_tokens | 32 |
| runs | 3 |

### Individual runs

| Run | HTTP 200 | HTTP 503 | 503 rate | Other HTTP | Transport | RPS | p50 ms | p95 ms | Duration |
|-----|----------|----------|----------|------------|-----------|-----|--------|--------|----------|
| 0 | 3 | 61 | 95.3% | 0 | 0 | 5.47 | 365.5 | 548.3 | 0.549 s |
| 1 | 3 | 61 | 95.3% | 0 | 0 | 5.63 | 366.1 | 532.5 | 0.533 s |
| 2 | 3 | 61 | 95.3% | 0 | 0 | 5.46 | 366.5 | 549.3 | 0.550 s |

### Interpretation

HTTP 503 responses are **intentional bounded rejection** under a deliberately oversubscribed scenario (32 concurrent clients, capacity for ~6 in-flight). The bounded queues reject excess requests immediately with HTTP 503 rather than allowing unbounded queueing.

All three runs: 0 transport errors, 0 other HTTP errors, metrics collection completed normally, server remained responsive. No 125-second transport-timeout collapse occurred (durations < 1 s).

Latency p50/p95 apply to successful HTTP 200 responses only.

## Limitations and caveats

- Results are specific to Apple M4 Pro / 24 GB / macOS 15.7.4 / Metal. Do not generalize to Linux, NVIDIA, cloud GPUs, other Macs, or other model sizes.
- Only 3 independent runs per cell. Medians are descriptive; no formal statistical significance is claimed.
- Baseline c=1 had exceptionally high run-to-run variance (RPS range 0.76–5.32). c=1 percentage claims are fragile and not used as headlines.
- At c=2, adaptive batching formed real batches but median throughput decreased vs baseline. This is documented honestly as an engineering nuance.
- The historical c=2 direction from an earlier campaign did not reproduce in this final campaign.
- Latency is HTTP end-to-end. TTFT and TPOT are not measured.
- Output-token throughput is end-to-end serving throughput, not raw llama decode TPS.
- No continuous batching, KV cache reuse, speculative decoding, or token streaming.

## Reproduction

See [benchmarks/README.md](../benchmarks/README.md) for harness usage. Machine-readable results: [`benchmarks/results/m6_final_summary.json`](../benchmarks/results/m6_final_summary.json).

All formal runs used `--require-clean` to enforce a clean git tree and record the project SHA in run metadata.
