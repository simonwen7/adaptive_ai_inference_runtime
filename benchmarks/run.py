#!/usr/bin/env python3
"""Benchmark runner for Adaptive AI Inference Runtime (stdlib only)."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from common import (
    LLAMA_PIN_SHA,
    SCHEMA_VERSION,
    BenchmarkError,
    ManagedServer,
    collect_hardware_metadata,
    derive_measured_batch_metrics,
    git_metadata,
    http_request,
    load_workload,
    model_metadata,
    summarize_latencies,
    verify_expected_model_checksum,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKLOAD = Path(__file__).resolve().parent / "workloads" / "real_mixed.jsonl"


PRESETS: Dict[str, Dict[str, Any]] = {
    "real-batching-baseline": {
        "backend": "llama",
        "workers": 1,
        "ctx_size": 2048,
        "max_batch_size": 1,
        "max_batch_wait_ms": 0,
        "scheduler": "workload",
        "scheduler_capacity": 32,
        "worker_queue_capacity": 8,
        "max_output_tokens": 32,
        "concurrencies": [1, 2, 4],
        "cold_requests": 1,
        "warmup_requests": 5,
        "measured_requests": 60,
        "runs": 3,
    },
    "real-batching-adaptive": {
        "backend": "llama",
        "workers": 1,
        "ctx_size": 2048,
        "max_batch_size": 4,
        "max_batch_wait_ms": 3,
        "scheduler": "workload",
        "scheduler_capacity": 32,
        "worker_queue_capacity": 8,
        "max_output_tokens": 32,
        "concurrencies": [1, 2, 4],
        "cold_requests": 1,
        "warmup_requests": 5,
        "measured_requests": 60,
        "runs": 3,
    },
    "overload": {
        "backend": "llama",
        "workers": 1,
        "ctx_size": 2048,
        "max_batch_size": 1,
        "max_batch_wait_ms": 0,
        "scheduler": "workload",
        "scheduler_capacity": 4,
        "worker_queue_capacity": 2,
        "max_output_tokens": 32,
        "concurrencies": [32],
        "cold_requests": 1,
        "warmup_requests": 0,
        "measured_requests": 64,
        "runs": 3,
    },
}


def build_server_args(cfg: Dict[str, Any], model_path: Optional[Path], host: str) -> List[str]:
    args = [
        "--host",
        host,
        "--port",
        "0",
        "--backend",
        str(cfg["backend"]),
        "--workers",
        str(cfg["workers"]),
        "--max-batch-size",
        str(cfg["max_batch_size"]),
        "--max-batch-wait-ms",
        str(cfg["max_batch_wait_ms"]),
        "--scheduler-capacity",
        str(cfg["scheduler_capacity"]),
        "--worker-queue-capacity",
        str(cfg["worker_queue_capacity"]),
        "--scheduler",
        str(cfg["scheduler"]),
    ]
    if cfg["backend"] == "llama":
        if model_path is None:
            raise BenchmarkError("llama backend requires --model-path")
        args.extend(
            [
                "--model-id",
                str(cfg.get("model_id", "qwen-small")),
                "--model-path",
                str(model_path),
                "--ctx-size",
                str(cfg["ctx_size"]),
            ]
        )
    return args


async def one_infer(
    host: str,
    port: int,
    model_id: str,
    prompt: str,
    max_output_tokens: int,
    timeout_ms: int,
) -> Dict[str, Any]:
    body = json.dumps(
        {
            "model_id": model_id,
            "prompt": prompt,
            "max_output_tokens": max_output_tokens,
            "timeout_ms": timeout_ms,
        }
    ).encode("utf-8")
    t0 = time.perf_counter()
    try:
        resp = await http_request(
            host,
            port,
            "POST",
            "/v1/infer",
            body=body,
            headers={"content-type": "application/json"},
            timeout_s=max(timeout_ms / 1000.0 + 5.0, 30.0),
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        payload: Dict[str, Any] = {}
        error_code = None
        output_tokens = 0
        try:
            payload = resp.json()
            if isinstance(payload.get("error"), dict):
                error_code = payload["error"].get("code")
            output = payload.get("output") or {}
            if isinstance(output, dict):
                output_tokens = int(output.get("output_tokens") or 0)
        except Exception:
            payload = {"raw": resp.text()}
        return {
            "status": resp.status,
            "latency_ms": latency_ms,
            "output_tokens": output_tokens,
            "error_code": error_code,
            "payload": payload,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "status": 0,
            "latency_ms": latency_ms,
            "output_tokens": 0,
            "error_code": "transport_error",
            "error": str(exc),
            "payload": {},
        }


async def run_phase(
    host: str,
    port: int,
    prompts: Sequence[Dict[str, str]],
    model_id: str,
    max_output_tokens: int,
    concurrency: int,
    request_count: int,
    timeout_ms: int,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    next_index = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_index
        while True:
            async with lock:
                if next_index >= request_count:
                    return
                idx = next_index
                next_index += 1
            prompt = prompts[idx % len(prompts)]
            item = await one_infer(
                host, port, model_id, prompt["prompt"], max_output_tokens, timeout_ms
            )
            item["id"] = f"{prompt['id']}-{idx}"
            results.append(item)

    await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(max(1, concurrency))])
    return results


def classify_results(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    successes = [r for r in items if r.get("status") == 200]
    rejections_503 = [r for r in items if r.get("status") == 503]
    other_http = [
        r for r in items if isinstance(r.get("status"), int) and r["status"] not in (0, 200, 503)
    ]
    transport = [r for r in items if r.get("status") == 0]
    latencies = [float(r["latency_ms"]) for r in successes]
    tokens = sum(int(r.get("output_tokens") or 0) for r in successes)
    return {
        "successes": successes,
        "rejections_503": rejections_503,
        "other_http_errors": other_http,
        "transport_errors": transport,
        "success_latencies_ms": latencies,
        "success_output_tokens": tokens,
    }


async def fetch_metrics(host: str, port: int) -> Dict[str, Any]:
    resp = await http_request(host, port, "GET", "/metrics", timeout_s=10.0)
    if resp.status != 200:
        raise BenchmarkError(f"/metrics status {resp.status}")
    return resp.json()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


async def execute_run(
    *,
    server_bin: Path,
    cfg: Dict[str, Any],
    model_path: Optional[Path],
    prompts: Sequence[Dict[str, str]],
    concurrency: int,
    host: str,
    scenario: str,
    run_index: int,
    repo_root: Path,
    require_clean: bool,
    measured_override: Optional[int],
    warmup_override: Optional[int],
    cold_override: Optional[int],
) -> Dict[str, Any]:
    project_sha, dirty = git_metadata(repo_root)
    if require_clean and dirty:
        raise BenchmarkError("working tree is dirty; refuse --require-clean run")

    # Frozen M6 llama presets: hard-fail before any cold/warmup/measured work.
    if cfg.get("backend") == "llama":
        if model_path is None:
            raise BenchmarkError("llama backend requires model_path")
        verify_expected_model_checksum(model_path)

    cold_n = cfg["cold_requests"] if cold_override is None else cold_override
    warmup_n = cfg["warmup_requests"] if warmup_override is None else warmup_override
    measured_n = cfg["measured_requests"] if measured_override is None else measured_override
    model_id = str(cfg.get("model_id", "qwen-small" if cfg["backend"] == "llama" else "model-a"))

    args = build_server_args(cfg, model_path, host)
    server = ManagedServer(server_bin, args, cwd=repo_root)
    port = None
    try:
        server.start()
        port = server._read_until_listen(timeout_s=120.0)
        await server.wait_healthy(host=host, overall_timeout_s=60.0)

        # Cold request(s): setup failures abort.
        for i in range(cold_n):
            prompt = prompts[i % len(prompts)]
            cold = await one_infer(
                host, port, model_id, prompt["prompt"], int(cfg["max_output_tokens"]), 120000
            )
            if cold["status"] != 200:
                raise BenchmarkError(f"cold request failed: {cold}")

        # Warmup discarded.
        if warmup_n > 0:
            warm = await run_phase(
                host,
                port,
                prompts,
                model_id,
                int(cfg["max_output_tokens"]),
                concurrency=min(concurrency, max(1, warmup_n)),
                request_count=warmup_n,
                timeout_ms=120000,
            )
            if any(r.get("status") != 200 for r in warm):
                raise BenchmarkError(f"warmup failure: {warm}")

        metrics_before = await fetch_metrics(host, port)
        wall0 = time.perf_counter()
        measured = await run_phase(
            host,
            port,
            prompts,
            model_id,
            int(cfg["max_output_tokens"]),
            concurrency=concurrency,
            request_count=measured_n,
            timeout_ms=120000,
        )
        wall1 = time.perf_counter()
        metrics_after = await fetch_metrics(host, port)
        wall_s = wall1 - wall0
        classified = classify_results(measured)
        latency_summary = (
            summarize_latencies(classified["success_latencies_ms"])
            if classified["success_latencies_ms"]
            else {"p50_ms": 0.0, "p95_ms": 0.0, "mean_ms": 0.0, "count": 0.0}
        )
        success_count = len(classified["successes"])
        token_sum = classified["success_output_tokens"]
        batch = derive_measured_batch_metrics(metrics_before, metrics_after)

        return {
            "schema_version": SCHEMA_VERSION,
            "project_git_sha": project_sha,
            "git_dirty": dirty,
            "llama_sha": LLAMA_PIN_SHA,
            "model": model_metadata(model_path) if model_path else None,
            "hardware": collect_hardware_metadata(),
            "scenario": scenario,
            "run_index": run_index,
            "configuration": {
                **{k: cfg[k] for k in cfg if k not in ("concurrencies", "runs")},
                "concurrency": concurrency,
                "host": host,
                "port": port,
            },
            "warmup_discarded": warmup_n,
            "cold_requests": cold_n,
            "wall_time_seconds": wall_s,
            "requests": [
                {
                    "id": r.get("id"),
                    "status": r.get("status"),
                    "latency_ms": r.get("latency_ms"),
                    "output_tokens": r.get("output_tokens"),
                    "error_code": r.get("error_code"),
                }
                for r in measured
            ],
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "run_summary": {
                "successes": success_count,
                "rejections_503": len(classified["rejections_503"]),
                "other_http_errors": len(classified["other_http_errors"]),
                "transport_errors": len(classified["transport_errors"]),
                "successful_rps": (success_count / wall_s) if wall_s > 0 else 0.0,
                "successful_output_tokens_per_second": (token_sum / wall_s) if wall_s > 0 else 0.0,
                "p50_ms": latency_summary["p50_ms"],
                "p95_ms": latency_summary["p95_ms"],
                "mean_batch_size": batch["mean_batch_size"],
                "multi_request_batches": batch["multi_request_batches"],
                "multi_request_batch_rate": batch["multi_request_batch_rate"],
                "batches_executed": batch["batches_executed"],
                "requests_executed_via_batches": batch["requests_executed_via_batches"],
            },
        }
    finally:
        server.stop()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIRUNTIME benchmark runner")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        help="Named reproducible scenario preset",
    )
    parser.add_argument("--server-bin", type=Path, required=False)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/airuntime-bench-out"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--measured-requests", type=int, default=None)
    parser.add_argument("--warmup-requests", type=int, default=None)
    parser.add_argument("--cold-requests", type=int, default=None)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Tiny functional smoke overrides")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.preset:
        print("error: --preset is required", file=sys.stderr)
        return 2
    cfg = dict(PRESETS[args.preset])
    server_bin = args.server_bin
    if server_bin is None:
        candidates = [
            REPO_ROOT / "build-llama" / "runtime-server",
            REPO_ROOT / "build" / "runtime-server",
        ]
        for cand in candidates:
            if cand.is_file():
                server_bin = cand
                break
    if server_bin is None or not server_bin.is_file():
        print("error: --server-bin required (runtime-server not found)", file=sys.stderr)
        return 2

    model_path = args.model_path
    if cfg["backend"] == "llama" and model_path is None:
        default_model = Path.home() / "Models/adaptive_ai_inference_runtime/Qwen3.5-0.8B-Q4_0.gguf"
        if default_model.is_file():
            model_path = default_model
        else:
            print("error: --model-path required for llama presets", file=sys.stderr)
            return 2

    if args.smoke:
        cfg["runs"] = 1
        measured_override = 4 if args.preset != "overload" else 16
        warmup_override = 1 if args.preset != "overload" else 0
        cold_override = 1
        concurrencies = [args.concurrency or (1 if args.preset != "overload" else 8)]
    else:
        measured_override = args.measured_requests
        warmup_override = args.warmup_requests
        cold_override = args.cold_requests
        concurrencies = (
            [args.concurrency] if args.concurrency is not None else list(cfg["concurrencies"])
        )
        if args.runs is not None:
            cfg["runs"] = args.runs

    prompts = load_workload(args.workload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    for concurrency in concurrencies:
        runs = int(cfg["runs"])
        for run_index in range(runs):
            scenario = f"{args.preset}/c{concurrency}"
            print(f"running {scenario} run={run_index} ...", flush=True)
            result = asyncio.run(
                execute_run(
                    server_bin=server_bin,
                    cfg=cfg,
                    model_path=model_path,
                    prompts=prompts,
                    concurrency=concurrency,
                    host=args.host,
                    scenario=scenario,
                    run_index=run_index,
                    repo_root=REPO_ROOT,
                    require_clean=args.require_clean,
                    measured_override=measured_override,
                    warmup_override=warmup_override,
                    cold_override=cold_override,
                )
            )
            out = args.output_dir / f"{args.preset}_c{concurrency}_run{run_index}.json"
            write_json(out, result)
            written.append(out)
            summary = result["run_summary"]
            print(
                f"  wrote {out.name}: rps={summary['successful_rps']:.2f} "
                f"p95={summary['p95_ms']:.1f}ms successes={summary['successes']} "
                f"503={summary['rejections_503']}",
                flush=True,
            )

    print(json.dumps({"written": [str(p) for p in written]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BenchmarkError as exc:
        print(f"benchmark error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
