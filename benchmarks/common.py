"""Shared helpers for Adaptive AI Inference Runtime benchmarks (stdlib only)."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import platform
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_VERSION = 1
LLAMA_PIN_SHA = "9cc33944f9b7a44243618d5522adae357d7fdc90"
EXPECTED_MODEL_SHA256 = "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf"


class BenchmarkError(RuntimeError):
    """Harness-level failure."""


def nearest_rank_percentile(sorted_values: Sequence[float], percentile: float) -> float:
    """Nearest-rank percentile. percentile in (0, 100]."""
    if not sorted_values:
        raise ValueError("empty sample")
    if percentile <= 0 or percentile > 100:
        raise ValueError("percentile must be in (0, 100]")
    n = len(sorted_values)
    rank = int(math.ceil((percentile / 100.0) * n))
    rank = min(max(rank, 1), n)
    return float(sorted_values[rank - 1])


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("empty sample")
    ordered = sorted(float(v) for v in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata(repo_root: Path) -> Tuple[str, bool]:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), text=True
    ).strip()
    dirty = (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root), text=True
        ).strip()
        != ""
    )
    return sha, dirty


def collect_hardware_metadata() -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "processor": platform.processor() or None,
    }
    try:
        uname = os.uname()
        meta["sysname"] = uname.sysname
        meta["release"] = uname.release
    except AttributeError:
        pass
    if sys.platform == "darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
            meta["cpu_model"] = brand
        except (OSError, subprocess.CalledProcessError):
            pass
        try:
            mem = int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            )
            meta["physical_memory_bytes"] = mem
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
    try:
        meta["compiler"] = subprocess.check_output(
            ["c++", "--version"], text=True, stderr=subprocess.STDOUT
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    try:
        meta["cmake"] = subprocess.check_output(
            ["cmake", "--version"], text=True
        ).splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        pass
    return meta


def model_metadata(model_path: Path) -> Dict[str, Any]:
    return {
        "filename": model_path.name,
        "size_bytes": model_path.stat().st_size,
        "sha256": sha256_file(model_path),
    }


def verify_expected_model_checksum(
    model_path: Path,
    expected_sha256: str = EXPECTED_MODEL_SHA256,
) -> str:
    """SHA-256 of GGUF file bytes must match frozen expected checksum."""
    actual = sha256_file(model_path)
    if actual != expected_sha256:
        raise BenchmarkError(
            f"model checksum mismatch for {model_path.name}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return actual


@dataclass
class HttpResponse:
    status: int
    headers: Dict[str, str]
    body: bytes

    def text(self) -> str:
        return self.body.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.text())


async def http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    body: Optional[bytes] = None,
    headers: Optional[Mapping[str, str]] = None,
    timeout_s: float = 60.0,
) -> HttpResponse:
    """One HTTP/1.1 request per TCP connection (matches runtime-server)."""
    payload = body or b""
    header_map = {k.lower(): v for k, v in (headers or {}).items()}
    if "host" not in header_map:
        header_map["host"] = f"{host}:{port}"
    if "connection" not in header_map:
        header_map["connection"] = "close"
    if payload and "content-length" not in header_map:
        header_map["content-length"] = str(len(payload))
    if payload and "content-type" not in header_map:
        header_map["content-type"] = "application/json"

    request_lines = [f"{method.upper()} {path} HTTP/1.1"]
    for key, value in header_map.items():
        request_lines.append(f"{key}: {value}")
    request_lines.append("")
    request_lines.append("")
    raw = "\r\n".join(request_lines[:-1]).encode("ascii") + b"\r\n" + payload

    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None
    try:
        connect_coro = asyncio.open_connection(host, port)
        reader, writer = await asyncio.wait_for(connect_coro, timeout=timeout_s)
        writer.write(raw)
        await writer.drain()

        status_line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
        if not status_line:
            raise BenchmarkError("empty HTTP status line")
        match = re.match(rb"HTTP/1\.[01] (\d{3}) ", status_line)
        if not match:
            raise BenchmarkError(f"malformed status line: {status_line!r}")
        status = int(match.group(1))

        resp_headers: Dict[str, str] = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout=timeout_s)
            if line in (b"\r\n", b"\n", b""):
                break
            text = line.decode("latin-1").rstrip("\r\n")
            if ":" not in text:
                raise BenchmarkError(f"malformed header: {text!r}")
            name, value = text.split(":", 1)
            resp_headers[name.strip().lower()] = value.strip()

        if "content-length" not in resp_headers:
            # Connection: close with no length — read until EOF.
            body_bytes = await asyncio.wait_for(reader.read(), timeout=timeout_s)
        else:
            length = int(resp_headers["content-length"])
            if length < 0:
                raise BenchmarkError("negative Content-Length")
            body_bytes = await asyncio.wait_for(reader.readexactly(length), timeout=timeout_s)
        return HttpResponse(status=status, headers=resp_headers, body=body_bytes)
    finally:
        if writer is not None:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass


LISTEN_RE = re.compile(r"Listening on http://[^:]+:(\d+)")


class ManagedServer:
    def __init__(
        self,
        binary: Path,
        args: Sequence[str],
        cwd: Optional[Path] = None,
    ) -> None:
        self.binary = binary
        self.args = list(args)
        self.cwd = cwd
        self.proc: Optional[subprocess.Popen[str]] = None
        self.port: Optional[int] = None
        self._stdout_buf: List[str] = []
        self._drain_thread: Optional[Any] = None
        self._stop_drain = False

    def start(self) -> None:
        cmd = [str(self.binary), *self.args]
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.cwd) if self.cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._stop_drain = False

        def drain() -> None:
            assert self.proc is not None and self.proc.stdout is not None
            while not self._stop_drain:
                line = self.proc.stdout.readline()
                if not line:
                    break
                self._stdout_buf.append(line)
                if len(self._stdout_buf) > 5000:
                    del self._stdout_buf[:-2000]

        self._drain_thread = threading.Thread(target=drain, daemon=True)
        self._drain_thread.start()

    def _read_until_listen(self, timeout_s: float = 30.0) -> int:
        assert self.proc is not None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BenchmarkError(
                    "runtime-server exited early:\n" + "".join(self._stdout_buf[-50:])
                )
            for line in list(self._stdout_buf):
                match = LISTEN_RE.search(line)
                if match:
                    self.port = int(match.group(1))
                    return self.port
            time.sleep(0.02)
        raise BenchmarkError(
            "timed out waiting for Listening line:\n" + "".join(self._stdout_buf[-50:])
        )

    async def wait_healthy(
        self, host: str = "127.0.0.1", overall_timeout_s: float = 15.0
    ) -> None:
        assert self.port is not None
        deadline = time.monotonic() + overall_timeout_s
        delay = 0.05
        last_error: Optional[str] = None
        while time.monotonic() < deadline:
            try:
                resp = await http_request(host, self.port, "GET", "/health", timeout_s=2.0)
                if resp.status == 200:
                    return
                last_error = f"status={resp.status}"
            except Exception as exc:  # noqa: BLE001 - collect readiness errors
                last_error = str(exc)
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 0.5)
        raise BenchmarkError(f"health readiness timeout: {last_error}")

    def stop(self, grace_s: float = 10.0) -> None:
        self._stop_drain = True
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    self.proc.kill()
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        if self._drain_thread is not None:
            self._drain_thread.join(timeout=1.0)
            self._drain_thread = None
        self.proc = None

    def start_and_ready(self, host: str = "127.0.0.1") -> int:
        self.start()
        port = self._read_until_listen()
        asyncio.run(self.wait_healthy(host=host))
        return port


def load_workload(path: Path) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "id" not in obj or "prompt" not in obj:
                raise BenchmarkError(f"workload line {line_no} missing id/prompt")
            prompts.append({"id": str(obj["id"]), "prompt": str(obj["prompt"])})
    if not prompts:
        raise BenchmarkError("empty workload")
    return prompts


def _sum_batch_counters(metrics: Mapping[str, Any]) -> Dict[str, int]:
    """Sum cumulative batch counters across workers (not gauges)."""
    workers = metrics.get("workers") or []
    batches = 0
    requests = 0
    multi = 0
    for worker in workers:
        bm = worker.get("batch_metrics") or {}
        batches += int(bm.get("batches_executed", 0))
        requests += int(bm.get("requests_executed_via_batches", 0))
        multi += int(bm.get("multi_request_batches", 0))
    return {
        "batches_executed": batches,
        "requests_executed_via_batches": requests,
        "multi_request_batches": multi,
    }


def _safe_counter_delta(name: str, before: int, after: int) -> int:
    if after < before:
        raise BenchmarkError(
            f"cumulative counter {name} decreased: before={before} after={after}"
        )
    return after - before


def derive_measured_batch_metrics(
    metrics_before: Mapping[str, Any],
    metrics_after: Mapping[str, Any],
) -> Dict[str, Any]:
    """Measured-window batch metrics from cumulative counter deltas.

    Runtime counters are cumulative. Isolate the measured campaign with:

        snapshot after cold+warmup  (metrics_before)
        snapshot after measured     (metrics_after)

    Gauges such as max_batch_size_observed are intentionally omitted —
    they are process-lifetime diagnostics and cannot be recovered by subtraction.
    """
    before = _sum_batch_counters(metrics_before)
    after = _sum_batch_counters(metrics_after)
    batches = _safe_counter_delta(
        "batches_executed", before["batches_executed"], after["batches_executed"]
    )
    requests = _safe_counter_delta(
        "requests_executed_via_batches",
        before["requests_executed_via_batches"],
        after["requests_executed_via_batches"],
    )
    multi = _safe_counter_delta(
        "multi_request_batches",
        before["multi_request_batches"],
        after["multi_request_batches"],
    )
    mean_batch = (float(requests) / float(batches)) if batches > 0 else 0.0
    multi_rate = (float(multi) / float(batches)) if batches > 0 else 0.0
    return {
        "batches_executed": batches,
        "requests_executed_via_batches": requests,
        "multi_request_batches": multi,
        "mean_batch_size": mean_batch,
        "multi_request_batch_rate": multi_rate,
    }


def summarize_latencies(latencies_ms: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(x) for x in latencies_ms)
    return {
        "p50_ms": nearest_rank_percentile(ordered, 50),
        "p95_ms": nearest_rank_percentile(ordered, 95),
        "mean_ms": sum(ordered) / len(ordered) if ordered else 0.0,
        "count": float(len(ordered)),
    }
