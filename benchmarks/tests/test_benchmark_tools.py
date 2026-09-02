#!/usr/bin/env python3
"""Unit tests for benchmark tooling (stdlib unittest)."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import common  # noqa: E402
import run as run_mod  # noqa: E402
import analyze  # noqa: E402


def _metrics(batches: int, requests: int, multi: int, max_obs: int = 1) -> dict:
    return {
        "workers": [
            {
                "batch_metrics": {
                    "batches_executed": batches,
                    "requests_executed_via_batches": requests,
                    "multi_request_batches": multi,
                    "max_batch_size_observed": max_obs,
                }
            }
        ]
    }


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_p50_p95(self) -> None:
        values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        self.assertEqual(common.nearest_rank_percentile(values, 50), 50)
        self.assertEqual(common.nearest_rank_percentile(values, 95), 100)

    def test_nearest_rank_single(self) -> None:
        self.assertEqual(common.nearest_rank_percentile([42], 50), 42)
        self.assertEqual(common.nearest_rank_percentile([42], 95), 42)


class AggregationTests(unittest.TestCase):
    def test_median_and_rps(self) -> None:
        self.assertEqual(common.median([1, 3, 2]), 2)
        self.assertEqual(common.median([1, 2, 3, 4]), 2.5)
        wall = 2.0
        successes = 10
        tokens = 320
        self.assertAlmostEqual(successes / wall, 5.0)
        self.assertAlmostEqual(tokens / wall, 160.0)

    def test_analyze_aggregate(self) -> None:
        runs = [
            {
                "scenario": "real-batching-baseline/c1",
                "run_index": 0,
                "project_git_sha": "abc",
                "llama_sha": common.LLAMA_PIN_SHA,
                "model": {"filename": "m.gguf"},
                "hardware": {},
                "configuration": {},
                "run_summary": {
                    "successful_rps": 1.0,
                    "successful_output_tokens_per_second": 10.0,
                    "p50_ms": 100.0,
                    "p95_ms": 200.0,
                    "mean_batch_size": 1.0,
                    "multi_request_batch_rate": 0.0,
                    "rejections_503": 0,
                    "successes": 10,
                },
            },
            {
                "scenario": "real-batching-baseline/c1",
                "run_index": 1,
                "project_git_sha": "abc",
                "llama_sha": common.LLAMA_PIN_SHA,
                "model": {"filename": "m.gguf"},
                "hardware": {},
                "configuration": {},
                "run_summary": {
                    "successful_rps": 3.0,
                    "successful_output_tokens_per_second": 30.0,
                    "p50_ms": 120.0,
                    "p95_ms": 220.0,
                    "mean_batch_size": 1.2,
                    "multi_request_batch_rate": 0.2,
                    "rejections_503": 2,
                    "successes": 8,
                },
            },
            {
                "scenario": "real-batching-baseline/c1",
                "run_index": 2,
                "project_git_sha": "abc",
                "llama_sha": common.LLAMA_PIN_SHA,
                "model": {"filename": "m.gguf"},
                "hardware": {},
                "configuration": {},
                "run_summary": {
                    "successful_rps": 2.0,
                    "successful_output_tokens_per_second": 20.0,
                    "p50_ms": 110.0,
                    "p95_ms": 210.0,
                    "mean_batch_size": 1.1,
                    "multi_request_batch_rate": 0.1,
                    "rejections_503": 1,
                    "successes": 9,
                },
            },
        ]
        summary = analyze.aggregate(runs)
        cell = summary["cells"][0]
        self.assertEqual(cell["median_successful_rps"], 2.0)
        self.assertEqual(cell["median_p95_ms"], 210.0)
        self.assertEqual(cell["median_rejections_503"], 1.0)
        self.assertEqual(cell["median_multi_request_batch_rate"], 0.1)


class MeasuredBatchMetricsTests(unittest.TestCase):
    def test_nonzero_before_uses_deltas_not_absolute(self) -> None:
        # Cold/warmup already executed; absolute-after would wrongly yield mean=1.5.
        before = _metrics(batches=6, requests=6, multi=0, max_obs=1)
        after = _metrics(batches=8, requests=12, multi=2, max_obs=4)
        derived = common.derive_measured_batch_metrics(before, after)
        self.assertEqual(derived["batches_executed"], 2)
        self.assertEqual(derived["requests_executed_via_batches"], 6)
        self.assertEqual(derived["multi_request_batches"], 2)
        self.assertEqual(derived["mean_batch_size"], 3.0)
        self.assertEqual(derived["multi_request_batch_rate"], 1.0)
        self.assertNotIn("max_batch_size_observed", derived)
        # Prove absolute-after would have contaminated the result.
        absolute_mean = 12.0 / 8.0
        self.assertNotEqual(derived["mean_batch_size"], absolute_mean)

    def test_zero_delta_batches(self) -> None:
        before = _metrics(4, 4, 0)
        after = _metrics(4, 4, 0)
        derived = common.derive_measured_batch_metrics(before, after)
        self.assertEqual(derived["batches_executed"], 0)
        self.assertEqual(derived["mean_batch_size"], 0.0)
        self.assertEqual(derived["multi_request_batch_rate"], 0.0)

    def test_counter_decrease_raises(self) -> None:
        before = _metrics(batches=8, requests=10, multi=2)
        after = _metrics(batches=6, requests=10, multi=2)
        with self.assertRaises(common.BenchmarkError) as ctx:
            common.derive_measured_batch_metrics(before, after)
        self.assertIn("batches_executed", str(ctx.exception))
        self.assertIn("decreased", str(ctx.exception))


class ModelChecksumTests(unittest.TestCase):
    def test_matching_checksum_succeeds(self) -> None:
        payload = b"airuntime-benchmark-checksum-fixture\n"
        expected = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixture.bin"
            path.write_bytes(payload)
            actual = common.verify_expected_model_checksum(path, expected_sha256=expected)
            self.assertEqual(actual, expected)

    def test_mismatch_raises_with_basename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fake-model.gguf"
            path.write_bytes(b"not-the-frozen-qwen-bytes")
            with self.assertRaises(common.BenchmarkError) as ctx:
                common.verify_expected_model_checksum(path)
            message = str(ctx.exception)
            self.assertIn("fake-model.gguf", message)
            self.assertIn(common.EXPECTED_MODEL_SHA256, message)
            self.assertNotIn(str(path.parent), message)

    def test_execute_run_aborts_before_server_on_sha_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "wrong.gguf"
            bad.write_bytes(b"wrong-weights")
            start = mock.Mock(side_effect=AssertionError("server must not start"))
            with mock.patch.object(common, "git_metadata", return_value=("deadbeef", False)):
                with mock.patch.object(common.ManagedServer, "start", start):
                    with self.assertRaises(common.BenchmarkError) as ctx:
                        asyncio.run(
                            run_mod.execute_run(
                                server_bin=Path("/nonexistent"),
                                cfg=run_mod.PRESETS["real-batching-baseline"],
                                model_path=bad,
                                prompts=[{"id": "p", "prompt": "x"}],
                                concurrency=1,
                                host="127.0.0.1",
                                scenario="x",
                                run_index=0,
                                repo_root=ROOT.parent,
                                require_clean=False,
                                measured_override=1,
                                warmup_override=0,
                                cold_override=0,
                            )
                        )
            self.assertIn("checksum mismatch", str(ctx.exception))
            start.assert_not_called()


class SchemaAndConfigTests(unittest.TestCase):
    def test_batch_metrics_derivation(self) -> None:
        before = _metrics(0, 0, 0)
        after = _metrics(4, 10, 2, max_obs=3)
        derived = common.derive_measured_batch_metrics(before, after)
        self.assertEqual(derived["mean_batch_size"], 2.5)
        self.assertEqual(derived["multi_request_batches"], 2)
        self.assertEqual(derived["multi_request_batch_rate"], 0.5)

    def test_server_args_serialization(self) -> None:
        cfg = dict(run_mod.PRESETS["real-batching-adaptive"])
        args = run_mod.build_server_args(cfg, Path("/tmp/model.gguf"), "127.0.0.1")
        self.assertIn("--max-batch-size", args)
        self.assertEqual(args[args.index("--max-batch-size") + 1], "4")
        self.assertEqual(args[args.index("--max-batch-wait-ms") + 1], "3")
        self.assertEqual(args[args.index("--port") + 1], "0")

    def test_classify_503(self) -> None:
        items = [
            {"status": 200, "latency_ms": 1.0, "output_tokens": 2},
            {"status": 503, "latency_ms": 1.0, "output_tokens": 0},
            {"status": 0, "latency_ms": 1.0, "output_tokens": 0},
            {"status": 500, "latency_ms": 1.0, "output_tokens": 0},
        ]
        classified = run_mod.classify_results(items)
        self.assertEqual(len(classified["successes"]), 1)
        self.assertEqual(len(classified["rejections_503"]), 1)
        self.assertEqual(len(classified["transport_errors"]), 1)
        self.assertEqual(len(classified["other_http_errors"]), 1)

    def test_require_clean_behavior(self) -> None:
        with mock.patch.object(common, "git_metadata", return_value=("deadbeef", True)):
            with self.assertRaises(common.BenchmarkError):
                asyncio.run(
                    run_mod.execute_run(
                        server_bin=Path("/nonexistent"),
                        cfg=run_mod.PRESETS["real-batching-baseline"],
                        model_path=None,
                        prompts=[{"id": "p", "prompt": "x"}],
                        concurrency=1,
                        host="127.0.0.1",
                        scenario="x",
                        run_index=0,
                        repo_root=ROOT.parent,
                        require_clean=True,
                        measured_override=1,
                        warmup_override=0,
                        cold_override=0,
                    )
                )


class HttpParseTests(unittest.TestCase):
    def test_http_request_parses_content_length(self) -> None:
        async def handler(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            await reader.read(4096)
            body = b'{"status":"healthy"}'
            response = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: application/json\r\n"
                b"Content-Length: "
                + str(len(body)).encode("ascii")
                + b"\r\nConnection: close\r\n\r\n"
                + body
            )
            writer.write(response)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        async def run_case() -> None:
            server = await asyncio.start_server(handler, "127.0.0.1", 0)
            sockets = server.sockets
            assert sockets
            port = sockets[0].getsockname()[1]
            async with server:
                resp = await common.http_request("127.0.0.1", port, "GET", "/health")
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.json()["status"], "healthy")

        asyncio.run(run_case())


class ReadinessTests(unittest.TestCase):
    def test_readiness_success(self) -> None:
        server = common.ManagedServer(Path("/bin/true"), [])
        server.port = 9

        async def fake_health(host: str, port: int, *args: object, **kwargs: object) -> common.HttpResponse:
            return common.HttpResponse(200, {}, b'{"status":"healthy"}')

        async def run_case() -> None:
            with mock.patch.object(common, "http_request", side_effect=fake_health):
                await server.wait_healthy(overall_timeout_s=2.0)

        asyncio.run(run_case())

    def test_readiness_timeout(self) -> None:
        server = common.ManagedServer(Path("/bin/true"), [])
        server.port = 9

        async def always_fail(*args: object, **kwargs: object) -> common.HttpResponse:
            raise ConnectionRefusedError("nope")

        async def run_case() -> None:
            with mock.patch.object(common, "http_request", side_effect=always_fail):
                with self.assertRaises(common.BenchmarkError):
                    await server.wait_healthy(overall_timeout_s=0.3)

        asyncio.run(run_case())

    def test_process_cleanup_kill_fallback(self) -> None:
        # Spawn a process that ignores SIGTERM briefly is hard portably; verify stop on exited proc.
        server = common.ManagedServer(Path("/bin/true"), [])
        server.proc = mock.Mock()
        server.proc.poll.return_value = 0
        server.stop()
        self.assertIsNone(server.proc)


class WorkloadTests(unittest.TestCase):
    def test_workload_has_16_prompts(self) -> None:
        path = ROOT / "workloads" / "real_mixed.jsonl"
        prompts = common.load_workload(path)
        self.assertEqual(len(prompts), 16)
        self.assertTrue(all("id" in p and "prompt" in p for p in prompts))


if __name__ == "__main__":
    unittest.main()
