#!/usr/bin/env python3
"""Aggregate benchmark run JSON files into summary tables (stdlib only)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from common import median


def load_runs(paths: Sequence[Path]) -> List[Dict[str, Any]]:
    runs: List[Dict[str, Any]] = []
    for path in paths:
        runs.append(json.loads(path.read_text(encoding="utf-8")))
    return runs


def aggregate(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[str(run.get("scenario", "unknown"))].append(run)

    cells: List[Dict[str, Any]] = []
    for scenario, items in sorted(groups.items()):
        def values(key: str) -> List[float]:
            return [float(item["run_summary"][key]) for item in items]

        cell = {
            "scenario": scenario,
            "runs": len(items),
            "median_successful_rps": median(values("successful_rps")),
            "median_successful_output_tokens_per_second": median(
                values("successful_output_tokens_per_second")
            ),
            "median_p50_ms": median(values("p50_ms")),
            "median_p95_ms": median(values("p95_ms")),
            "median_mean_batch_size": median(values("mean_batch_size")),
            "median_multi_request_batch_rate": median(values("multi_request_batch_rate")),
            "median_rejections_503": median(values("rejections_503")),
            "run_values": [
                {
                    "run_index": item.get("run_index"),
                    "successful_rps": item["run_summary"]["successful_rps"],
                    "successful_output_tokens_per_second": item["run_summary"][
                        "successful_output_tokens_per_second"
                    ],
                    "p50_ms": item["run_summary"]["p50_ms"],
                    "p95_ms": item["run_summary"]["p95_ms"],
                    "mean_batch_size": item["run_summary"]["mean_batch_size"],
                    "multi_request_batch_rate": item["run_summary"][
                        "multi_request_batch_rate"
                    ],
                    "successes": item["run_summary"]["successes"],
                    "rejections_503": item["run_summary"]["rejections_503"],
                }
                for item in items
            ],
            "project_git_sha": items[0].get("project_git_sha"),
            "llama_sha": items[0].get("llama_sha"),
            "model": items[0].get("model"),
            "hardware": items[0].get("hardware"),
            "configuration": items[0].get("configuration"),
        }
        cells.append(cell)
    return {"schema_version": 1, "cells": cells}


def markdown_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "| Scenario | Runs | RPS | tokens/s | p50 ms | p95 ms | mean batch | multi-batch rate | 503 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["cells"]:
        lines.append(
            "| {scenario} | {runs} | {rps:.2f} | {tps:.2f} | {p50:.1f} | {p95:.1f} | "
            "{batch:.2f} | {mrate:.2f} | {rej:.0f} |".format(
                scenario=cell["scenario"],
                runs=cell["runs"],
                rps=cell["median_successful_rps"],
                tps=cell["median_successful_output_tokens_per_second"],
                p50=cell["median_p50_ms"],
                p95=cell["median_p95_ms"],
                batch=cell["median_mean_batch_size"],
                mrate=cell["median_multi_request_batch_rate"],
                rej=cell["median_rejections_503"],
            )
        )
    return "\n".join(lines) + "\n"


def expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for item in inputs:
        path = Path(item)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.json")))
        else:
            paths.append(path)
    return paths


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="AIRUNTIME benchmark analyzer")
    parser.add_argument("inputs", nargs="+", help="Run JSON files or directories")
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    paths = expand_inputs(args.inputs)
    if not paths:
        print("no input files", file=sys.stderr)
        return 2
    runs = load_runs(paths)
    summary = aggregate(runs)
    table = markdown_table(summary)
    print(table)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
