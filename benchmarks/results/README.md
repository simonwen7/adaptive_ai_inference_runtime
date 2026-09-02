# Benchmark results

Final curated result JSON/Markdown files are produced **only** from a clean committed git SHA (after M6 infrastructure acceptance), using:

```bash
python3 benchmarks/run.py ... --require-clean --output-dir /tmp/...
python3 benchmarks/analyze.py /tmp/... --summary-json benchmarks/results/summary.json
```

Do not commit raw per-request dumps, server logs, or smoke outputs here.
