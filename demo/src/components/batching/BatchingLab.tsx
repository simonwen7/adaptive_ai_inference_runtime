import { useState } from "react";
import type { Concurrency, Mode } from "../../data/benchmark-data";
import { getCell, getComparison } from "../../data/benchmark-data";
import ConcurrencySelector from "./ConcurrencySelector";
import MetricPanel from "./MetricPanel";
import RunVariance from "./RunVariance";

export default function BatchingLab() {
  const [mode, setMode] = useState<Mode>("adaptive");
  const [concurrency, setConcurrency] = useState<Concurrency>(4);
  const [showRuns, setShowRuns] = useState(false);
  const cell = getCell(mode, concurrency);
  const cmp = getComparison(concurrency);

  return (
    <div className="rounded-[24px] border border-white/12 bg-white/[0.03] p-5 sm:p-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-2">
          <p className="text-[10px] uppercase tracking-[0.24em] text-white/45">Mode</p>
          <div className="flex gap-2">
            {(["baseline", "adaptive"] as const).map((item) => (
              <button
                key={item}
                type="button"
                onClick={() => setMode(item)}
                className={`rounded-full border px-4 py-2 text-sm ${
                  mode === item
                    ? "border-cyan-300/30 bg-cyan-400/[0.08] text-cyan-50"
                    : "border-white/15 bg-white/[0.04] text-white/70"
                }`}
              >
                {item[0].toUpperCase() + item.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="space-y-2">
          <p className="text-[10px] uppercase tracking-[0.24em] text-white/45">Concurrency</p>
          <ConcurrencySelector value={concurrency} onChange={setConcurrency} />
        </div>
      </div>

      <div className="mt-6 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        <MetricPanel label="RPS" value={cell.median.rps.toFixed(2)} />
        <MetricPanel label="Serving output tok/s" value={cell.median.output_tokens_per_s.toFixed(2)} />
        <MetricPanel label="p50 HTTP E2E" value={`${cell.median.p50_ms.toFixed(1)} ms`} />
        <MetricPanel label="p95 HTTP E2E" value={`${cell.median.p95_ms.toFixed(1)} ms`} />
        <MetricPanel label="Mean batch size" value={cell.median.mean_batch_size.toFixed(2)} />
        <MetricPanel label="Multi-request batch rate" value={`${(cell.median.multi_request_batch_rate * 100).toFixed(0)}%`} />
      </div>

      <div className="mt-6 rounded-2xl border border-white/12 bg-black/20 p-4 text-sm text-white/75">
        <p className="font-medium text-white/90">Baseline vs Adaptive at c{concurrency}</p>
        <p className="mt-2">ΔRPS: {cmp.delta_rps_pct > 0 ? "+" : ""}{cmp.delta_rps_pct.toFixed(1)}%</p>
        <p>ΔServing output tok/s: {cmp.delta_output_tokens_per_s_pct > 0 ? "+" : ""}{cmp.delta_output_tokens_per_s_pct.toFixed(1)}%</p>
        <p>Δp95: {cmp.delta_p95_pct > 0 ? "+" : ""}{cmp.delta_p95_pct.toFixed(1)}%</p>
        {concurrency === 1 ? <p className="mt-2 text-cyan-100/80">High variance: baseline RPS span 0.76-5.32.</p> : null}
      </div>

      <button
        type="button"
        onClick={() => setShowRuns((v) => !v)}
        className="mt-4 rounded-full border border-white/15 bg-white/[0.04] px-4 py-2 text-sm text-white/80"
      >
        {showRuns ? "Hide run-level evidence" : "View run-level evidence"}
      </button>
      <RunVariance runs={cell.runs} show={showRuns} />
    </div>
  );
}
