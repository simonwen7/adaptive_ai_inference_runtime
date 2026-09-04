import { getCell, getComparison } from "../../data/benchmark-data";

function Card({ concurrency }: { concurrency: 2 | 4 }) {
  const adaptive = getCell("adaptive", concurrency);
  const cmp = getComparison(concurrency);
  return (
    <article className="rounded-[22px] border border-white/12 bg-white/[0.03] p-5">
      <p className="text-[11px] uppercase tracking-[0.26em] text-cyan-100/55">Concurrency {concurrency}</p>
      <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
        <p className="text-white/60">Mean batch</p><p className="text-white">{adaptive.median.mean_batch_size.toFixed(2)}</p>
        <p className="text-white/60">Multi-request rate</p><p className="text-white">{(adaptive.median.multi_request_batch_rate * 100).toFixed(0)}%</p>
        <p className="text-white/60">Delta RPS</p><p className={cmp.delta_rps_pct < 0 ? "text-rose-300" : "text-emerald-300"}>{cmp.delta_rps_pct > 0 ? "+" : ""}{cmp.delta_rps_pct.toFixed(1)}%</p>
        <p className="text-white/60">Delta p95</p><p className={cmp.delta_p95_pct > 0 ? "text-rose-300" : "text-emerald-300"}>{cmp.delta_p95_pct > 0 ? "+" : ""}{cmp.delta_p95_pct.toFixed(1)}%</p>
      </div>
      <p className="mt-4 text-sm text-white/65">
        {concurrency === 2
          ? "Batches formed, but end-to-end performance worsened in this frozen campaign."
          : "Higher concurrency provided enough parallel work for this workload to benefit from static batching."}
      </p>
    </article>
  );
}

export default function WorkloadDependency() {
  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card concurrency={2} />
      <Card concurrency={4} />
    </div>
  );
}
