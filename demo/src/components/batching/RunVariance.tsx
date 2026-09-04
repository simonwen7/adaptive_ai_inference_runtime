type Run = {
  rps: number;
  output_tokens_per_s: number;
  p50_ms: number;
  p95_ms: number;
  mean_batch_size: number;
  multi_request_batch_rate: number;
};

type Props = {
  runs: Run[];
  show: boolean;
};

export default function RunVariance({ runs, show }: Props) {
  if (!show) return null;
  return (
    <div className="mt-4 overflow-x-auto rounded-2xl border border-white/12 bg-white/[0.02] p-4">
      <p className="mb-3 text-sm text-white/65">
        3 independent runs per cell. Medians are descriptive; no statistical significance claimed.
      </p>
      <table className="w-full min-w-[640px] text-left text-sm text-white/75">
        <thead className="text-white/50">
          <tr>
            <th className="pb-2">Run</th><th>RPS</th><th>tok/s</th><th>p50</th><th>p95</th><th>Batch</th><th>Multi</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run, idx) => (
            <tr key={idx} className="border-t border-white/10">
              <td className="py-2">#{idx}</td>
              <td>{run.rps.toFixed(2)}</td>
              <td>{run.output_tokens_per_s.toFixed(2)}</td>
              <td>{run.p50_ms.toFixed(1)} ms</td>
              <td>{run.p95_ms.toFixed(1)} ms</td>
              <td>{run.mean_batch_size.toFixed(2)}</td>
              <td>{(run.multi_request_batch_rate * 100).toFixed(0)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
