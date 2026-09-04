import { benchmarkData } from "../../data/benchmark-data";
import { useReducedMotion } from "../../lib/useReducedMotion";

export default function BackpressureQueue() {
  const reducedMotion = useReducedMotion();
  const run = benchmarkData.overload.runs[0];
  const accepted = run.http_200;
  const rejected = run.http_503;
  const total = accepted + rejected;

  return (
    <div className="rounded-[24px] border border-white/12 bg-white/[0.03] p-5">
      <p className="text-sm text-white/65">
        64 request markers below represent measured counts. Motion is illustrative only.
      </p>
      <div
        className={`mt-4 grid grid-cols-8 gap-1.5 sm:grid-cols-16 ${reducedMotion ? "" : "queue-pulse"}`}
        aria-label="Overload request outcomes"
      >
        {Array.from({ length: total }, (_, i) => (
          <span
            key={i}
            className={`h-3 rounded ${i < accepted ? "bg-emerald-300/80" : "bg-rose-300/80"}`}
            title={i < accepted ? "Accepted (HTTP 200)" : "Bounded rejection (HTTP 503)"}
          />
        ))}
      </div>
      <div className="mt-4 grid gap-2 text-sm text-white/75 sm:grid-cols-2">
        <p>Accepted (HTTP 200): {accepted}</p>
        <p>Bounded rejects (HTTP 503): {rejected}</p>
        <p>Other HTTP errors: {run.other_http}</p>
        <p>Transport errors: {run.transport_errors}</p>
      </div>
    </div>
  );
}
