import { benchmarkData, getCell, getComparison } from "../../data/benchmark-data";
import { githubUrl } from "../../data/architecture";

export default function Hero() {
  const c4Base = getCell("baseline", 4);
  const c4Adaptive = getCell("adaptive", 4);
  const c4Cmp = getComparison(4);

  return (
    <section id="top" className="mx-auto w-full max-w-6xl px-6 pb-8 pt-14 sm:px-8 sm:pt-20">
      <p className="text-[11px] font-medium uppercase tracking-[0.3em] text-cyan-100/55">
        Systems Engineering / 03
      </p>
      <h1 className="mt-4 text-4xl font-semibold tracking-[-0.04em] text-white sm:text-6xl">
        Adaptive AI Inference Runtime
      </h1>
      <p className="mt-6 max-w-3xl text-lg text-white/65">
        A systems-oriented C++ inference runtime exploring batching, routing, residency,
        backpressure, and real llama.cpp serving.
      </p>
      <div className="mt-5 flex flex-wrap gap-2 text-xs text-white/65">
        {["C++20", "llama.cpp", "GGUF", "Metal", "Boost.Beast", "CMake"].map((pill) => (
          <span key={pill} className="rounded-full border border-white/12 bg-white/[0.04] px-3 py-1">
            {pill}
          </span>
        ))}
      </div>
      <div className="mt-8 flex flex-wrap items-center gap-3">
        <a
          href="#flow"
          className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-5 py-2.5 text-sm font-medium text-cyan-50 hover:bg-cyan-400/20"
        >
          Explore Runtime
        </a>
        <a
          href={githubUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="rounded-full border border-white/15 bg-white/[0.07] px-5 py-2.5 text-sm font-medium text-white/85 hover:bg-white/[0.12]"
        >
          GitHub ↗
        </a>
      </div>
      <div className="mt-10 grid gap-4 rounded-[24px] border border-white/12 bg-white/[0.03] p-6 sm:grid-cols-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-white/40">At concurrency 4</p>
          <p className="mt-2 text-3xl font-semibold text-cyan-100">+{c4Cmp.delta_rps_pct.toFixed(1)}%</p>
          <p className="text-sm text-white/55">median serving throughput</p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-white/40">Latency</p>
          <p className="mt-2 text-3xl font-semibold text-cyan-100">{c4Cmp.delta_p95_pct.toFixed(1)}%</p>
          <p className="text-sm text-white/55">p95 HTTP E2E latency</p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-white/40">Batch</p>
          <p className="mt-2 text-2xl font-semibold text-white">
            {c4Base.median.mean_batch_size.toFixed(2)} → {c4Adaptive.median.mean_batch_size.toFixed(2)}
          </p>
          <p className="text-sm text-white/55">mean batch size</p>
        </div>
        <div>
          <p className="text-[11px] uppercase tracking-[0.24em] text-white/40">Context</p>
          <p className="mt-2 text-sm text-white/80">
            {benchmarkData.hardware.cpu_model} · {Math.round(benchmarkData.hardware.physical_memory_bytes / 1024 ** 3)} GB
          </p>
          <p className="text-sm text-white/55">{benchmarkData.model.filename.replace("-Q4_0.gguf", " Q4_0")} · llama.cpp / Metal</p>
        </div>
      </div>
    </section>
  );
}
