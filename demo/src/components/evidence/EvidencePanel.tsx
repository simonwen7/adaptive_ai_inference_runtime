import { benchmarkData } from "../../data/benchmark-data";
import { githubUrl } from "../../data/architecture";

export default function EvidencePanel() {
  return (
    <div className="rounded-[24px] border border-white/12 bg-white/[0.03] p-5 sm:p-6">
      <div className="grid gap-4 text-sm text-white/75 sm:grid-cols-2">
        <p><span className="text-white/55">Benchmark source SHA:</span> <span className="break-all">{benchmarkData.source.project_git_sha}</span></p>
        <p><span className="text-white/55">llama.cpp pin:</span> <span className="break-all">{benchmarkData.source.llama_git_sha}</span></p>
        <p><span className="text-white/55">Model:</span> {benchmarkData.model.filename}</p>
        <p><span className="text-white/55">Model SHA:</span> <span className="break-all">{benchmarkData.model.sha256}</span></p>
        <p><span className="text-white/55">Hardware:</span> Apple M4 Pro · 24 GB unified memory · Metal</p>
        <p><span className="text-white/55">Method:</span> 3 runs/cell, 1 cold, 5 warmup, 60 measured, run-level medians, HTTP E2E latency</p>
      </div>
      <div className="mt-5 flex flex-wrap gap-3">
        <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="rounded-full border border-white/15 bg-white/[0.06] px-4 py-2 text-sm text-white/85">
          View Source on GitHub ↗
        </a>
        <a href={`${githubUrl}/blob/main/docs/benchmarks.md`} target="_blank" rel="noopener noreferrer" className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-4 py-2 text-sm text-cyan-50">
          Read Benchmark Evidence ↗
        </a>
      </div>
    </div>
  );
}
