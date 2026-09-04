import Header from "./components/layout/Header";
import Hero from "./components/hero/Hero";
import DisclosureBanner from "./components/layout/DisclosureBanner";
import Section from "./components/layout/Section";
import RequestFlowDiagram from "./components/flow/RequestFlowDiagram";
import SystemsCardGrid from "./components/systems/SystemsCardGrid";
import BatchingLab from "./components/batching/BatchingLab";
import WorkloadDependency from "./components/workload/WorkloadDependency";
import LlamaBackendStack from "./components/backend/LlamaBackendStack";
import BackpressureQueue from "./components/reliability/BackpressureQueue";
import EvidencePanel from "./components/evidence/EvidencePanel";
import { benchmarkData } from "./data/benchmark-data";
import { githubUrl } from "./data/architecture";

export default function App() {
  const overloadConfig = benchmarkData.overload.configuration;
  return (
    <div className="min-h-screen bg-[#05070b] text-white">
      <Header />
      <main id="main-content">
        <Hero />
        <DisclosureBanner />
        <Section
          id="flow"
          eyebrow="Interactive Request Flow"
          title="How one request moves through the runtime"
          subtitle="Select any stage to inspect mechanism and rationale. Animation is architectural, not live inference telemetry."
        >
          <RequestFlowDiagram />
        </Section>
        <Section id="systems" eyebrow="Systems Engineering" title="Core runtime mechanisms">
          <SystemsCardGrid />
        </Section>
        <Section
          id="batching"
          eyebrow="Dynamic Batching Lab"
          title="Frozen evidence, interactive exploration"
          subtitle="Switch mode and concurrency to inspect real measured medians and run-level evidence."
        >
          <BatchingLab />
        </Section>
        <Section
          id="workload"
          eyebrow="Workload-Dependent Results"
          title="Batch formation is not the same as a performance win"
        >
          <WorkloadDependency />
        </Section>
        <Section
          id="backend"
          eyebrow="Real llama.cpp Backend"
          title="Execution stack and model boundary"
        >
          <LlamaBackendStack />
        </Section>
        <Section
          id="reliability"
          eyebrow="Reliability / Backpressure"
          title="Bounded overload rejection behavior"
          subtitle={`Frozen overload preset (workers=${overloadConfig.workers}, max_batch_size=${overloadConfig.max_batch_size}, scheduler_capacity=${overloadConfig.scheduler_capacity}, worker_queue_capacity=${overloadConfig.worker_queue_capacity}, concurrency=${overloadConfig.concurrency}, measured_requests=${overloadConfig.measured_requests}). Rejection rate: ${benchmarkData.overload.median.rejection_rate_pct.toFixed(1)}%.`}
        >
          <BackpressureQueue />
        </Section>
        <Section id="evidence" eyebrow="Evidence / Reproducibility" title="Benchmark provenance and methodology">
          <EvidencePanel />
        </Section>
        <section className="mx-auto w-full max-w-6xl px-6 pb-16 pt-6 sm:px-8">
          <div className="rounded-[24px] border border-white/12 bg-white/[0.03] p-8 text-center">
            <h2 className="text-2xl font-semibold tracking-[-0.03em]">Built to make inference-runtime trade-offs visible.</h2>
            <div className="mt-6 flex flex-wrap justify-center gap-3">
              <a href={githubUrl} target="_blank" rel="noopener noreferrer" className="rounded-full border border-white/15 bg-white/[0.06] px-5 py-2.5 text-sm text-white/85">
                View Source on GitHub ↗
              </a>
              <a href={`${githubUrl}/blob/main/docs/benchmarks.md`} target="_blank" rel="noopener noreferrer" className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-5 py-2.5 text-sm text-cyan-50">
                Read Benchmark Evidence ↗
              </a>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
