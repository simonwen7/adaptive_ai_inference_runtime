import { useMemo, useState } from "react";
import { flowStages } from "../../data/architecture";
import { useReducedMotion } from "../../lib/useReducedMotion";
import FlowStep from "./FlowStep";

export default function RequestFlowDiagram() {
  const [activeIndex, setActiveIndex] = useState(0);
  const reducedMotion = useReducedMotion();
  const active = flowStages[activeIndex];
  const markerStyle = useMemo(() => {
    if (reducedMotion) return {};
    const pct = (activeIndex / (flowStages.length - 1)) * 100;
    return { left: `${pct}%` };
  }, [activeIndex, reducedMotion]);

  return (
    <div className="space-y-6">
      <div className="relative rounded-[24px] border border-white/12 bg-white/[0.03] p-4 sm:p-6">
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-white/60">Run request flow (illustrative architecture animation)</p>
          <button
            type="button"
            onClick={() => setActiveIndex((current) => (current + 1) % flowStages.length)}
            className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-4 py-1.5 text-xs text-cyan-50"
          >
            Run request flow
          </button>
        </div>
        <div className="relative hidden h-10 items-center sm:flex">
          <div className="absolute left-0 right-0 h-px bg-white/20" />
          <span
            className="absolute h-3 w-3 -translate-x-1/2 rounded-full bg-cyan-300 transition-all duration-500"
            style={markerStyle}
          />
        </div>
        <div className="mt-4 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {flowStages.map((stage, index) => (
            <FlowStep
              key={stage.id}
              active={index === activeIndex}
              label={stage.label}
              zone={stage.zone}
              onClick={() => setActiveIndex(index)}
            />
          ))}
        </div>
      </div>
      <div className="rounded-[24px] border border-white/12 bg-black/20 p-5">
        <p className="text-[11px] uppercase tracking-[0.24em] text-cyan-100/55">{active.zone}</p>
        <h3 className="mt-2 text-xl font-semibold text-white">{active.label}</h3>
        <p className="mt-3 text-white/65">{active.mechanism}</p>
        <p className="mt-2 text-sm text-white/50">{active.why}</p>
      </div>
    </div>
  );
}
