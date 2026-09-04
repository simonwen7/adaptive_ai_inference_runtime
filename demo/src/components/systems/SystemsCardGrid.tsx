import { systemsCards } from "../../data/architecture";

export default function SystemsCardGrid() {
  return (
    <div className="grid gap-3 sm:grid-cols-2">
      {systemsCards.map(([title, mechanism, why]) => (
        <article key={title} className="rounded-[22px] border border-white/12 bg-white/[0.03] p-5">
          <h3 className="text-lg font-semibold text-white">{title}</h3>
          <p className="mt-3 text-sm text-white/65">
            <span className="text-white/80">Mechanism:</span> {mechanism}
          </p>
          <p className="mt-2 text-sm text-white/55">
            <span className="text-white/70">Why it matters:</span> {why}
          </p>
        </article>
      ))}
    </div>
  );
}
