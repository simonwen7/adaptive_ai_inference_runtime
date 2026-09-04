import type { Concurrency } from "../../data/benchmark-data";

type Props = {
  value: Concurrency;
  onChange: (value: Concurrency) => void;
};

export default function ConcurrencySelector({ value, onChange }: Props) {
  return (
    <div className="flex flex-wrap gap-2" role="radiogroup" aria-label="Concurrency">
      {[1, 2, 4].map((item) => (
        <button
          key={item}
          type="button"
          role="radio"
          aria-checked={value === item}
          onClick={() => onChange(item as Concurrency)}
          className={`rounded-full border px-4 py-2 text-sm ${
            value === item
              ? "border-cyan-300/30 bg-cyan-400/[0.08] text-cyan-50"
              : "border-white/15 bg-white/[0.04] text-white/70"
          }`}
        >
          c{item}
        </button>
      ))}
    </div>
  );
}
