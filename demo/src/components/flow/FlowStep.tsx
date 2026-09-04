type Props = {
  active: boolean;
  label: string;
  zone: string;
  onClick: () => void;
};

export default function FlowStep({ active, label, zone, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full rounded-2xl border px-4 py-3 text-left transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-200/80 ${
        active
          ? "border-cyan-300/30 bg-cyan-400/[0.08] text-cyan-50"
          : "border-white/12 bg-white/[0.03] text-white/80 hover:bg-white/[0.06]"
      }`}
    >
      <p className="text-[10px] uppercase tracking-[0.24em] text-white/45">{zone}</p>
      <p className="mt-1 text-sm font-medium">{label}</p>
    </button>
  );
}
