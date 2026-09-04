type Props = {
  label: string;
  value: string;
  hint?: string;
};

export default function MetricPanel({ label, value, hint }: Props) {
  return (
    <div className="rounded-2xl border border-white/12 bg-black/20 p-4">
      <p className="text-[10px] uppercase tracking-[0.24em] text-white/40">{label}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      {hint ? <p className="mt-2 text-xs text-white/45">{hint}</p> : null}
    </div>
  );
}
