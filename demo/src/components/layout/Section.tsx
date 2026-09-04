import type { ReactNode } from "react";

type Props = {
  id: string;
  eyebrow?: string;
  title: string;
  subtitle?: string;
  children: ReactNode;
};

export default function Section({ id, eyebrow, title, subtitle, children }: Props) {
  return (
    <section id={id} className="mx-auto w-full max-w-6xl scroll-mt-28 px-6 py-14 sm:px-8">
      {eyebrow ? (
        <p className="text-[11px] font-medium uppercase tracking-[0.28em] text-cyan-100/55">
          {eyebrow}
        </p>
      ) : null}
      <h2 className="mt-3 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-4xl">{title}</h2>
      {subtitle ? <p className="mt-4 max-w-3xl text-white/60">{subtitle}</p> : null}
      <div className="mt-8">{children}</div>
    </section>
  );
}
