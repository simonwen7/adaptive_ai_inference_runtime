import { githubUrl } from "../../data/architecture";

const links = [
  ["flow", "Flow"],
  ["batching", "Batching"],
  ["reliability", "Reliability"],
  ["evidence", "Evidence"],
] as const;

export default function Header() {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#05070b]/85 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between gap-3 px-6 py-3 sm:px-8">
        <a href="#top" className="text-xs font-semibold uppercase tracking-[0.3em] text-white/85">
          AIR
        </a>
        <nav aria-label="Primary" className="flex flex-wrap items-center justify-end gap-2">
          {links.map(([id, label]) => (
            <a
              key={id}
              href={`#${id}`}
              className="rounded-full border border-white/10 px-3 py-1.5 text-xs text-white/70 hover:bg-white/10 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-200/80"
            >
              {label}
            </a>
          ))}
          <a
            href={githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-full border border-cyan-300/20 bg-cyan-400/10 px-3 py-1.5 text-xs text-cyan-50 hover:bg-cyan-400/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-cyan-200/80"
          >
            GitHub ↗
          </a>
        </nav>
      </div>
    </header>
  );
}
