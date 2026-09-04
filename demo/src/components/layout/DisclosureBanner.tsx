export default function DisclosureBanner() {
  return (
    <div className="mx-auto w-full max-w-6xl px-6 pt-6 sm:px-8">
      <p className="rounded-2xl border border-cyan-300/20 bg-cyan-400/[0.06] px-4 py-3 text-sm text-cyan-50/85">
        This is an interactive systems visualization based on measured benchmark evidence.
        It does not execute C++/llama.cpp inference in the browser or on a remote server.
      </p>
    </div>
  );
}
