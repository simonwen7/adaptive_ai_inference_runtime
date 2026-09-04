import { benchmarkData } from "../../data/benchmark-data";

const stack = [
  "IModelBackend",
  "LlamaCppBackend",
  "pinned llama.cpp",
  "Qwen3.5-0.8B-Q4_0 GGUF",
  "Metal / CPU",
];

export default function LlamaBackendStack() {
  return (
    <div className="rounded-[24px] border border-white/12 bg-white/[0.03] p-6">
      <div className="mx-auto flex max-w-sm flex-col gap-2">
        {stack.map((item, idx) => (
          <div key={item} className="text-center">
            {idx > 0 ? <p className="pb-2 text-white/30">↓</p> : null}
            <p className="rounded-2xl border border-white/12 bg-black/25 px-4 py-3 text-sm text-white/80">{item}</p>
          </div>
        ))}
      </div>
      <p className="mt-5 text-sm text-white/65">
        Real static multi-sequence infer_batch, greedy decoding, and context management run
        through llama.cpp ({benchmarkData.source.llama_git_sha.slice(0, 12)}...) with Apple
        Metal offload and CPU fallback via --gpu-layers 0.
      </p>
    </div>
  );
}
