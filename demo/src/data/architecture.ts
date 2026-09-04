export const githubUrl = "https://github.com/simonwen7/adaptive_ai_inference_runtime";

export const flowStages = [
  {
    zone: "Ingress",
    id: "http",
    label: "HTTP",
    mechanism: "Receives POST /v1/infer and non-infer service endpoints.",
    why: "Provides a simple API boundary for clients and observability.",
  },
  {
    zone: "Ingress",
    id: "admission",
    label: "Admission",
    mechanism: "Attempts bounded queue admission and rejects overflow with QueueFull.",
    why: "Prevents unbounded backlog growth under overload.",
  },
  {
    zone: "Control Plane",
    id: "scheduler",
    label: "Scheduler",
    mechanism: "Orders queued requests with policy (FIFO or workload-aware).",
    why: "Controls fairness, starvation risk, and batching opportunities.",
  },
  {
    zone: "Control Plane",
    id: "router",
    label: "Router",
    mechanism: "Chooses the best worker lane using load and model locality signals.",
    why: "Balances throughput against model residency costs.",
  },
  {
    zone: "Execution",
    id: "worker",
    label: "Worker",
    mechanism: "Executes serially on a dedicated worker thread and lane.",
    why: "Provides isolation and predictable execution ownership.",
  },
  {
    zone: "Execution",
    id: "batch-builder",
    label: "BatchBuilder",
    mechanism: "Forms same-model static batches with bounded wait and size.",
    why: "Trades latency for serving efficiency when concurrency supports batching.",
  },
  {
    zone: "Execution",
    id: "model-manager",
    label: "ModelManager",
    mechanism: "Maintains model residency and eviction under memory constraints.",
    why: "Keeps hot models resident while respecting finite memory budgets.",
  },
  {
    zone: "Backend",
    id: "imodel-backend",
    label: "IModelBackend",
    mechanism: "Defines a stable execution interface used by workers.",
    why: "Separates runtime control-plane policy from model implementation details.",
  },
  {
    zone: "Backend",
    id: "llama-backend",
    label: "LlamaCppBackend",
    mechanism: "Implements backend execution through pinned llama.cpp APIs.",
    why: "Enables real GGUF inference while preserving runtime architecture boundaries.",
  },
  {
    zone: "Backend",
    id: "llama",
    label: "llama.cpp",
    mechanism: "Performs real model compute, context handling, and decode steps.",
    why: "Provides production-grade model execution primitives.",
  },
  {
    zone: "Backend",
    id: "metal",
    label: "Metal",
    mechanism: "Uses Apple Metal offload when available; CPU fallback remains possible.",
    why: "Matches measured M6 evidence environment on Apple Silicon.",
  },
] as const;

export const systemsCards = [
  ["Dynamic batching", "BatchBuilder forms bounded same-model batches.", "Improves efficiency when enough parallel work is present."],
  ["Workload-aware scheduling", "Scheduler can prioritize based on workload and waiting state.", "Makes queue behavior tunable instead of fixed-order only."],
  ["Residency-aware routing", "Router considers worker load and model locality.", "Reduces avoidable reload churn under multi-model traffic."],
  ["Cost-aware eviction", "Model residency uses estimate-based eviction policies.", "Maintains service under memory pressure with explicit trade-offs."],
  ["Bounded backpressure", "Admission uses finite queue capacities.", "Overload is rejected early with HTTP 503 instead of stalling globally."],
  ["Cancellation & deadlines", "Request lifecycle is first-terminal-wins with cooperative cancellation.", "Prevents stale work from dominating capacity."],
  ["Multi-worker runtime", "Runtime routes across worker lanes with isolated execution threads.", "Supports concurrency while preserving per-worker ownership."],
  ["HTTP serving", "Boost.Beast/Asio serves infer, health, metrics, and runtime endpoints.", "Keeps control-plane and serving behavior observable end-to-end."],
] as const;
