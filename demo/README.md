# Adaptive AI Inference Runtime Showcase

Static interactive systems showcase for Project 3, built from the committed runtime architecture and benchmark evidence.

## What this is

- A static frontend experience that explains request flow, scheduling, batching, backend boundaries, and overload behavior.
- A data-backed benchmark lab sourced from committed evidence.
- An architecture visualization and evidence presentation layer for recruiter and technical review.

## What this is not

- Not live C++/llama.cpp inference in the browser.
- Not remote backend execution.
- Not live GPU telemetry.
- Not a replacement for core runtime/docs.

## Evidence source of truth

Authoritative tracked source:

- `../benchmarks/results/m6_final_summary.json`

Build-time generated frontend artifact:

- `src/data/generated/m6-final-summary.json` (generated, gitignored)

## Data sync and validation

`npm run sync:data`:

1. Reads `../benchmarks/results/m6_final_summary.json`
2. Validates critical identity fields (schema, evidence SHA, model SHA, llama SHA)
3. Writes normalized frontend-safe JSON into `src/data/generated/`

`npm run check:data`:

- Verifies key benchmark contract values (c4 deltas, c2/c4 batch utilization, overload run counts)
- Fails fast if evidence contract drifts unexpectedly

## Install

```bash
cd demo
npm ci
```

## Build

```bash
npm run build
```

Build pipeline:

1. `sync:data` — reads repository-tracked canonical evidence from `../benchmarks/results/m6_final_summary.json`
2. `check:data`
3. TypeScript lint (`tsc --noEmit`)
4. Vite production build → `dist`

Generated frontend-safe JSON under `src/data/generated/` is produced at build time and is gitignored. After a successful build, the static site has no runtime dependency on the parent evidence file.

## Optional local development (user-run only)

```bash
cd demo
npm ci
npm run dev
```

This repository task does not start dev servers automatically.

## Vercel deployment plan

Deploy from the GitHub repository `adaptive_ai_inference_runtime` with these settings:

| Setting | Value |
|---------|-------|
| Repository | `adaptive_ai_inference_runtime` |
| Root Directory | `demo` |
| Framework Preset | Vite |
| Build Command | `npm run build` |
| Output Directory | `dist` |
| Include source files outside of the Root Directory in the Build Step | **ENABLED** |

### Why outside-root inclusion is required

The showcase build intentionally reads the repository-tracked canonical benchmark evidence from:

```text
../benchmarks/results/m6_final_summary.json
```

No benchmark data is duplicated inside the showcase source. With Root Directory set to `demo`, Vercel must be allowed to include source files outside that directory during the build step so `sync:data` can reach the parent evidence file.

### Dashboard steps

1. Import the GitHub repository `adaptive_ai_inference_runtime`.
2. Set Root Directory to `demo`.
3. In the Root Directory / build settings, enable:
   **Include source files outside of the Root Directory in the Build Step**.
4. Framework Preset: Vite.
5. Build Command: `npm run build`.
6. Output Directory: `dist`.
7. Deploy.

### Fail-closed behavior

If outside-root source inclusion is disabled, `npm run build` on Vercel will fail during `sync:data` because the canonical benchmark JSON is outside `demo/`.

This is intentional. The build must not silently use stale generated artifacts, hard-coded values, or remote fetches when canonical evidence is unavailable.
