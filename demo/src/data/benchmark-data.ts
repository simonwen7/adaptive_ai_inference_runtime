import raw from "./generated/m6-final-summary.json";

export type Mode = "baseline" | "adaptive";
export type Concurrency = 1 | 2 | 4;

export const benchmarkData = raw;

export function getCell(mode: Mode, concurrency: Concurrency) {
  const cell = benchmarkData.batching.cells.find(
    (item) => item.mode === mode && item.concurrency === concurrency,
  );
  if (!cell) {
    throw new Error(`Missing benchmark cell ${mode}/c${concurrency}`);
  }
  return cell;
}

export function getComparison(concurrency: Concurrency) {
  const cmp = benchmarkData.comparisons.cells.find((item) => item.concurrency === concurrency);
  if (!cmp) {
    throw new Error(`Missing comparison for c${concurrency}`);
  }
  return cmp;
}
