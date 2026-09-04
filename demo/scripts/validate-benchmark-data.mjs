import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoDir = path.resolve(__dirname, "..");
const generatedPath = path.resolve(demoDir, "src/data/generated/m6-final-summary.json");

const REQUIRED = {
  evidenceSha: "dfc00d73033af62f1888bac014d5586086c5b30d",
  modelSha: "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
  llamaSha: "9cc33944f9b7a44243618d5522adae357d7fdc90",
  c4RpsDelta: 101.0,
  c4P95Delta: -26.1,
};

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function approx(actual, expected, tolerance = 0.01) {
  return Math.abs(actual - expected) <= tolerance;
}

async function main() {
  const data = JSON.parse(await readFile(generatedPath, "utf8"));
  assert(data.source.project_git_sha === REQUIRED.evidenceSha, "Evidence SHA mismatch");
  assert(data.model.sha256 === REQUIRED.modelSha, "Model SHA mismatch");
  assert(data.source.llama_git_sha === REQUIRED.llamaSha, "llama SHA mismatch");

  const byCell = new Map(
    data.batching.cells.map((cell) => [`${cell.mode}/c${cell.concurrency}`, cell]),
  );
  const c4Base = byCell.get("baseline/c4");
  const c4Adaptive = byCell.get("adaptive/c4");
  const c2Adaptive = byCell.get("adaptive/c2");
  assert(c4Base && c4Adaptive && c2Adaptive, "Missing required batching cells");
  assert(approx(c4Base.median.rps, 2.17), "Unexpected c4 baseline rps");
  assert(approx(c4Adaptive.median.rps, 4.35), "Unexpected c4 adaptive rps");
  assert(approx(c2Adaptive.median.mean_batch_size, 2.0), "Unexpected c2 mean batch");
  assert(approx(c2Adaptive.median.multi_request_batch_rate, 1.0), "Unexpected c2 multi-request rate");
  assert(approx(c4Adaptive.median.mean_batch_size, 4.0), "Unexpected c4 mean batch");
  assert(approx(c4Adaptive.median.multi_request_batch_rate, 1.0), "Unexpected c4 multi-request rate");

  const c4Cmp = data.comparisons.cells.find((cell) => cell.concurrency === 4);
  assert(c4Cmp, "Missing c4 comparison cell");
  assert(approx(c4Cmp.delta_rps_pct, REQUIRED.c4RpsDelta), "Unexpected c4 delta_rps_pct");
  assert(approx(c4Cmp.delta_p95_pct, REQUIRED.c4P95Delta), "Unexpected c4 delta_p95_pct");

  assert(data.overload.runs.length === 3, "Overload run count mismatch");
  data.overload.runs.forEach((run, index) => {
    assert(run.http_200 === 3, `Overload run ${index} http_200 mismatch`);
    assert(run.http_503 === 61, `Overload run ${index} http_503 mismatch`);
    assert(run.transport_errors === 0, `Overload run ${index} transport mismatch`);
  });

  console.log("CHECK_OK benchmark contract valid");
}

main().catch((error) => {
  console.error(`CHECK_FAIL ${error.message}`);
  process.exit(1);
});
