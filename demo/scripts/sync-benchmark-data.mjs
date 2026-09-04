import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const demoDir = path.resolve(__dirname, "..");
const sourcePath = path.resolve(demoDir, "../benchmarks/results/m6_final_summary.json");
const outputDir = path.resolve(demoDir, "src/data/generated");
const outputPath = path.resolve(outputDir, "m6-final-summary.json");

const EXPECTED = {
  schemaVersion: 1,
  projectSha: "dfc00d73033af62f1888bac014d5586086c5b30d",
  modelSha: "57d1997790d1744fba5b40a7317df71ea5e2acee28c47e78f0cce39c0703f8cf",
  llamaSha: "9cc33944f9b7a44243618d5522adae357d7fdc90",
};

function assert(condition, message) {
  if (!condition) {
    throw new Error(message);
  }
}

function normalize(raw) {
  return {
    schema_version: raw.schema_version,
    source: raw.source,
    model: raw.model,
    hardware: raw.hardware,
    methodology: raw.methodology,
    batching: raw.batching,
    comparisons: raw.comparisons,
    overload: raw.overload,
    caveats: raw.caveats,
  };
}

async function main() {
  const payload = JSON.parse(await readFile(sourcePath, "utf8"));
  assert(payload.schema_version === EXPECTED.schemaVersion, "Unexpected schema_version");
  assert(payload.source?.project_git_sha === EXPECTED.projectSha, "Unexpected benchmark source SHA");
  assert(payload.model?.sha256 === EXPECTED.modelSha, "Unexpected model SHA");
  assert(payload.source?.llama_git_sha === EXPECTED.llamaSha, "Unexpected llama SHA");

  const normalized = normalize(payload);
  await mkdir(outputDir, { recursive: true });
  await writeFile(outputPath, `${JSON.stringify(normalized, null, 2)}\n`, "utf8");

  console.log(`SYNC_OK ${path.relative(demoDir, outputPath)}`);
}

main().catch((error) => {
  console.error(`SYNC_FAIL ${error.message}`);
  process.exit(1);
});
