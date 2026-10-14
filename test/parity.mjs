// Parity test: the JS pipeline against the Python indra-ocr reference outputs
// (tools/make_reference.py) on the same grayscale pages.
// Run: npm run models && npm test   (fixtures in test/fixtures, not committed)
import { readFile, readdir } from "node:fs/promises";
import * as ort from "onnxruntime-web";
import { IndraOCR } from "../src/pipeline.js";

const MAX_CER = 0.01; // markdown character error rate per page
const MAX_BOX_PX = 2; // line box corners, pixels
const fixtures = new URL("./fixtures/", import.meta.url);
const models = new URL("./models/", import.meta.url);

function readPgm(buf) {
  let pos = 0;
  const fields = [];
  while (fields.length < 4) {
    while (/\s/.test(String.fromCharCode(buf[pos]))) pos++;
    let s = "";
    while (!/\s/.test(String.fromCharCode(buf[pos]))) s += String.fromCharCode(buf[pos++]);
    fields.push(s);
  }
  pos++;
  const [, w, h] = fields.map(Number);
  return { data: buf.subarray(pos, pos + w * h), width: w, height: h };
}

function levenshtein(a, b) {
  a = [...a]; b = [...b];
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}

const load = async (name) => {
  const buf = await readFile(new URL(name, models));
  return name.endsWith(".json") ? buf.toString("utf8") : new Uint8Array(buf);
};
const ocr = await IndraOCR.create(ort, load);
const only = process.argv[2];
const names = (await readdir(fixtures)).filter((f) => f.endsWith(".json")).map((f) => f.slice(0, -5))
  .filter((n) => !only || n.includes(only)).sort();

let failed = 0;
for (const name of names) {
  const ref = JSON.parse(await readFile(new URL(name + ".json", fixtures), "utf8"));
  const gray = readPgm(await readFile(new URL(name + ".pgm", fixtures)));
  const t0 = performance.now();
  const page = await ocr.page(gray);
  const ms = performance.now() - t0;
  const exp = ref.markdown ?? "", got = page.markdown ?? "";
  const cer = levenshtein(exp, got) / Math.max(exp.length, 1);
  const sameLines = page.lines.filter((l, i) => ref.lines[i]?.text === l.text).length;
  let boxPx = 0;
  page.lines.forEach((l, i) => {
    if (ref.lines[i]) l.box.forEach((v, k) => { boxPx = Math.max(boxPx, Math.abs(v - ref.lines[i].box[k])); });
  });
  const problems = [];
  if (page.rotation !== ref.rotation) problems.push(`rotation ${page.rotation} != ${ref.rotation}`);
  if (Math.abs(page.skew - ref.skew) > 0.11) problems.push(`skew ${page.skew} != ${ref.skew}`);
  if (cer > MAX_CER) problems.push(`markdown CER ${(cer * 100).toFixed(2)}%`);
  if (page.lines.length !== ref.lines.length) problems.push("line count differs");
  if (boxPx > MAX_BOX_PX) problems.push(`box off by ${boxPx.toFixed(1)} px`);
  console.log(
    `${problems.length ? "FAIL" : "ok  "} ${name.padEnd(16)} lines ${page.lines.length}/${ref.lines.length}` +
    ` same ${sameLines}  box ${boxPx.toFixed(1)}px  CER ${(cer * 100).toFixed(2)}%  ${Math.round(ms)} ms  ${problems.join("; ")}`,
  );
  if (problems.length) {
    failed++;
    if (process.env.VERBOSE) console.log("--- expected\n" + exp + "\n--- got\n" + got);
  }
}
console.log(failed ? `${failed} of ${names.length} pages failed` : `all ${names.length} pages pass`);
process.exit(failed ? 1 : 0);
