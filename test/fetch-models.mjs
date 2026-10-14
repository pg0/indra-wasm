// Download the model files the tests use into test/models (not committed).
import { mkdir, writeFile, access } from "node:fs/promises";
import { MODEL_FILES, modelUrl } from "../src/models.js";

const dir = new URL("./models/", import.meta.url);
await mkdir(dir, { recursive: true });
for (const name of MODEL_FILES) {
  const path = new URL(name, dir);
  try { await access(path); continue; } catch {}
  const res = await fetch(modelUrl(name));
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
  await writeFile(path, new Uint8Array(await res.arrayBuffer()));
  console.log("fetched", name);
}
