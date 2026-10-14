// Model files come straight from the Hugging Face repo (they are not part of
// this project; see the README for their license). The browser keeps them in
// the Cache API, so they download once.
export const HF_REPO = "datalab-to/indra-ocr";
export const HF_REVISION = "v3.1"; // the model set indra-ocr 0.3.1 was tested against
export const MODEL_FILES = [
  "meta.json", "det.onnx", "rec.onnx", "order.onnx", "orient.onnx", "script.onnx", "rec_cjk.onnx",
];

export const modelUrl = (name, base = `https://huggingface.co/${HF_REPO}/resolve/${HF_REVISION}/`) => base + name;

/** A loader for IndraOCR.create: cached fetch, bytes for .onnx, text for .json. */
export function cachedLoader({ base, cacheName = `indra-ocr-${HF_REVISION}`, onProgress } = {}) {
  return async (name) => {
    const url = modelUrl(name, base);
    let res;
    const cache = typeof caches !== "undefined" ? await caches.open(cacheName).catch(() => null) : null;
    if (cache) res = await cache.match(url);
    if (!res) {
      onProgress?.({ name, state: "download" });
      res = await fetch(url);
      if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
      if (cache) await cache.put(url, res.clone()).catch(() => {});
    }
    onProgress?.({ name, state: "ready" });
    return name.endsWith(".json") ? res.text() : new Uint8Array(await res.arrayBuffer());
  };
}
