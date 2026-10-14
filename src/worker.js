// Web worker: loads the models once and OCRs grayscale pages posted to it,
// so the page stays responsive while ONNX Runtime runs.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { IndraOCR } from "./pipeline.js";
import { cachedLoader } from "./models.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
// Threads need a cross-origin isolated page; elsewhere ONNX Runtime runs single-threaded.
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

let ready = null;

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === "init") {
      const load = cachedLoader({ base: msg.modelBase, onProgress: (p) => self.postMessage({ type: "progress", ...p }) });
      ready ??= IndraOCR.create(ort, load, { furniture: !!msg.furniture });
      await ready;
      self.postMessage({ type: "ready", id: msg.id });
    } else if (msg.type === "page") {
      const ocr = await ready;
      const page = await ocr.page({ data: msg.data, width: msg.width, height: msg.height }, { turn: msg.turn ?? true });
      self.postMessage({ type: "page", id: msg.id, page: { ...page, text: page.text } });
    }
  } catch (e) {
    self.postMessage({ type: "error", id: msg.id, error: String(e?.message ?? e) });
  }
};
