// Browser API: PDFs and images in, markdown out. OCR runs in a web worker;
// PDF pages are rendered here with pdf.js.
//
//   import { Indra } from "./src/indra.js";
//   const indra = new Indra({ onProgress: console.log });
//   const { markdown, pages } = await indra.recognize(file);

const PDFJS = "https://cdn.jsdelivr.net/npm/pdfjs-dist@6.3.289/build/";
const DPI = 200;
const MAX_PAGE_PIXELS = 100e6;
const MAX_CANVAS_SIDE = 16384;

let pdfjsLib = null;
async function pdfjs() {
  if (!pdfjsLib) {
    pdfjsLib = await import(PDFJS + "pdf.min.mjs");
    pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS + "pdf.worker.min.mjs";
  }
  return pdfjsLib;
}

/** RGBA pixels on white -> grayscale with OpenCV's RGB2GRAY weights. */
function toGray(rgba, width, height) {
  const out = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < out.length; i++, p += 4) {
    const a = rgba[p + 3], ia = 255 - a;
    const r = (rgba[p] * a + 255 * ia) / 255, g = (rgba[p + 1] * a + 255 * ia) / 255, b = (rgba[p + 2] * a + 255 * ia) / 255;
    out[i] = (r * 4899 + g * 9617 + b * 1868 + 8192) >> 14;
  }
  return { data: out, width, height };
}

function canvas(w, h) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(w, h);
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  return c;
}

async function imageGray(blob) {
  const bmp = await createImageBitmap(blob);
  const c = canvas(bmp.width, bmp.height);
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return { gray: toGray(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height), image: c };
}

async function* pdfPages(bytes, password) {
  const lib = await pdfjs();
  const task = lib.getDocument({ data: bytes, password });
  const doc = await task.promise;
  try {
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      const base = page.getViewport({ scale: 1 });
      let scale = Math.min(DPI / 72, Math.sqrt(MAX_PAGE_PIXELS / (base.width * base.height)));
      scale = Math.min(scale, MAX_CANVAS_SIDE / Math.max(base.width, base.height));
      const vp = page.getViewport({ scale });
      const c = canvas(Math.ceil(vp.width), Math.ceil(vp.height));
      const ctx = c.getContext("2d", { willReadFrequently: true });
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      await page.render({ canvasContext: ctx, canvas: c, viewport: vp }).promise;
      page.cleanup();
      const gray = toGray(ctx.getImageData(0, 0, c.width, c.height).data, c.width, c.height);
      yield { index: i - 1, count: doc.numPages, gray, image: c };
    }
  } finally {
    await task.destroy();
  }
}

const isPdf = (bytes) => new TextDecoder("latin1").decode(bytes.subarray(0, 1024)).includes("%PDF-");

export class Indra {
  /**
   * onProgress({ stage: "model" | "page", ... }). modelBase: a folder URL with
   * the model files (default: the Hugging Face repo). furniture: keep page
   * headers and footers in the markdown.
   */
  constructor({ onProgress, modelBase, furniture = false, workerUrl } = {}) {
    this.onProgress = onProgress;
    this.worker = new Worker(workerUrl ?? new URL("./worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.nextId = 1;
    this.worker.onmessage = ({ data }) => {
      if (data.type === "progress") return this.onProgress?.({ stage: "model", ...data });
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      data.type === "error" ? p.reject(new Error(data.error)) : p.resolve(data);
    };
    this.ready = this.call({ type: "init", modelBase, furniture });
  }

  call(msg, transfer = []) {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  /** OCR of one grayscale page { data: Uint8Array, width, height }. */
  async page(gray, { turn = true, tables = true } = {}) {
    await this.ready;
    const { page } = await this.call({ type: "page", data: gray.data, width: gray.width, height: gray.height, turn, tables }, [gray.data.buffer]);
    return page;
  }

  /**
   * A PDF or image (File, Blob or ArrayBuffer) -> { markdown, pages }.
   * tables: false writes tables as plain text rows instead of HTML.
   * onPage({ index, count, page, image }) runs after each page; image is the
   * page as rendered for OCR (a canvas), e.g. for a side-by-side preview.
   */
  async recognize(file, { password, maxPages, tables = true, onPage } = {}) {
    const bytes = new Uint8Array(file instanceof ArrayBuffer ? file : await file.arrayBuffer());
    const pages = [];
    if (isPdf(bytes)) {
      const it = pdfPages(bytes, password);
      try {
        let next = it.next();
        for (;;) {
          const { value, done } = await next;
          if (done || (maxPages && value.index >= maxPages)) break;
          // pdf.js renders the following page on this thread while the worker OCRs this one.
          next = it.next();
          next.catch(() => {});
          const { index, count, gray, image } = value;
          const total = maxPages ? Math.min(count, maxPages) : count;
          this.onProgress?.({ stage: "page", page: index + 1, pages: total });
          pages.push(await this.page(gray, { tables }));
          onPage?.({ index, count: total, page: pages[pages.length - 1], image });
        }
      } finally {
        await it.return();
      }
    } else {
      this.onProgress?.({ stage: "page", page: 1, pages: 1 });
      const { gray, image } = await imageGray(new Blob([bytes]));
      pages.push(await this.page(gray, { tables }));
      onPage?.({ index: 0, count: 1, page: pages[0], image });
    }
    const markdown = pages.map((p) => p.markdown ?? p.text).join("\n\n") + "\n";
    return { markdown, pages };
  }

  terminate() {
    this.worker.terminate();
  }
}
