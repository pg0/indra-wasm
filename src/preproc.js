// Ported from indra-ocr 0.3.1 det_post.py and rec_preproc.py (Apache-2.0, Datalab); changed: rewritten in JavaScript.
// Detector input and output decoding (det_post.py) and the recognizer's line
// crops and batching (rec_preproc.py).
import { resize, pyRound, connectedComponents } from "./imgproc.js";

const CORE_X_SHRINK = 0.15;
const OUT_STRIDE = 2;
const INPUT_MULTIPLE = 32;

export const LINE_HEIGHT = 32;
export const PAD_Y = 0.2;
export const PAD_X = 0.3;
const WIDTH_MULTIPLE = 8;

/** Page -> [H, W] ink map padded to a multiple of 32, and the scale. */
export function pageToInput(gray, longSide) {
  const { width: w, height: h } = gray;
  const scale = longSide / Math.max(h, w);
  const nh = Math.max(1, pyRound(h * scale));
  const nw = Math.max(1, pyRound(w * scale));
  let g = gray;
  if (nh !== h || nw !== w) g = resize(gray, nw, nh, scale < 1 ? "area" : "linear");
  const ph = Math.ceil(nh / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const pw = Math.ceil(nw / INPUT_MULTIPLE) * INPUT_MULTIPLE;
  const out = new Float32Array(ph * pw);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) out[y * pw + x] = 1 - g.data[y * nw + x] / 255;
  }
  return { data: out, width: pw, height: ph, scale };
}

/**
 * Detector output [3, h, w] (core logit, top dist, bottom dist) -> boxes
 * [x0, y0, x1, y1, score] in detector input pixels.
 */
export function decode(out, w, h) {
  const n = w * h;
  const f = Math.fround;
  const prob = new Float32Array(n);
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    prob[i] = f(1 / f(1 + f(Math.exp(-out[i]))));
    mask[i] = prob[i] > 0.5 ? 1 : 0;
  }
  const cc = connectedComponents(mask, w, h);
  if (cc.n <= 1) return [];
  const cnt = new Float64Array(cc.n), y0 = new Float64Array(cc.n);
  const y1 = new Float64Array(cc.n), sc = new Float64Array(cc.n);
  for (let i = 0; i < n; i++) {
    const id = cc.labels[i];
    if (!id) continue;
    const ys = (Math.floor(i / w) + 0.5) * OUT_STRIDE;
    cnt[id] += 1;
    y0[id] += f(ys - out[n + i]);
    y1[id] += f(ys + out[2 * n + i]);
    sc[id] += prob[i];
  }
  const boxes = [];
  for (let i = 1; i < cc.n; i++) {
    if (cc.area[i] < 2) continue;
    const c = Math.max(cnt[i], 1);
    const top = y0[i] / c, bot = y1[i] / c;
    const lh = Math.max(1, bot - top);
    const ex = Math.min(CORE_X_SHRINK * lh, 0.5 * lh);
    const x0 = cc.left[i] * OUT_STRIDE - ex;
    const x1 = (cc.left[i] + cc.width[i]) * OUT_STRIDE + ex;
    boxes.push([f(x0), f(top), f(x1), f(bot), f(sc[i] / c)]);
  }
  return boxes;
}

/** One line crop from the page: an ink map [height, W] (Float32Array). */
export function cropLine(gray, box, height = LINE_HEIGHT, maxWidth = 4096) {
  const [x0, y0, x1, y1] = box;
  const h = Math.max(1, y1 - y0);
  const X0 = x0 - PAD_X * h, X1 = x1 + PAD_X * h;
  const Y0 = y0 - PAD_Y * h, Y1 = y1 + PAD_Y * h;
  const { width: W, height: H, data } = gray;
  const ix0 = Math.floor(X0), iy0 = Math.floor(Y0);
  const ix1 = Math.ceil(X1), iy1 = Math.ceil(Y1);
  const cw = Math.max(1, ix1 - ix0), ch = Math.max(1, iy1 - iy0);
  const crop = new Uint8Array(cw * ch).fill(255);
  const sx0 = Math.max(0, ix0), sy0 = Math.max(0, iy0);
  const sx1 = Math.min(W, ix1), sy1 = Math.min(H, iy1);
  if (sx1 > sx0 && sy1 > sy0) {
    for (let y = sy0; y < sy1; y++) {
      crop.set(data.subarray(y * W + sx0, y * W + sx1), (y - iy0) * cw + (sx0 - ix0));
    }
  }
  const outW = Math.max(WIDTH_MULTIPLE, Math.min(maxWidth, pyRound((cw * height) / ch)));
  const r = resize({ data: crop, width: cw, height: ch }, outW, height, ch > height ? "area" : "linear");
  const ink = new Float32Array(outW * height);
  for (let i = 0; i < ink.length; i++) ink[i] = 1 - r.data[i] / 255;
  return { data: ink, width: outW, height, cropWidth: cw };
}

/** Right-pad ink maps to one width: { data [B,1,H,W], dims, widths }. */
export function padBatch(lines, multiple = WIDTH_MULTIPLE) {
  const H = lines[0].height;
  let Wm = Math.max(...lines.map((l) => l.width));
  Wm = Math.ceil(Wm / multiple) * multiple;
  const data = new Float32Array(lines.length * H * Wm);
  lines.forEach((l, i) => {
    for (let y = 0; y < H; y++) {
      data.set(l.data.subarray(y * l.width, (y + 1) * l.width), i * H * Wm + y * Wm);
    }
  });
  return { data, dims: [lines.length, 1, H, Wm], widths: lines.map((l) => l.width) };
}

/** Line indices in batches of similar width (narrowest first). */
export function widthBatches(widths, maxBatch = 64, maxRatio = 1.3, minWidth = 64, maxArea = 64 * 640) {
  const order = widths.map((_, i) => i).sort((a, b) => widths[a] - widths[b]);
  const batches = [];
  for (const i of order) {
    const last = batches[batches.length - 1];
    const wi = Math.max(widths[i], minWidth);
    if (
      last &&
      last.length < maxBatch &&
      wi <= maxRatio * Math.max(widths[last[0]], minWidth) &&
      (last.length + 1) * wi <= maxArea
    ) {
      last.push(i);
    } else {
      batches.push([i]);
    }
  }
  return batches;
}
