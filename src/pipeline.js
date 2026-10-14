// Ported from indra-ocr 0.3.1 pipeline.py (Apache-2.0, Datalab); changed: rewritten in JavaScript.
// The OCR pipeline (pipeline.py) on ONNX Runtime Web: orientation, deskew,
// line detection, CTC recognition (Latin or CJK recognizer per line),
// line splitting, reading order, layout blocks and markdown.
import { resize, warpAffine, rotationMatrix, rotateClockwise, pyRound, median } from "./imgproc.js";
import { pageToInput, decode, cropLine, padBatch, widthBatches, PAD_X } from "./preproc.js";
import { lineFeatures, decodeOrder, decodeBlocksOrder, orderXYCut, N_FEATS } from "./order.js";
import { pageBlocks, pageMarkdown } from "./layout.js";
import { Vocab } from "./vocab.js";

const ADAPTIVE_MIN_H = 9.0;
const ADAPTIVE_TARGET_H = 12.0;
const ADAPTIVE_MAX_SIDE = 2048;
const SPLIT_GAP = 3.0;
const CELL_GAP = 1.8;
const GUTTER_GAP = 1.6;
const GUTTER_EDGES = 3;
const ORIENT_SIDE = 768;
const ORIENT_MIN = 0.95;
const MIN_CONF = 0.5;
const DESKEW_MAX = 5.0;
const DESKEW_MIN = 0.3;

const now = () => (typeof performance !== "undefined" ? performance.now() : Date.now());
const r1 = (v) => Math.round(v * 10) / 10;

function shrinkTo(gray, side) {
  const s = side / Math.max(gray.width, gray.height);
  return resize(gray, Math.max(1, pyRound(gray.width * s)), Math.max(1, pyRound(gray.height * s)), "area");
}

/** Degrees (counter-clockwise) that straighten the text rows of a page. */
export function estimateSkew(gray, side = 800) {
  const small = shrinkTo(gray, side);
  const { width: w, height: h } = small;
  const ink = new Float32Array(w * h);
  let sum = 0;
  for (let i = 0; i < ink.length; i++) { ink[i] = (255 - small.data[i]) / 255; sum += ink[i]; }
  if (sum / ink.length < 0.003) return 0;
  const img = { data: ink, width: w, height: h };
  const sharpness = (a) => {
    const r = warpAffine(img, rotationMatrix(w / 2, h / 2, a), "constant", 0).data;
    let prev = 0, acc = 0;
    for (let y = 0; y < h; y++) {
      let p = 0;
      for (let x = 0; x < w; x++) p += r[y * w + x];
      p = Math.fround(p);
      if (y) acc += (p - prev) * (p - prev);
      prev = p;
    }
    return acc;
  };
  const argmaxOver = (start, stop, step) => {
    let best = start, bestV = -Infinity;
    const n = Math.ceil((stop - start) / step);
    for (let i = 0; i < n; i++) {
      const a = start + i * step;
      const v = sharpness(a);
      if (v > bestV) { bestV = v; best = a; }
    }
    return best;
  };
  const best = argmaxOver(-DESKEW_MAX, DESKEW_MAX + 1e-6, 0.5);
  return argmaxOver(best - 0.5, best + 0.5 + 1e-6, 0.1);
}

function unrotateBox(box, degrees, w, h) {
  const M = rotationMatrix(w / 2, h / 2, -degrees);
  const [x0, y0, x1, y1] = box;
  const xs = [], ys = [];
  for (const [x, y] of [[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) {
    xs.push(M[0] * x + M[1] * y + M[2]);
    ys.push(M[3] * x + M[4] * y + M[5]);
  }
  const c = (v, hi) => Math.min(Math.max(v, 0), hi);
  return [r1(c(Math.min(...xs), w)), r1(c(Math.min(...ys), h)), r1(c(Math.max(...xs), w)), r1(c(Math.max(...ys), h))];
}

export class IndraOCR {
  /**
   * ort: the onnxruntime-web module. load(name): a Promise of the model file's
   * bytes (Uint8Array) or, for meta.json, its text. rec_cjk.onnx is loaded on
   * the first page with Chinese lines.
   */
  static async create(ort, load, { furniture = false, sessionOptions = {} } = {}) {
    const ocr = new IndraOCR();
    ocr.ort = ort;
    ocr.load = load;
    ocr.furniture = furniture;
    ocr.opts = { graphOptimizationLevel: "all", ...sessionOptions };
    const sess = async (name) => ort.InferenceSession.create(await load(name), ocr.opts);
    const meta = JSON.parse(await load("meta.json"));
    [ocr.det, ocr.rec, ocr.order, ocr.orient, ocr.script] = await Promise.all(
      ["det.onnx", "rec.onnx", "order.onnx", "orient.onnx", "script.onnx"].map(sess),
    );
    ocr.vocab = Vocab.fromJson(meta.vocab);
    ocr.vocabCjk = Vocab.fromJson(meta.vocab_cjk);
    ocr.longSide = meta.det_long_side;
    ocr.height = meta.rec_height;
    ocr.recCjk = null;
    return ocr;
  }

  async run(session, feeds) {
    const out = await session.run(feeds);
    return session.outputNames.map((n) => out[n]);
  }

  tensor(type, data, dims) {
    return new this.ort.Tensor(type, data, dims);
  }

  async cjkRecognizer() {
    if (!this.recCjk) this.recCjk = this.ort.InferenceSession.create(await this.load("rec_cjk.onnx"), this.opts);
    return this.recCjk;
  }

  async orientation(gray) {
    const small = shrinkTo(gray, ORIENT_SIDE);
    const x = new Float32Array(ORIENT_SIDE * ORIENT_SIDE);
    for (let y = 0; y < small.height; y++) {
      for (let i = 0; i < small.width; i++) x[y * ORIENT_SIDE + i] = 1 - small.data[y * small.width + i] / 255;
    }
    const [p] = await this.run(this.orient, { x: this.tensor("float32", x, [1, 1, ORIENT_SIDE, ORIENT_SIDE]) });
    let k = 0;
    for (let i = 1; i < 4; i++) if (p.data[i] > p.data[k]) k = i;
    return p.data[k] > ORIENT_MIN ? k : 0;
  }

  /** The page turned upright and straightened: { gray, k, skew }. */
  async upright(gray, turn = true) {
    const k = turn ? await this.orientation(gray) : 0;
    if (k) gray = rotateClockwise(gray, k);
    let a = estimateSkew(gray);
    if (Math.abs(a) >= DESKEW_MIN) {
      gray = warpAffine(gray, rotationMatrix(gray.width / 2, gray.height / 2, a), "replicate");
    } else {
      a = 0;
    }
    return { gray, k, skew: a };
  }

  async detectAt(gray, longSide) {
    const inp = pageToInput(gray, longSide);
    const [out] = await this.run(this.det, { x: this.tensor("float32", inp.data, [1, 1, inp.height, inp.width]) });
    const [, , h, w] = out.dims;
    return { boxes: decode(out.data, w, h), scale: inp.scale };
  }

  async detect(gray) {
    let { boxes, scale } = await this.detectAt(gray, this.longSide);
    if (boxes.length >= 8) {
      const medH = median(boxes.map((b) => Math.fround(b[3] - b[1])));
      if (medH < ADAPTIVE_MIN_H) {
        const ls = Math.floor(Math.trunc(Math.min(ADAPTIVE_MAX_SIDE, (this.longSide * ADAPTIVE_TARGET_H) / Math.max(medH, 1))) / 32) * 32;
        if (ls > this.longSide) ({ boxes, scale } = await this.detectAt(gray, ls));
      }
    }
    const s = Math.fround(scale);
    const { width: w, height: h } = gray;
    return boxes.map((b) => {
      const v = b.slice(0, 4).map((x) => Math.fround(x / s));
      return [
        Math.min(Math.max(v[0], 0), w), Math.min(Math.max(v[1], 0), h),
        Math.min(Math.max(v[2], 0), w), Math.min(Math.max(v[3], 0), h), b[4],
      ];
    });
  }

  async batchFeeds(crops) {
    const { data, dims, widths } = padBatch(crops);
    return {
      feeds: {
        x: this.tensor("float32", data, dims),
        widths: this.tensor("int64", BigInt64Array.from(widths.map(BigInt)), [widths.length]),
      },
      width: dims[3],
    };
  }

  async hanziLines(crops) {
    const out = new Array(crops.length).fill(false);
    for (const idx of widthBatches(crops.map((c) => c.width))) {
      const { feeds } = await this.batchFeeds(idx.map((j) => crops[j]));
      const [p] = await this.run(this.script, feeds);
      idx.forEach((j, k) => { out[j] = p.data[k] > 0.5; });
    }
    return out;
  }

  /** [text, conf, tokens] per line; tokens are [id or char, page x]. */
  async recognize(gray, boxes) {
    const crops = boxes.map((b) => cropLine(gray, b, this.height));
    const out = crops.map(() => ["", 0, []]);
    if (!crops.length) return { rec: out, hanzi: [] };
    const hanzi = await this.hanziLines(crops);
    const groups = [
      [this.rec, this.vocab, crops.map((_, j) => j).filter((j) => !hanzi[j]), false],
      [null, this.vocabCjk, crops.map((_, j) => j).filter((j) => hanzi[j]), true],
    ];
    for (let [sess, vocab, sel, asChars] of groups) {
      if (!sel.length) continue;
      if (asChars) sess = await this.cjkRecognizer();
      for (const part of widthBatches(sel.map((j) => crops[j].width))) {
        const idx = part.map((i) => sel[i]);
        const { feeds, width } = await this.batchFeeds(idx.map((j) => crops[j]));
        const [best, pmax, lengths] = await this.run(sess, feeds);
        const T = best.dims[1];
        const stride = width / Math.max(T, 1);
        idx.forEach((j, k) => {
          const n = Number(lengths.data[k]);
          const keep = [];
          let prev = -1;
          for (let t = 0; t < n; t++) {
            const id = Number(best.data[k * T + t]);
            if (id !== 0 && (t === 0 || id !== prev)) keep.push(t);
            prev = id;
          }
          const [x0, y0, x1, y1] = boxes[j];
          const h = Math.max(1, y1 - y0);
          const ix0 = Math.floor(x0 - PAD_X * h);
          const cw = Math.ceil(x1 + PAD_X * h) - ix0;
          const f = cw / crops[j].width;
          const ids = keep.map((t) => Number(best.data[k * T + t]));
          const xs = keep.map((t) => ix0 + (t + 0.5) * stride * f);
          const toks = ids.map((id, q) => [asChars ? vocab.chars[id - 1] : id, xs[q]]);
          let conf = 0;
          if (keep.length) {
            let s = 0;
            for (const t of keep) s += pmax.data[k * T + t];
            conf = Math.fround(s / keep.length);
          }
          out[j] = conf >= MIN_CONF ? [vocab.decode(ids), conf, toks] : ["", conf, []];
        });
      }
    }
    return { rec: out, hanzi };
  }

  isSpace(tok) {
    return tok === " " || tok === this.vocab.index.get(" ");
  }

  decodeTokens(toks) {
    if (toks.length && typeof toks[0][0] === "string") {
      return toks.map((t) => t[0]).join("").split(/\s+/u).filter(Boolean).join(" ");
    }
    return this.vocab.decode(toks.map((t) => t[0]));
  }

  gutterCuts(boxes, rec) {
    return boxes.map((b, i) => {
      const h = Math.max(1, b[3] - b[1]);
      const xs = rec[i][2].filter((t) => !this.isSpace(t[0])).map((t) => t[1]);
      const mine = [];
      for (let q = 0; q + 1 < xs.length; q++) {
        const a = xs[q], c = xs[q + 1];
        if (c - a <= GUTTER_GAP * h) continue;
        const gl = a + 0.35 * h, gr = c - 0.35 * h;
        let near = 0;
        boxes.forEach((o, k) => {
          if (k !== i && (Math.abs(o[2] - gl) < 0.5 * h || Math.abs(o[0] - gr) < 0.5 * h)) near++;
        });
        if (near >= GUTTER_EDGES) mine.push((a + c) / 2);
      }
      return mine;
    });
  }

  /** Cut a line at gaps wider than gap line heights: [[box, text, tokens]]. */
  splitLine(box, toks, gap, at = []) {
    const h = Math.max(1, box[3] - box[1]);
    const segs = [[]];
    let prev = null;
    at = at.slice().sort((a, b) => a - b);
    for (const tk of toks) {
      const x = tk[1];
      if (!this.isSpace(tk[0])) {
        if (prev !== null && (x - prev > gap * h || at.some((c) => prev < c && c < x))) segs.push([]);
        prev = x;
      }
      segs[segs.length - 1].push(tk);
    }
    if (segs.length === 1) return [[box.slice(0, 4), this.decodeTokens(toks), toks]];
    const out = [];
    for (const sg of segs) {
      const text = this.decodeTokens(sg).replace(/^\s+|\s+$/gu, "");
      const xs = sg.filter((t) => !this.isSpace(t[0])).map((t) => t[1]);
      if (text && xs.length) {
        out.push([[Math.max(box[0], Math.min(...xs) - 0.35 * h), box[1], Math.min(box[2], Math.max(...xs) + 0.35 * h), box[3]], text, sg]);
      }
    }
    return out;
  }

  async readingOrder(boxes, texts, W, H) {
    const n = boxes.length;
    if (n < 2) return { order: n ? orderXYCut(boxes) : [], layout: null };
    const feats = lineFeatures(boxes, W, H, texts);
    const outs = await this.run(this.order, {
      feats: this.tensor("float32", feats, [1, n, N_FEATS]),
      mask: this.tensor("bool", new Uint8Array(n).fill(1), [1, n]),
    });
    const a = outs[0].data;
    if (outs.length >= 3) {
      const cls = outs[1].data, C = outs[1].dims[2], same = outs[2].data;
      return { order: decodeBlocksOrder(a, same, boxes, cls, C), layout: { cls, C, same } };
    }
    return { order: decodeOrder(a, n), layout: null };
  }

  /** OCR of one grayscale page { data: Uint8Array, width, height }. */
  async page(gray, { turn = true, tables = true } = {}) {
    const t = {};
    let t0 = now();
    const up = await this.upright(gray, turn);
    t.orient_ms = now() - t0;
    t0 = now();
    const boxes = await this.detect(up.gray);
    t.det_ms = now() - t0;
    t0 = now();
    const { rec } = await this.recognize(up.gray, boxes);
    t.rec_ms = now() - t0;
    const page = await this.assemble(up.gray.width, up.gray.height, boxes, rec, t, { tables });
    page.rotation = 90 * up.k;
    page.skew = Math.round(up.skew * 100) / 100;
    if (up.skew) {
      const { width: w, height: h } = up.gray;
      for (const ln of page.lines) ln.box = unrotateBox(ln.box, up.skew, w, h);
      for (const b of page.blocks) b.bbox = unrotateBox(b.bbox, up.skew, w, h);
    }
    return page;
  }

  async assemble(W, H, boxes0, rec0, t, { tables = true } = {}) {
    const t0 = now();
    const cuts = this.gutterCuts(boxes0, rec0);
    const boxes = [], rec = [];
    boxes0.forEach((b, i) => {
      for (const [bx, tx, tk] of this.splitLine(b, rec0[i][2], SPLIT_GAP, cuts[i])) {
        boxes.push(bx.map(Math.fround));
        rec.push([tx, rec0[i][1], tk]);
      }
    });
    const texts = rec.map((r) => r[0]);
    const { order, layout } = await this.readingOrder(boxes, texts, W, H);
    t.order_ms = now() - t0;
    t.total_ms = Object.values(t).reduce((s, v) => s + v, 0);
    const lines = order.map((i) => ({ box: boxes[i].map(r1), text: rec[i][0], conf: Math.round(rec[i][1] * 1e4) / 1e4 }));
    let markdown = null, blocks = [];
    if (layout) {
      blocks = pageBlocks(boxes, texts, order, layout.cls, layout.C, layout.same);
      const cells = new Map(boxes.map((b, i) => [i, this.splitLine(b, rec[i][2], CELL_GAP).map(([bx, tx]) => [bx, tx])]));
      markdown = pageMarkdown(boxes, texts, order, layout.cls, layout.C, layout.same, { keepFurniture: this.furniture, cells, tables });
    } else if (boxes.length < 2) {
      markdown = texts.join("\n");
    }
    return { lines, markdown, blocks, timings: t, rotation: 0, skew: 0, get text() { return lines.map((l) => l.text).join("\n"); } };
  }
}
