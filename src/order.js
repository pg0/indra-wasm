// Ported from indra-ocr 0.3.1 order_post.py (Apache-2.0, Datalab); changed: rewritten in JavaScript.
// Reading order (order_post.py): line features for the order model, decoding
// of its pairwise logits, XY-cut and top-down baselines, block clustering.
import { median, pyRound } from "./imgproc.js";

export const BLOCK_CLASSES = [
  "Text", "Section-Header", "Page-Header", "Page-Footer", "List-Group",
  "Table", "Caption", "Footnote", "Table-Of-Contents", "Form",
];
const GRID = new Set(["Table", "Form", "Table-Of-Contents"].map((c) => BLOCK_CLASSES.indexOf(c)));
export const N_FEATS = 16;

const DIGIT = /[\p{Nd}⁰¹²³⁴-⁹₀-₉①-⑨⑴-⑼⒈-⒐⓵-⓽❶-❾➀-➈➊-➒]/u;
const pyIsDigit = (s) => s.length > 0 && [...s].every((c) => DIGIT.test(c));

/** Python str.strip() for whitespace. */
export const strip = (s) => s.replace(/^\s+|\s+$/gu, "");

/** [N] boxes, page size, texts -> Float32Array [N * 16]. */
export function lineFeatures(boxes, W, H, texts) {
  const n = boxes.length;
  const out = new Float32Array(n * N_FEATS);
  const f = Math.fround;
  const hs = boxes.map((b) => Math.max(f(b[3] - b[1]), 1e-3));
  const medH = median(hs.map(f));
  boxes.forEach((b, i) => {
    const w = Math.max(f(b[2] - b[0]), 1e-3), h = hs[i];
    const g = [
      b[0] / W, b[1] / H, b[2] / W, b[3] / H,
      (b[0] + b[2]) / (2 * W), (b[1] + b[3]) / (2 * H),
      w / W, Math.log(h / medH), Math.log(w / h) / 4, (medH / H) * 20,
    ];
    out.set(g, i * N_FEATS);
    const t = strip(texts[i] ?? "");
    if (!t) return;
    const cs = [...t];
    const o = i * N_FEATS + 10;
    out[o] = Math.log1p(cs.length) / 5;
    out[o + 1] = cs.filter((c) => DIGIT.test(c)).length / cs.length;
    const head = cs.slice(0, 3).join("").replace(/[.)]+$/u, "");
    out[o + 2] = "•-–▪◦*·→".includes(cs[0]) || (pyIsDigit(head) && cs.length > 3) ? 1 : 0;
    out[o + 3] = t.endsWith("-") ? 1 : 0;
    out[o + 4] = ".!?:;".includes(cs[cs.length - 1]) ? 1 : 0;
    out[o + 5] = /^\p{Lu}$/u.test(cs[0]) ? 1 : 0;
  });
  return out;
}

const logaddexp0 = (x) => (x > 0 ? x + Math.log1p(Math.exp(-x)) : Math.log1p(Math.exp(x)));

/** Order from a pairwise logit matrix a (n*n, a[i][j] > 0: i before j). */
export function decodeOrder(a, n, sweeps = 4) {
  if (n <= 1) return [...Array(n).keys()];
  const lp = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) lp[i] = Math.fround(-logaddexp0(-a[i]));
  const score = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let r = 0, c = 0;
    for (let j = 0; j < n; j++) { r += lp[i * n + j]; c += lp[j * n + i]; }
    score[i] = -r + c;
  }
  let order = [...Array(n).keys()].sort((x, y) => score[x] - score[y]);
  for (let s = 0; s < sweeps; s++) {
    let improved = false;
    for (const x of order.slice()) {
      const rest = order.filter((v) => v !== x);
      const m = rest.length;
      const gains = new Float64Array(m + 1);
      let acc = 0;
      for (let k = 1; k <= m; k++) { acc += lp[rest[k - 1] * n + x]; gains[k] = acc; }
      acc = 0;
      for (let k = m - 1; k >= 0; k--) { acc += lp[x * n + rest[k]]; gains[k] += acc; }
      let k = 0;
      for (let i = 1; i <= m; i++) if (gains[i] > gains[k]) k = i;
      const cur = order.indexOf(x);
      if (gains[k] > gains[cur] + 1e-6) {
        rest.splice(k, 0, x);
        order = rest;
        improved = true;
      }
    }
    if (!improved) break;
  }
  return order;
}

/** Rows top to bottom (centres within half a line height share a row), left to right. */
export function orderTopDown(boxes) {
  if (!boxes.length) return [];
  const h = median(boxes.map((b) => Math.fround(b[3] - b[1])));
  const unit = Math.max(h * 0.5, 1e-3);
  const key = boxes.map((b) => pyRound(Math.fround((b[1] + b[3]) / 2) / unit));
  return boxes.map((_, i) => i).sort((i, j) => key[i] - key[j] || boxes[i][0] - boxes[j][0]);
}

/** Recursive XY-cut at the widest whitespace gap on either axis. */
export function orderXYCut(boxes) {
  const bestCut = (ids, axis) => {
    const lo = ids.map((i) => boxes[i][axis === 1 ? 1 : 0]);
    const hi = ids.map((i) => boxes[i][axis === 1 ? 3 : 2]);
    const o = lo.map((_, k) => k).sort((p, q) => lo[p] - lo[q]);
    let end = hi[o[0]], gap = 0, best = -1;
    for (let k = 1; k < o.length; k++) {
      const g = lo[o[k]] - end;
      if (g > gap) { gap = g; best = k; }
      end = Math.max(end, hi[o[k]]);
    }
    return [gap, best, o];
  };
  const rec = (ids) => {
    if (ids.length <= 1) return ids.slice();
    const [gy, ky, oy] = bestCut(ids, 1);
    const [gx, kx, ox] = bestCut(ids, 0);
    if (ky < 0 && kx < 0) return orderTopDown(ids.map((i) => boxes[i])).map((k) => ids[k]);
    const [k, o] = gy >= gx ? [ky, oy] : [kx, ox];
    return rec(o.slice(0, k).map((i) => ids[i])).concat(rec(o.slice(k).map((i) => ids[i])));
  };
  return rec(boxes.map((_, i) => i));
}

/** Average-linkage clustering of pairwise "same group" logits (n*n). */
export function cluster(logits, n) {
  if (n === 0) return [];
  const S = Float64Array.from(logits);
  for (let i = 0; i < n; i++) S[i * n + i] = 0;
  const size = new Float64Array(n).fill(1);
  const alive = new Uint8Array(n).fill(1);
  const members = [...Array(n).keys()].map((i) => [i]);
  const M = Float64Array.from(S);
  for (let i = 0; i < n; i++) M[i * n + i] = -Infinity;
  const argmaxRow = (j) => {
    let k = 0;
    for (let i = 1; i < n; i++) if (M[j * n + i] > M[j * n + k]) k = i;
    return k;
  };
  const best = new Int32Array(n), bestV = new Float64Array(n);
  for (let j = 0; j < n; j++) { best[j] = argmaxRow(j); bestV[j] = M[j * n + best[j]]; }
  for (let it = 0; it < n - 1; it++) {
    let a = 0;
    for (let i = 1; i < n; i++) if (bestV[i] > bestV[a]) a = i;
    let b = best[a];
    if (!(bestV[a] > 0)) break;
    [a, b] = [Math.min(a, b), Math.max(a, b)];
    for (let i = 0; i < n; i++) S[a * n + i] += S[b * n + i];
    for (let i = 0; i < n; i++) S[i * n + a] += S[i * n + b];
    S[a * n + a] = 0;
    size[a] += size[b];
    alive[b] = 0;
    members[a] = members[a].concat(members[b]);
    const row = new Float64Array(n);
    for (let i = 0; i < n; i++) row[i] = alive[i] && i !== a ? S[a * n + i] / (size[a] * size[i]) : -Infinity;
    for (let i = 0; i < n; i++) {
      M[a * n + i] = row[i]; M[i * n + a] = row[i];
      M[b * n + i] = -Infinity; M[i * n + b] = -Infinity;
    }
    bestV[b] = -Infinity;
    let ba = 0;
    for (let i = 1; i < n; i++) if (row[i] > row[ba]) ba = i;
    best[a] = ba;
    bestV[a] = row[ba];
    for (let j = 0; j < n; j++) {
      if (!alive[j] || j === a) continue;
      if (best[j] === a || best[j] === b) { best[j] = argmaxRow(j); bestV[j] = M[j * n + best[j]]; }
    }
    for (let j = 0; j < n; j++) {
      if (alive[j] && j !== a && row[j] > bestV[j]) { best[j] = a; bestV[j] = row[j]; }
    }
  }
  const out = [];
  for (let i = 0; i < n; i++) if (alive[i]) out.push(members[i].slice().sort((x, y) => x - y));
  return out;
}

/** Index of the largest summed class logit over lines idx (cls: n*C). */
export function blockClass(cls, C, idx) {
  const sum = new Float64Array(C);
  for (const i of idx) for (let c = 0; c < C; c++) sum[c] += cls[i * C + c];
  let k = 0;
  for (let c = 1; c < C; c++) if (sum[c] > sum[k]) k = c;
  return k;
}

/** Reading order with the layout heads: blocks first, then lines inside blocks. */
export function decodeBlocksOrder(a, same, boxes, cls, C) {
  const n = boxes.length;
  if (n <= 1) return [...Array(n).keys()];
  const blocks = cluster(same, n);
  const inner = blocks.map((b) => {
    const bx = b.map((i) => boxes[i]);
    const local = GRID.has(blockClass(cls, C, b)) ? orderTopDown(bx) : orderXYCut(bx);
    return local.map((k) => b[k]);
  });
  const m = blocks.length;
  const mean = new Float64Array(m * m);
  for (let p = 0; p < m; p++) {
    for (let q = p + 1; q < m; q++) {
      let s = 0;
      for (const i of blocks[p]) for (const j of blocks[q]) s += a[i * n + j];
      mean[p * m + q] = s / (blocks[p].length * blocks[q].length);
    }
  }
  const ab = new Float32Array(m * m);
  for (let p = 0; p < m; p++) {
    for (let q = 0; q < m; q++) ab[p * m + q] = p < q ? mean[p * m + q] : p > q ? -mean[q * m + p] : 0;
  }
  return decodeOrder(ab, m).flatMap((k) => inner[k]);
}
