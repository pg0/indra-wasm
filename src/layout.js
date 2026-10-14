// Ported from indra-ocr 0.3.1 layout_post.py (Apache-2.0, Datalab); changed: rewritten in JavaScript.
// Page markdown from the ordered lines and the layout heads (layout_post.py).
import { median } from "./imgproc.js";
import { BLOCK_CLASSES, blockClass, strip } from "./order.js";

const HEADER = BLOCK_CLASSES.indexOf("Page-Header");
const FOOTER = BLOCK_CLASSES.indexOf("Page-Footer");
const SECTION = BLOCK_CLASSES.indexOf("Section-Header");
const LIST = BLOCK_CLASSES.indexOf("List-Group");
const TABLE = BLOCK_CLASSES.indexOf("Table");

const WHOLE_MATH = /^\$[^$]+\$$/u;
const ITEM = /^\s*([•\-–▪◦*·→]|\(?\p{Nd}{1,3}[.)]|\(?[a-zA-Z][.)])(\s|$)/u;

const isAlpha = (c) => /^\p{L}$/u.test(c);
const isLower = (c) => /^\p{Ll}$/u.test(c);

function isCjk(c) {
  const o = c.codePointAt(0);
  return (
    (o >= 0x3000 && o <= 0x30ff) || (o >= 0x3400 && o <= 0x9fff) ||
    (o >= 0xf900 && o <= 0xfaff) || (o >= 0xff00 && o <= 0xffef) ||
    (o >= 0x20000 && o <= 0x2fa1f)
  );
}

/** Join the lines of one paragraph; hyphenated line ends are rejoined. */
export function joinLines(texts) {
  let out = [];
  for (const raw of texts) {
    const t = [...strip(raw)];
    if (!t.length) continue;
    const n = out.length;
    if (n > 1 && out[n - 1] === "-" && isAlpha(out[n - 2]) && isAlpha(t[0]) && isLower(t[0])) {
      out = out.slice(0, -1).concat(t);
    } else if (n && isCjk(out[n - 1]) && isCjk(t[0])) {
      out = out.concat(t);
    } else {
      out = n ? out.concat([" "], t) : t;
    }
  }
  return out.join("");
}

/** Runs of the reading order with a positive same-block logit between neighbours. */
export function splitBlocks(order, same, n) {
  const blocks = [];
  for (const i of order) {
    const last = blocks[blocks.length - 1];
    if (last && same[last[last.length - 1] * n + i] > 0) last.push(i);
    else blocks.push([i]);
  }
  return blocks;
}

const escapeHtml = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

function tableRows(boxes) {
  const cy = boxes.map((b) => Math.fround((b[1] + b[3]) / 2));
  const h = median(boxes.map((b) => Math.fround(b[3] - b[1])));
  const rows = [];
  const order = cy.map((_, i) => i).sort((a, b) => cy[a] - cy[b]);
  for (const i of order) {
    const last = rows[rows.length - 1];
    if (last && Math.abs(cy[i] - last.reduce((s, k) => s + cy[k], 0) / last.length) < 0.5 * h) last.push(i);
    else rows.push([i]);
  }
  return rows.map((r) => r.sort((a, b) => boxes[a][0] - boxes[b][0]));
}

function mergeWrapped(rows, boxes) {
  if (rows.length < 3) return rows;
  const h = median(boxes.map((b) => Math.fround(b[3] - b[1])));
  const gaps = [];
  for (let k = 0; k + 1 < rows.length; k++) {
    const top = Math.min(...rows[k + 1].map((i) => boxes[i][1]));
    const bot = Math.max(...rows[k].map((i) => boxes[i][3]));
    gaps.push(Math.fround(top - bot));
  }
  const med = median(gaps);
  const out = [rows[0].slice()];
  let prev = rows[0];
  rows.slice(1).forEach((r, k) => {
    const g = gaps[k];
    if (g < 0.5 * med && g < 0.6 * h && r.length < prev.length) out[out.length - 1].push(...r);
    else out.push(r.slice());
    prev = r;
  });
  return out;
}

/** An HTML table from cell lines (rows by vertical centre, columns from the fullest rows). */
export function tableHtml(boxes, texts) {
  let rows = tableRows(boxes);
  const spanX = Math.max(...boxes.map((b) => b[2])) - Math.min(...boxes.map((b) => b[0]));
  const widest = Math.max(...rows.map((r) => r.length));
  if (
    rows.length > 2 && rows[0].length === 1 && widest > 1 &&
    boxes[rows[0][0]][2] - boxes[rows[0][0]][0] > 0.6 * spanX
  ) {
    const keep = texts.map((_, k) => k).filter((k) => k !== rows[0][0]);
    return texts[rows[0][0]] + "\n\n" + tableHtml(keep.map((k) => boxes[k]), keep.map((k) => texts[k]));
  }
  const width = widest;
  const full = rows.filter((r) => r.length === width);
  rows = mergeWrapped(rows, boxes);
  const bands = [];
  for (let c = 0; c < width; c++) {
    bands.push([median(full.map((r) => boxes[r[c]][0])), median(full.map((r) => boxes[r[c]][2]))]);
  }
  const edges = [-Infinity];
  for (let c = 0; c + 1 < width; c++) edges.push((bands[c][1] + bands[c + 1][0]) / 2);
  edges.push(Infinity);
  // np.searchsorted(edges, x, side="right") - 1
  const band = (x) => {
    let k = 0;
    while (k < edges.length && edges[k] <= x) k++;
    return k - 1;
  };
  const out = ["<table>"];
  rows.forEach((r, ri) => {
    const tag = ri === 0 && rows.length > 1 ? "th" : "td";
    const cells = new Map();
    const sorted = r.slice().sort((a, b) => boxes[a][1] - boxes[b][1] || boxes[a][0] - boxes[b][0]);
    for (const k of sorted) {
      const [x0, , x1] = boxes[k];
      const first = band(x0), last = band(x1);
      let span = [];
      for (let c = Math.max(first, 0); c <= Math.min(last, width - 1); c++) {
        if (Math.min(x1, bands[c][1]) - Math.max(x0, bands[c][0]) > 0.5 * (bands[c][1] - bands[c][0])) span.push(c);
      }
      if (!span.length) span = [Math.min(Math.max(first, 0), width - 1)];
      const c0 = span[0];
      const prev = cells.get(c0);
      const text = prev === undefined ? texts[k] : prev[1] + " " + texts[k];
      cells.set(c0, [prev === undefined ? span.length : Math.max(prev[0], span.length), text]);
    }
    const tds = [];
    let c = 0;
    while (c < width) {
      if (cells.has(c)) {
        const [n, text] = cells.get(c);
        const attr = n > 1 ? ` colspan="${n}"` : "";
        tds.push(`<${tag}${attr}>${escapeHtml(text)}</${tag}>`);
        c += n;
      } else {
        tds.push(`<${tag}></${tag}>`);
        c += 1;
      }
    }
    out.push("<tr>" + tds.join("") + "</tr>");
  });
  out.push("</table>");
  return out.join("\n");
}

const r1 = (v) => Math.round(v * 10) / 10;

/** Layout blocks in reading order: { label, bbox, text }. */
export function pageBlocks(boxes, texts, order, cls, C, same) {
  const n = boxes.length;
  return splitBlocks(order, same, n).map((blk) => ({
    label: BLOCK_CLASSES[blockClass(cls, C, blk)],
    bbox: [
      r1(Math.min(...blk.map((i) => boxes[i][0]))), r1(Math.min(...blk.map((i) => boxes[i][1]))),
      r1(Math.max(...blk.map((i) => boxes[i][2]))), r1(Math.max(...blk.map((i) => boxes[i][3]))),
    ],
    text: blk.map((i) => texts[i]).join("\n"),
  }));
}

/** A table as plain text: one line per row, cells separated by two spaces. */
export function tableText(boxes, texts) {
  return mergeWrapped(tableRows(boxes), boxes).map((r) => r.map((k) => texts[k]).join("  ")).join("\n");
}

/**
 * Markdown for one page. cells: line index -> [[box, text]] when a line holds
 * several cells. tables: false writes table blocks as plain text rows.
 */
export function pageMarkdown(boxes, texts, order, cls, C, same, { keepFurniture = false, cells = null, tables = true } = {}) {
  const n = boxes.length;
  const parts = [];
  for (const blk of splitBlocks(order, same, n)) {
    const c = blockClass(cls, C, blk);
    const tb = blk.map((i) => texts[i]);
    if ((c === HEADER || c === FOOTER) && !keepFurniture) continue;
    if (tb.every((t) => WHOLE_MATH.test(strip(t)))) {
      parts.push(tb.map((t) => "$$" + strip(t).slice(1, -1) + "$$").join("\n"));
      continue;
    }
    if (c === SECTION) {
      parts.push("## " + joinLines(tb));
    } else if (c === TABLE && blk.length > 1) {
      const cb = [], ct = [];
      for (const i of blk) {
        const list = cells?.get(i)?.length ? cells.get(i) : [[boxes[i], texts[i]]];
        for (const [bx, tx] of list) { cb.push(bx.map(Math.fround)); ct.push(tx); }
      }
      parts.push(tables ? tableHtml(cb, ct) : tableText(cb, ct));
    } else if (c === LIST) {
      const items = [];
      for (const t of tb) {
        if (!items.length || ITEM.test(t)) items.push([t]);
        else items[items.length - 1].push(t);
      }
      parts.push(items.map(joinLines).join("\n"));
    } else {
      parts.push(joinLines(tb));
    }
  }
  return parts.filter((p) => p).join("\n\n");
}
