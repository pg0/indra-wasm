// The few OpenCV operations the pipeline needs, ported to plain JS.
// Images are { data, width, height }: data is a Uint8Array (grayscale) or a
// Float32Array (ink maps). Coordinate mapping and fixed-point steps follow
// OpenCV, so results stay within a gray level of cv2 on the same input.

export function pyRound(x) {
  // Python round(): halves go to the even neighbour.
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// cvRound / saturate_cast<int>(double): round half to even.
const cvRound = pyRound;

function clampU8(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// One axis of OpenCV's linear resize table: source index and weight per output.
// areaMode is cv2.INTER_AREA when it does not shrink on both axes.
function linearTable(srcSize, dstSize, areaMode) {
  const scale = srcSize / dstSize;
  const invScale = dstSize / srcSize;
  const idx = new Int32Array(dstSize);
  const frac = new Float32Array(dstSize);
  for (let d = 0; d < dstSize; d++) {
    let s, f;
    if (!areaMode) {
      f = Math.fround((d + 0.5) * scale - 0.5);
      s = Math.floor(f);
      f = Math.fround(f - s);
    } else {
      s = Math.floor(d * scale);
      f = Math.fround(d + 1 - (s + 1) * invScale);
      f = f <= 0 ? 0 : Math.fround(f - Math.floor(f));
    }
    if (s < 0) { s = 0; f = 0; }
    if (s >= srcSize - 1) { s = srcSize - 1; f = 0; }
    idx[d] = s;
    frac[d] = f;
  }
  return { idx, frac };
}

function resizeLinearU8(src, dw, dh, areaMode) {
  const { data, width: sw, height: sh } = src;
  const xt = linearTable(sw, dw, areaMode);
  const yt = linearTable(sh, dh, areaMode);
  // 11-bit fixed-point weights, as OpenCV uses for 8-bit images.
  const ax0 = new Int32Array(dw), ax1 = new Int32Array(dw);
  for (let x = 0; x < dw; x++) {
    ax0[x] = cvRound((1 - xt.frac[x]) * 2048);
    ax1[x] = cvRound(xt.frac[x] * 2048);
  }
  const rowCache = new Map();
  const hrow = (sy) => {
    let r = rowCache.get(sy);
    if (r) return r;
    r = new Int32Array(dw);
    const off = sy * sw;
    for (let x = 0; x < dw; x++) {
      const sx = xt.idx[x];
      const sx1 = sx + 1 < sw ? sx + 1 : sx;
      r[x] = data[off + sx] * ax0[x] + data[off + sx1] * ax1[x];
    }
    if (rowCache.size > 4) rowCache.delete(rowCache.keys().next().value);
    rowCache.set(sy, r);
    return r;
  };
  const out = new Uint8Array(dw * dh);
  for (let y = 0; y < dh; y++) {
    const sy = yt.idx[y];
    const b0 = cvRound((1 - yt.frac[y]) * 2048);
    const b1 = cvRound(yt.frac[y] * 2048);
    const r0 = hrow(sy);
    const r1 = hrow(sy + 1 < sh ? sy + 1 : sy);
    const o = y * dw;
    for (let x = 0; x < dw; x++) {
      // OpenCV's SIMD rounding for the vertical pass of 8-bit linear resize.
      const v = (((b0 * (r0[x] >> 4)) >> 16) + ((b1 * (r1[x] >> 4)) >> 16) + 2) >> 2;
      out[o + x] = clampU8(v);
    }
  }
  return { data: out, width: dw, height: dh };
}

// OpenCV's area table: source index and weight for each output cell.
function areaTable(srcSize, dstSize) {
  const scale = srcSize / dstSize;
  const tab = [];
  for (let d = 0; d < dstSize; d++) {
    const fs1 = d * scale;
    const fs2 = fs1 + scale;
    const cell = Math.min(scale, srcSize - fs1);
    let s2 = Math.floor(fs2);
    let s1 = Math.ceil(fs1);
    s2 = Math.min(s2, srcSize - 1);
    s1 = Math.min(s1, s2);
    const w = [];
    if (s1 - fs1 > 1e-3) w.push([s1 - 1, (s1 - fs1) / cell]);
    for (let s = s1; s < s2; s++) w.push([s, 1 / cell]);
    if (fs2 - s2 > 1e-3) w.push([s2, Math.min(Math.min(fs2 - s2, 1), cell) / cell]);
    tab.push(w);
  }
  // Flat typed arrays: entries start[d] .. start[d + 1] - 1 belong to cell d.
  const start = new Int32Array(dstSize + 1);
  for (let d = 0; d < dstSize; d++) start[d + 1] = start[d] + tab[d].length;
  const idx = new Int32Array(start[dstSize]), wt = new Float64Array(start[dstSize]);
  tab.flat().forEach(([s, w], k) => { idx[k] = s; wt[k] = w; });
  return { start, idx, wt };
}

function resizeAreaU8(src, dw, dh) {
  const { data, width: sw, height: sh } = src;
  const sx = sw / dw, sy = sh / dh;
  const out = new Uint8Array(dw * dh);
  const ix = Math.round(sx), iy = Math.round(sy);
  if (Math.abs(sx - ix) < 1e-12 && Math.abs(sy - iy) < 1e-12) {
    // Integer factor: plain box average (OpenCV's fast path).
    const area = ix * iy;
    for (let y = 0; y < dh; y++) {
      for (let x = 0; x < dw; x++) {
        let sum = 0;
        for (let j = 0; j < iy; j++) {
          const off = (y * iy + j) * sw + x * ix;
          for (let i = 0; i < ix; i++) sum += data[off + i];
        }
        out[y * dw + x] = Math.floor((sum + (area >> 1)) / area);
      }
    }
    return { data: out, width: dw, height: dh };
  }
  const xt = areaTable(sw, dw);
  const yt = areaTable(sh, dh);
  const rows = new Float64Array(sh * dw);
  for (let y = 0; y < sh; y++) {
    const off = y * sw;
    for (let x = 0; x < dw; x++) {
      let acc = 0;
      for (let k = xt.start[x]; k < xt.start[x + 1]; k++) acc += data[off + xt.idx[k]] * xt.wt[k];
      rows[y * dw + x] = acc;
    }
  }
  // Row by row over the source rows of each output row: same summation order
  // per pixel as a per-pixel loop, but sequential memory access.
  const acc = new Float64Array(dw);
  for (let y = 0; y < dh; y++) {
    acc.fill(0);
    for (let k = yt.start[y]; k < yt.start[y + 1]; k++) {
      const r = yt.idx[k] * dw, w = yt.wt[k];
      for (let x = 0; x < dw; x++) acc[x] += rows[r + x] * w;
    }
    const o = y * dw;
    for (let x = 0; x < dw; x++) out[o + x] = clampU8(cvRound(acc[x]));
  }
  return { data: out, width: dw, height: dh };
}

/** cv2.resize of a uint8 image; interp is "area" or "linear". */
export function resize(src, dw, dh, interp) {
  if (dw === src.width && dh === src.height) return { data: src.data.slice(), width: dw, height: dh };
  if (interp === "area") {
    if (src.width / dw >= 1 && src.height / dh >= 1) return resizeAreaU8(src, dw, dh);
    return resizeLinearU8(src, dw, dh, true);
  }
  return resizeLinearU8(src, dw, dh, false);
}

/** cv2.getRotationMatrix2D: [a, b, c, d, e, f] for x' = a x + b y + c, y' = d x + e y + f. */
export function rotationMatrix(cx, cy, degrees) {
  const r = (degrees * Math.PI) / 180;
  const a = Math.cos(r), b = Math.sin(r);
  return [a, b, (1 - a) * cx - b * cy, -b, a, b * cx + (1 - a) * cy];
}

function invertAffine(m) {
  const [a, b, c, d, e, f] = m;
  let det = a * e - b * d;
  det = det !== 0 ? 1 / det : 0;
  const A = e * det, B = -b * det, D = -d * det, E = a * det;
  return [A, B, -A * c - B * f, D, E, -D * c - E * f];
}

/**
 * cv2.warpAffine with INTER_LINEAR (same output size). border: "constant"
 * (value 0) or "replicate". Works on Uint8Array and Float32Array images; the
 * source position is quantized to 1/32 pixel as in OpenCV.
 */
export function warpAffine(src, m, border = "constant", borderValue = 0) {
  const { data, width: w, height: h } = src;
  const isU8 = data instanceof Uint8Array;
  const out = isU8 ? new Uint8Array(w * h) : new Float32Array(w * h);
  const M = invertAffine(m);
  const AB = 1024, ROUND = 16;
  const adelta = new Int32Array(w), bdelta = new Int32Array(w);
  for (let x = 0; x < w; x++) {
    adelta[x] = cvRound(M[0] * x * AB);
    bdelta[x] = cvRound(M[3] * x * AB);
  }
  // Bilinear weights per 1/32 sub-pixel offset (fixed point for 8-bit images).
  const tab = new Float64Array(32 * 32 * 4);
  for (let fy = 0; fy < 32; fy++) {
    for (let fx = 0; fx < 32; fx++) {
      const ax = fx / 32, ay = fy / 32;
      let ws = [(1 - ax) * (1 - ay), ax * (1 - ay), (1 - ax) * ay, ax * ay];
      if (isU8) {
        const iw = ws.map((v) => cvRound(v * 32768));
        const sum = iw[0] + iw[1] + iw[2] + iw[3];
        if (sum !== 32768) {
          let k = 0;
          for (let i = 1; i < 4; i++) if (iw[i] > iw[k]) k = i;
          iw[k] += 32768 - sum;
        }
        ws = iw;
      } else {
        ws = ws.map((v) => Math.fround(v));
      }
      tab.set(ws, (fy * 32 + fx) * 4);
    }
  }
  const pix = (x, y) => {
    if (x >= 0 && x < w && y >= 0 && y < h) return data[y * w + x];
    if (border === "replicate") {
      return data[Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)];
    }
    return borderValue;
  };
  const constant = border === "constant";
  // Ink span of each source row (lo > hi when the row is blank): a pixel whose
  // 2x2 neighbourhood lies outside both spans is exactly 0, so mostly white
  // pages skip most of the arithmetic.
  const lo = new Int32Array(h).fill(w), hi = new Int32Array(h).fill(-1);
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let a = 0, b = w - 1;
    while (a < w && data[o + a] === 0) a++;
    if (a === w) continue;
    while (data[o + b] === 0) b--;
    lo[y] = a;
    hi[y] = b;
  }
  for (let y = 0; y < h; y++) {
    const X0 = cvRound((M[1] * y + M[2]) * AB) + ROUND;
    const Y0 = cvRound((M[4] * y + M[5]) * AB) + ROUND;
    for (let x = 0; x < w; x++) {
      const X = (X0 + adelta[x]) >> 5;
      const Y = (Y0 + bdelta[x]) >> 5;
      const sx = X >> 5, sy = Y >> 5;
      const t = ((Y & 31) * 32 + (X & 31)) * 4;
      let v;
      // Interior pixels read the 2x2 neighbourhood directly; this loop is the
      // hot path of the skew search, which warps the page 32 times.
      if (sx >= 0 && sy >= 0 && sx < w - 1 && sy < h - 1) {
        if ((sx + 1 < lo[sy] || sx > hi[sy]) && (sx + 1 < lo[sy + 1] || sx > hi[sy + 1])) {
          out[y * w + x] = 0;
          continue;
        }
        const i = sy * w + sx;
        v = data[i] * tab[t] + data[i + 1] * tab[t + 1] + data[i + w] * tab[t + 2] + data[i + w + 1] * tab[t + 3];
      } else if (constant && (sx < -1 || sx >= w || sy < -1 || sy >= h)) {
        out[y * w + x] = borderValue;
        continue;
      } else {
        v = pix(sx, sy) * tab[t] + pix(sx + 1, sy) * tab[t + 1] +
          pix(sx, sy + 1) * tab[t + 2] + pix(sx + 1, sy + 1) * tab[t + 3];
      }
      out[y * w + x] = isU8 ? clampU8((v + 16384) >> 15) : v;
    }
  }
  return { data: out, width: w, height: h };
}

/** np.rot90(gray, -k): the image turned clockwise by k quarter turns. */
export function rotateClockwise(src, k) {
  k = ((k % 4) + 4) % 4;
  const { data, width: w, height: h } = src;
  if (k === 0) return src;
  if (k === 2) {
    const out = new Uint8Array(w * h);
    for (let i = 0; i < w * h; i++) out[i] = data[w * h - 1 - i];
    return { data: out, width: w, height: h };
  }
  const out = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = data[y * w + x];
      // clockwise: (x, y) -> (h-1-y, x); counter-clockwise: (x, y) -> (y, w-1-x)
      if (k === 1) out[x * h + (h - 1 - y)] = v;
      else out[(w - 1 - x) * h + y] = v;
    }
  }
  return { data: out, width: h, height: w };
}

/**
 * cv2.connectedComponentsWithStats with 4-connectivity on a 0/1 mask.
 * Labels follow raster order of each component's first pixel.
 * Returns { n, labels, left, top, width, height, area } (index 0 = background).
 */
export function connectedComponents(mask, w, h) {
  const labels = new Int32Array(w * h);
  const left = [0], top = [0], right = [0], bottom = [0], area = [0];
  const stack = new Int32Array(w * h);
  let n = 1;
  for (let p = 0; p < w * h; p++) {
    if (!mask[p] || labels[p]) continue;
    const id = n++;
    let sp = 0;
    stack[sp++] = p;
    labels[p] = id;
    let l = w, t = h, r = -1, b = -1, a = 0;
    while (sp) {
      const q = stack[--sp];
      const x = q % w, y = (q - x) / w;
      a++;
      if (x < l) l = x;
      if (x > r) r = x;
      if (y < t) t = y;
      if (y > b) b = y;
      if (x > 0 && mask[q - 1] && !labels[q - 1]) { labels[q - 1] = id; stack[sp++] = q - 1; }
      if (x < w - 1 && mask[q + 1] && !labels[q + 1]) { labels[q + 1] = id; stack[sp++] = q + 1; }
      if (y > 0 && mask[q - w] && !labels[q - w]) { labels[q - w] = id; stack[sp++] = q - w; }
      if (y < h - 1 && mask[q + w] && !labels[q + w]) { labels[q + w] = id; stack[sp++] = q + w; }
    }
    left.push(l); top.push(t); right.push(r); bottom.push(b); area.push(a);
  }
  const width = right.map((r, i) => (i ? r - left[i] + 1 : 0));
  const height = bottom.map((b, i) => (i ? b - top[i] + 1 : 0));
  return { n, labels, left, top, width, height, area };
}

/** np.median of a numeric array. */
export function median(values) {
  const v = Float64Array.from(values).sort();
  const n = v.length;
  if (!n) return NaN;
  return n % 2 ? v[(n - 1) / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}
