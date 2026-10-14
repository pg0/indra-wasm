# indra-wasm

[Indra OCR](https://github.com/datalab-to/indra) in the browser. PDFs and images go in, markdown comes out, and nothing leaves the device. The indra-ocr 0.3.1 pipeline is ported to JavaScript on ONNX Runtime Web (WebAssembly). There is no build step: serve the folder and open `index.html`.

## Use it

Any static file server works:

```
npx serve .          # or: python -m http.server
```

Open the page, drop a PDF or an image, and copy or download the markdown. On the first run the models download from Hugging Face: about 9 MB for Latin text, plus 9 MB the first time a page holds Chinese. The browser caches them after that.

In your own page:

```js
import { Indra } from "./src/indra.js";

const indra = new Indra({ onProgress: (p) => console.log(p) });
const { markdown, pages } = await indra.recognize(file); // File, Blob or ArrayBuffer
// pages[i]: { markdown, lines: [{ box, text, conf }], blocks, rotation, skew, timings }
```

OCR runs in a web worker. PDF pages are rendered with pdf.js at 200 dpi. ONNX Runtime Web and pdf.js load from jsDelivr at pinned versions (1.30.0 and 6.3.289).

## What it does

The whole indra-ocr OCR path, step for step:

- the page is turned upright (orient.onnx) and its tilt is straightened
- lines are detected, with a second, sharper pass for small print
- each line is read by the Latin recognizer, or by the CJK recognizer when script.onnx finds Chinese
- lines are split at column gutters and table cells
- the order model gives the reading order and the layout blocks
- markdown is built from them: headings, paragraphs with hyphenated words rejoined, lists, HTML tables

Not ported: the PDF text-layer path (`--mode auto` / `text`), math as LaTeX, and multi-page TIFF (browsers cannot decode TIFF).

## Accuracy against the Python package

`npm test` runs the JavaScript pipeline on the same grayscale pages as the Python package and compares the results (fixtures from `tools/make_reference.py`). On 13 pages (a 9-page paper, the same page turned 90 degrees, the same page tilted 2 degrees, a mixed Chinese and Latin page, and a small image), the markdown and every line text are identical. Line boxes differ by at most 0.1 px.

PDFs go through pdf.js here and through pdfium in Python. The two rasterize glyphs slightly differently, so PDF output can differ in a few places, mostly inside tables and figures (99.5 % character similarity on the 9-page paper).

Speed on the 9-page paper at 200 dpi (Edge on a Ryzen 9 9950X): 3.3 s with 4 threads, 6.7 s with one; the Python package takes 2.5 s on 4 cores. Threads need a cross-origin isolated page. `coi.js` takes care of that on hosts that cannot send the COOP/COEP headers, such as GitHub Pages: a small service worker adds them and the page reloads once. More than 4 threads did not help.

## Tests

```
npm install
npm run models                       # downloads the models to test/models
python tools/make_reference.py ...   # needs indra-ocr and pillow; writes test/fixtures
npm test
```

## License

The code is Apache-2.0 (see LICENSE and NOTICE). It is a port of indra-ocr by Datalab, also Apache-2.0.

The model weights are not part of this repository. They are downloaded from [datalab-to/indra-ocr](https://huggingface.co/datalab-to/indra-ocr) and licensed by Datalab under a [modified OpenRAIL-M license](https://huggingface.co/datalab-to/indra-ocr/blob/v3.1/LICENSE). By using them you agree to its terms, including the use restrictions in its Attachment A. In short, commercial use is not allowed for organizations with more than $20M in revenue or funding (personal and research use excepted), or for any product or service that competes with Datalab's. Read the license before you use or host this.

Not affiliated with or endorsed by Datalab.
