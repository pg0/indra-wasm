"""Reference outputs of the Python Indra pipeline, for the JS parity test.

Writes test/fixtures/<name>.pgm (the grayscale page the pipeline reads) and
<name>.json (intermediate and final results).  Run with the indra-ocr venv:
    python tools/make_reference.py <bench.pdf> <test.png> <cjk_font> <out_dir>
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

from indra_ocr.models import model_dir
from indra_ocr.pipeline import OrtOCR, orient_input

pdf_path, png_path, font_path, out = sys.argv[1:5]
out = Path(out)
out.mkdir(parents=True, exist_ok=True)
pages = {}

doc = pdfium.PdfDocument(pdf_path)
for i in range(len(doc)):
    arr = doc[i].render(scale=200 / 72.0, grayscale=True).to_numpy()
    pages[f"bench_p{i + 1}"] = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
pages["invoice_png"] = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

img = Image.new("L", (1400, 900), 255)
d = ImageDraw.Draw(img)
f_big, f = ImageFont.truetype(font_path, 44), ImageFont.truetype(font_path, 30)
d.text((60, 50), "第三章 光学字符识别", font=f_big, fill=0)
cjk_lines = [
    "光学字符识别是把图像中的文字转换成可编辑文本的技术。",
    "本系统在浏览器中运行，不需要服务器，也不上传任何文件。",
    "模型大小约为十八兆字节，首次加载后会被缓存。",
    "Indra OCR supports Latin text as well: 2026-10-14, 1.234,56 EUR",
    "日本語のテキストも読めますか？ ひらがなとカタカナ。",
]
for k, t in enumerate(cjk_lines):
    d.text((60, 160 + k * 70), t, font=f, fill=0)
pages["cjk_mixed"] = np.array(img)

p1 = pages["bench_p1"]
pages["bench_p1_rot90"] = np.ascontiguousarray(np.rot90(p1, 1))
h, w = p1.shape
M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
pages["bench_p1_skew2"] = cv2.warpAffine(p1, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)

ocr = OrtOCR(model_dir(None), threads=4)
for name, gray in pages.items():
    hh, ww = gray.shape
    with open(out / f"{name}.pgm", "wb") as fh:
        fh.write(f"P5\n{ww} {hh}\n255\n".encode())
        fh.write(np.ascontiguousarray(gray, np.uint8).tobytes())
    orient_p = ocr.orient.run(None, {"x": orient_input(gray)[None, None]})[0][0].tolist()
    up, k, skew = ocr.upright(gray)
    boxes = ocr.detect(up)
    rec = ocr.recognize(up, boxes)
    crops_hanzi = None
    if ocr.script is not None and len(boxes):
        from indra_ocr.line_mask import crop_lines
        crops_hanzi = ocr.hanzi_lines(crop_lines(up, boxes[:, :4], ocr.height, None))
    page = ocr(gray)
    ref = {
        "name": name,
        "size": [ww, hh],
        "orient_p": orient_p,
        "rotation": page.rotation,
        "skew": page.skew,
        "det_boxes": [[round(float(v), 3) for v in b] for b in boxes],
        "hanzi": crops_hanzi,
        "rec": [[t, round(c, 5)] for t, c, _ in rec],
        "lines": [{"box": ln.box, "text": ln.text, "conf": ln.conf} for ln in page.lines],
        "markdown": page.markdown,
        "blocks": page.blocks,
    }
    (out / f"{name}.json").write_text(json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8")
    print(name, ww, hh, "rot", page.rotation, "skew", page.skew, "lines", len(page.lines))
