// Ported from indra-ocr 0.3.1 vocab.py (Apache-2.0, Datalab); changed: rewritten in JavaScript.
// Recognizer vocabulary: class 0 is the CTC blank, class i + 1 is chars[i].
// A token is a character, or a LaTeX command such as "\\alpha".

export class Vocab {
  constructor(chars) {
    this.chars = chars;
    this.index = new Map(chars.map((c, i) => [c, i + 1]));
    this.hasCommands = chars.some((c) => [...c].length > 1);
  }

  static fromJson(s) {
    const d = JSON.parse(s);
    if ((d.blank ?? 0) !== 0) throw new Error("only blank=0 vocabularies are supported");
    return new Vocab(d.chars);
  }

  decode(ids) {
    const toks = [];
    for (const i of ids) if (i > 0 && i <= this.chars.length) toks.push(this.chars[i - 1]);
    if (!this.hasCommands) return toks.join("");
    const out = [];
    for (let k = 0; k < toks.length; k++) {
      const t = toks[k];
      out.push(t);
      // "\\alpha" + "x" must not read back as "\\alphax"
      const nxt = toks[k + 1] ?? "";
      if (t.length > 2 && t[0] === "\\" && /^\p{L}+$/u.test(t.slice(1)) && /^\p{L}/u.test(nxt)) {
        out.push(" ");
      }
    }
    return out.join("");
  }
}
