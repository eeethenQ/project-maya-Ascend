"""ascend/make_tokenizer.py - turn the model's Hugging Face tokenizer into the directory serve/ reads.

The W8A8 export carries no tokenizer, and `serve/server.py` does not read Hugging Face's `tokenizer.json`
directly: it reads a pack tokenizer directory (`vocab.json`, `merges.txt`, and a small `tokenizer.json`
naming the pre-tokenizer - `tools/strata_tokenizer.py:202-214`).  DeepSeek's pre-tokenizer is already
transcribed there as `deepseek3` ("DeepSeek-V3 .. V4.1's tokenizer.json", `tools/strata_tokenizer.py:87`),
so the conversion is mechanical and *verifiable*: after writing the directory, this encodes a corpus with
both tokenizers and compares.

    python -m ascend.make_tokenizer --hf /mnt/weights/DeepSeek-V4-Flash-0731-w8a8 \\
        --out /home/q00999609/dsv41-tokenizer --template tools/ds4/chat_template.jinja --check
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

PRE = "deepseek3"

# Strings that exercise the pre-tokenizer's boundaries: digits, contractions, CJK, code, the specials.
CHECK_TEXTS = [
    "The capital of France is",
    "1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13",
    "don't stop, it's fine - we've been here",
    "def f(x: int) -> int:\n    return x * 2 + 1\n",
    "混合中文与 English，以及标点。还有数字 1234567。",
    "    indented line\n\ttabbed line\n\n\nblank lines above",
    "https://example.com/a?b=c&d=e#frag",
    "  leading and trailing spaces  ",
    "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>",
    "emoji 🙂 and combining é and ½ and Ⅻ",
]


def _write(path: pathlib.Path, text: str) -> None:
    """Write a file, never through a symlink.

    An earlier version of this script wrote `out/tokenizer.json` when `out` had been assembled by
    symlinking the model's own `tokenizer.json` into it, and the write followed the link into the shared
    model directory.  Replacing the link first, and writing through a temporary file, makes that
    impossible: nothing outside `out_dir` can be reached from here.
    """
    if path.is_symlink() or path.exists():
        path.unlink()
    tmp = path.with_name(path.name + ".writing")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def convert(hf_dir: pathlib.Path, out_dir: pathlib.Path, template: pathlib.Path | None = None) -> dict:
    """Write `vocab.json`, `merges.txt`, `tokenizer.json` (and the template) for `serve/`."""
    src = json.loads((pathlib.Path(hf_dir) / "tokenizer.json").read_text(encoding="utf-8"))
    model = src["model"]
    if not isinstance(model, dict) or model.get("type") != "BPE":
        got = model.get("type") if isinstance(model, dict) else type(model).__name__
        raise SystemExit(f"expected a BPE tokenizer, got {got!r}")
    vocab: dict[str, int] = dict(model["vocab"])
    special_ids: dict[str, int] = {}
    types: dict[int, int] = {}
    for added in src.get("added_tokens", []):
        vocab.setdefault(added["content"], int(added["id"]))
        # llama.cpp's token types: 1 normal, 3 control, 4 user-defined.  A CONTROL token is only matched
        # when the caller asks for it (`encode(..., parse_special=True)`), a USER_DEFINED one always - and
        # both detokenize to their literal text, which DeepSeek's `｜`-wrapped tokens need.
        types[int(added["id"])] = 3 if added.get("special") else 4
        if added.get("special"):
            special_ids[added["content"]] = int(added["id"])
    # every id 0..n-1 must be named, or the reader's reverse table has holes
    n = max(vocab.values()) + 1
    named = {i for i in vocab.values()}
    if len(named) != n:
        raise SystemExit(f"the vocabulary has {n} ids but {len(named)} distinct ones")
    merges = model.get("merges") or []
    if merges and isinstance(merges[0], (list, tuple)):
        merges = [" ".join(m) for m in merges]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    token_types = [types.get(i, 1) for i in range(n)]
    _write(out / "vocab.json", json.dumps(vocab, ensure_ascii=False))
    _write(out / "merges.txt", "\n".join(merges))
    _write(out / "token_type.json", json.dumps(token_types))
    meta = {"model": "gpt2", "pre": PRE, "vocab_size": n, "n_merges": len(merges),
            "special_ids": special_ids, "add_bos_token": False}
    _write(out / "tokenizer.json", json.dumps(meta, ensure_ascii=False, indent=1))
    if template is not None and pathlib.Path(template).is_file():
        _write(out / "chat_template.jinja", pathlib.Path(template).read_text(encoding="utf-8"))
    return {"vocab": n, "merges": len(merges), "specials": len(special_ids)}


def check(hf_dir: pathlib.Path, out_dir: pathlib.Path, texts: list[str] | None = None) -> int:
    """Encode the same text with Hugging Face's tokenizer and with the pack one; any difference fails."""
    from tokenizers import Tokenizer as HFTokenizer

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools"))
    import strata_tokenizer as ST

    hf = HFTokenizer.from_file(str(pathlib.Path(hf_dir) / "tokenizer.json"))
    pack = ST.Tokenizer.from_dir(out_dir)
    bad = 0
    for text in texts or CHECK_TEXTS:
        want = hf.encode(text, add_special_tokens=False).ids
        got = pack.encode(text, parse_special=True)
        if want != got:
            bad += 1
            print(f"MISMATCH {text!r}\n  hf   {want}\n  pack {got}", flush=True)
            continue
        # and back again
        back = pack.decode(got)
        if back != text:
            print(f"ROUND TRIP {text!r} -> {back!r}", flush=True)
    print(f"checked {len(texts or CHECK_TEXTS)} texts: {bad} mismatch(es)", flush=True)
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf", required=True, help="a directory with the model's tokenizer.json")
    ap.add_argument("--out", required=True, help="the pack tokenizer directory to write")
    ap.add_argument("--template", help="chat_template.jinja to copy in (tools/ds4/chat_template.jinja)")
    ap.add_argument("--check", action="store_true", help="encode a corpus with both tokenizers and compare")
    a = ap.parse_args(argv)
    info = convert(pathlib.Path(a.hf), pathlib.Path(a.out), pathlib.Path(a.template) if a.template else None)
    print(f"wrote {a.out}: {info['vocab']} ids, {info['merges']} merges, {info['specials']} special tokens")
    if a.check:
        return 1 if check(pathlib.Path(a.hf), pathlib.Path(a.out)) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
