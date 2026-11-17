"""ascend/tests/test_make_tokenizer.py - the Hugging Face tokenizer into the layout `serve/` reads.

The W8A8 export ships no tokenizer, so one is converted from the V4 export (same vocabulary).  Two things
must hold and are tested here: the conversion writes the four files `tools/strata_tokenizer.Tokenizer.from_dir`
reads, with the special tokens marked as llama.cpp CONTROL tokens (otherwise they are split into
characters, which would break every DeepSeek prompt); and it can never write outside its output directory,
because the first version of this script wrote through a symlink into the shared model directory.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend.make_tokenizer import PRE, convert

try:
    import tokenizers  # noqa: F401
    HAS_TOKENIZERS = True
except Exception:                                       # noqa: BLE001
    HAS_TOKENIZERS = False


def tiny_hf_tokenizer() -> dict:
    """A byte-level BPE of the shape Hugging Face writes, small enough to build by hand."""
    vocab = {
        "<|begin|>": 0, "<|end|>": 1, "<|user|>": 2,
        "h": 3, "e": 4, "l": 5, "o": 6, "he": 7, "ll": 8, "llo": 9, "hello": 10,
        "w": 11, "r": 12, "d": 13, "or": 14, "wor": 15, "ld": 16, "world": 17, " ": 18,
    }
    def added(i: int, content: str) -> dict:
        return {"id": i, "content": content, "single_word": False, "lstrip": False, "rstrip": False,
                "normalized": False, "special": True}

    added = [added(0, "<|begin|>"), added(1, "<|end|>"), added(2, "<|user|>")]
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "normalizer": None,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True,
                          "use_regex": True},
        "post_processor": None,
        "decoder": None,
        "model": {"type": "BPE", "vocab": {k: v for k, v in vocab.items() if v >= 3},
                  # every merge must name a token the vocabulary has, or Hugging Face refuses the file
                  "merges": ["h e", "l l", "ll o", "he llo", "o r", "w or", "l d", "wor ld"]},
        "added_tokens": added,
    }


class TheConversion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.hf = root / "hf"
        self.hf.mkdir()
        (self.hf / "tokenizer.json").write_text(json.dumps(tiny_hf_tokenizer()))
        self.out = root / "pack"
        self.template = root / "chat_template.jinja"
        self.template.write_text("{%- if x %}\uff5cDSML\uff5c{%- endif %}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_the_four_files_serve_reads_are_written(self):
        info = convert(self.hf, self.out, self.template)
        for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json", "chat_template.jinja"):
            self.assertTrue((self.out / name).is_file(), f"{name} is missing")
        self.assertEqual(info["vocab"], 19)
        self.assertEqual(info["specials"], 3)

    def test_the_meta_names_the_deepseek_pre_tokenizer(self):
        convert(self.hf, self.out, self.template)
        meta = json.loads((self.out / "tokenizer.json").read_text())
        self.assertEqual(meta["pre"], PRE)
        self.assertEqual(meta["pre"], "deepseek3")
        self.assertFalse(meta["add_bos_token"])
        self.assertEqual(meta["special_ids"]["<|user|>"], 2)

    def test_every_id_is_named_including_the_added_tokens(self):
        convert(self.hf, self.out, self.template)
        vocab = json.loads((self.out / "vocab.json").read_text())
        ids = sorted(vocab.values())
        self.assertEqual(ids, list(range(19)), "each id 0..n-1 named once, or the reverse table has holes")
        self.assertEqual(vocab["<|begin|>"], 0)
        self.assertEqual(vocab["<|user|>"], 2)
        self.assertEqual(vocab[" "], 18)

    def test_specials_are_marked_control_and_merges_kept_in_order(self):
        convert(self.hf, self.out, self.template)
        types = json.loads((self.out / "token_type.json").read_text())
        self.assertEqual(types[:3], [3, 3, 3], "an added special is a CONTROL token (llama.cpp type 3)")
        self.assertTrue(all(t == 1 for t in types[3:]), "ordinary tokens are type 1")
        merges = (self.out / "merges.txt").read_text().split("\n")
        self.assertEqual(merges[0], "h e", "merges keep their rank order")
        self.assertEqual(len(merges), 8)

    def test_the_template_is_copied_and_is_recognised_as_deepseek(self):
        convert(self.hf, self.out, self.template)
        self.assertIn("\uff5cDSML\uff5c", (self.out / "chat_template.jinja").read_text(),
                      "serve/frontend.py keys DeepSeek's prompt format on this tag")

    def test_a_write_never_follows_a_symlink_out_of_the_output_directory(self):
        """The bug this guard exists for: `out/tokenizer.json` was a symlink into the model directory."""
        victim = Path(self.tmp.name) / "shared.json"
        victim.write_text("ORIGINAL")
        self.out.mkdir()
        (self.out / "tokenizer.json").symlink_to(victim)
        convert(self.hf, self.out, self.template)
        self.assertEqual(victim.read_text(), "ORIGINAL", "the symlink's target must be untouched")
        self.assertFalse((self.out / "tokenizer.json").is_symlink(), "the link is replaced, not written through")
        self.assertEqual(json.loads((self.out / "tokenizer.json").read_text())["pre"], PRE)

    def test_a_tokenizer_that_is_not_bpe_is_refused_before_anything_is_written(self):
        doc = tiny_hf_tokenizer()
        doc["model"] = "WordPiece"
        (self.hf / "tokenizer.json").write_text(json.dumps(doc))
        with self.assertRaises(SystemExit):
            convert(self.hf, self.out, self.template)
        self.assertFalse((self.out / "vocab.json").exists())

    @unittest.skipUnless(HAS_TOKENIZERS, "needs the `tokenizers` package")
    def test_the_pack_tokenizer_reproduces_hugging_face(self):
        from ascend.make_tokenizer import check
        convert(self.hf, self.out, self.template)
        # a single pre-token, so the comparison is about the merges and the byte alphabet rather than
        # about the 19-token fixture's ability to reproduce DeepSeek's pre-tokenizer
        bad = check(self.hf, self.out, ["hello"])
        self.assertEqual(bad, 0)


if __name__ == "__main__":
    unittest.main()
