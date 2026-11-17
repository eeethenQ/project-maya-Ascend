"""ascend/tests/test_eval_gsm8k.py - reading answers and gold values out of GSM8K text.

The harness's judgement is only as good as these two regexes, and a wrong extraction would look like a
model error.  Pure functions, so they run in CI without torch or a card.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ascend.gsm8k_text import answer_of, gold_of, load_questions, to_number

GSM = [
    {"question": "Janet's ducks", "answer": "16 - 3 - 4 = <<16-3-4=9>>9\n9 * 2 = <<9*2=18>>18\n#### 18"},
    {"question": "A robe", "answer": "2/2 = <<2/2=1>>1\n2 + 1 = <<2+1=3>>3\n#### 3"},
    {"question": "Negative", "answer": "#### -12"},
    {"question": "Thousands", "answer": "#### 1,250"},
    {"question": "Decimal", "answer": "#### 3.5"},
    {"question": "No marker", "answer": "the answer is 7"},
]


class TheGold(unittest.TestCase):
    def test_the_marker_wins_over_the_calculator_noise(self):
        # `<<16-3-4=9>>` also contains numbers; only what follows #### is the answer
        self.assertEqual(gold_of(GSM[0]), 18.0)
        self.assertEqual(gold_of(GSM[1]), 3.0)

    def test_signs_commas_and_decimals(self):
        self.assertEqual(gold_of(GSM[2]), -12.0)
        self.assertEqual(gold_of(GSM[3]), 1250.0)
        self.assertEqual(gold_of(GSM[4]), 3.5)

    def test_a_missing_marker_is_no_gold(self):
        self.assertIsNone(gold_of(GSM[5]))

    def test_to_number_rejects_words(self):
        self.assertIsNone(to_number("eighteen"))
        self.assertEqual(to_number(" 18. "), 18.0)


class TheAnswer(unittest.TestCase):
    def test_the_last_marker_wins(self):
        self.assertEqual(answer_of("so 3+4=7\n#### 7\nand finally\n#### 8"), 8.0)

    def test_a_bare_final_number_is_accepted(self):
        self.assertEqual(answer_of("She makes 18 dollars."), 18.0)
        self.assertEqual(answer_of("The answer is -12"), -12.0)

    def test_commas_and_decimals(self):
        self.assertEqual(answer_of("#### 1,250"), 1250.0)
        self.assertEqual(answer_of("#### 3.5"), 3.5)

    def test_no_number_at_all(self):
        self.assertIsNone(answer_of("I cannot solve this."))


class TheSample(unittest.TestCase):
    def test_the_sample_is_deterministic_and_skips_answers_without_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in GSM * 4) + "\n")
            a = load_questions(path, 5, seed=0)
            b = load_questions(path, 5, seed=0)
            c = load_questions(path, 5, seed=1)
            self.assertEqual(len(a), 5)
            self.assertEqual([r["question"] for r in a], [r["question"] for r in b])
            self.assertNotEqual([r["question"] for r in a], [r["question"] for r in c])
            self.assertTrue(all("####" in r["answer"] for r in a), "rows without a gold are dropped")

    def test_asking_for_more_than_there_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gsm8k.jsonl"
            path.write_text(json.dumps(GSM[0]) + "\n")
            self.assertEqual(len(load_questions(path, 10, seed=0)), 1)


if __name__ == "__main__":
    unittest.main()
