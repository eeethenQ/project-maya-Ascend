"""ascend/gsm8k_text.py - reading GSM8K questions, gold answers and a model's answer out of text.

Separate from `eval_gsm8k` (which imports the engine and therefore torch) so the judging logic can be
tested in CI without a card.  A wrong extraction here would look like a model error, so it is the part of
the harness worth pinning down.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")
FINAL = re.compile(r"####\s*(-?[\d,]*\.?\d+)")

INSTRUCTION = ("Solve the problem. Think it through, then write the final answer on its own last line as:\n"
               "#### <number>\n\n")


def to_number(text: str) -> float | None:
    try:
        return float(str(text).replace(",", "").strip().rstrip("."))
    except (ValueError, AttributeError):
        return None


def gold_of(row: dict) -> float | None:
    """The dataset's answer: what follows `####`, never the calculator noise in `<<...>>`."""
    m = FINAL.search(row.get("answer", ""))
    return to_number(m.group(1)) if m else None


def answer_of(text: str) -> float | None:
    """The model's answer: after the last `####`, else the last number in the text."""
    marks = FINAL.findall(text or "")
    if marks:
        got = to_number(marks[-1])
        if got is not None:
            return got
    for candidate in reversed(NUMBER.findall((text or "").replace(",", ""))):
        got = to_number(candidate)
        if got is not None:
            return got
    return None


def load_questions(path: str | Path, n: int, seed: int) -> list[dict]:
    """`n` questions drawn deterministically from a GSM8K jsonl, keeping only rows with a gold answer."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [r for r in rows if "####" in r.get("answer", "")]
    random.Random(seed).shuffle(rows)
    return rows[:n]
