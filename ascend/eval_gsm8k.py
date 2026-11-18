"""ascend/eval_gsm8k.py - accuracy and speed on a sample of GSM8K, through this engine.

Not a benchmark score: it samples `--n` questions from a GSM8K jsonl, renders each with the *repository's
own* DeepSeek-V4.1 encoder (`serve/ds41_encoding.py`, the vendored official one), decodes greedily and reads
the final number.  It exists to answer two questions about this port - "is it right?" and "how fast is it?" -
on a task with checkable answers, and to compare answers across changes (Engram on/off, a different expert
tier, a different matmul).

    python -m ascend.eval_gsm8k --model /mnt/weights/... --tokenizer <tokenizer.json> \\
        --data gsm8k-test.jsonl --n 10 --seed 0 --expert-devices npu:0,npu:1,npu:2,npu:3

Decoding is greedy and non-thinking by default: at this stage's speed a thinking answer would take tens of
minutes per question, and the point is comparability rather than the best achievable score.  Generation stops
as soon as a final `#### <number>` line appears, so a question costs what its answer costs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .dsv41 import runtime as rt

from .gsm8k_text import FINAL, INSTRUCTION, answer_of, gold_of, load_questions, to_number


def render(tokenizer, question: str, thinking: bool, effort) -> list[int]:
    """The prompt's tokens, through the repository's official DeepSeek-V4.1 encoder."""
    prompt = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from serve.ds41_encoding import encode_messages
        prompt = encode_messages(
            [{"role": "user", "content": INSTRUCTION + question}],
            thinking_mode="thinking" if thinking else "chat",
            reasoning_effort=effort,
        )
    except Exception as e:                                    # noqa: BLE001 - fall back to the minimal form
        print(f"  (falling back to the minimal chat form: {e})", file=sys.stderr)

    if prompt is None:
        prompt = "<｜begin▁of▁sentence｜><｜User｜>" + INSTRUCTION + question + "<｜Assistant｜></think>"
    return tokenizer.encode(prompt, add_special_tokens=False).ids


def generate(model, tokenizer, state, ids: list[int], max_tokens: int, stop_on_final: bool = True):
    """Greedy decode from `ids`, stopping on eos or on a final `#### <number>` line."""
    t0 = time.perf_counter()
    logits = model.forward(ids, state, logits_last_only=True)
    prefill_s = time.perf_counter() - t0
    eos = 1
    try:
        got = tokenizer.token_to_id("<｜end▁of▁sentence｜>")
        if got is not None:
            eos = int(got)
    except Exception:                                         # noqa: BLE001
        pass
    out: list[int] = []
    t0 = time.perf_counter()
    for _ in range(max_tokens):
        tok = int(logits[0].argmax())
        if tok == eos:
            break
        out.append(tok)
        text = tokenizer.decode(out, skip_special_tokens=False)
        if stop_on_final and FINAL.search(text) and "\n" in text[FINAL.search(text).end():]:
            break
        logits = model.forward([tok], state, logits_last_only=True)
    decode_s = time.perf_counter() - t0
    return tokenizer.decode(out, skip_special_tokens=False), prefill_s, decode_s, len(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--expert-devices", default="")
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--engram-tokenizer", default="",
                    help="enables Engram; the same value as --tokenizer is right")
    ap.add_argument("--graphs", action="store_true",
                    help="capture the one-token decode path as per-layer NPU graphs (the serve engine's "
                         "fast path).  Every question gets a fresh State, and the graphs belong to the "
                         "State, so this re-captures the 80 graphs per question; that capture time is "
                         "included in the decode seconds reported.")
    ap.add_argument("--graph-verify", action="store_true",
                    help="run the did-it-capture guard on every graph (slow: 4 eager runs per graph)")
    ap.add_argument("--thinking", action="store_true", help="thinking mode (far slower)")
    ap.add_argument("--effort", default=None)
    ap.add_argument("--out", help="write the per-question report here as JSON")
    ap.add_argument("--show", action="store_true", help="print each answer in full")
    a = ap.parse_args()
    t_start = time.perf_counter()

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(a.tokenizer)
    questions = load_questions(Path(a.data), a.n, a.seed)
    print(f"{len(questions)} questions from {a.data} (seed {a.seed})")

    run = rt.build(model=a.model, device=a.device, dtype=a.dtype,
                   expert_devices=[d for d in a.expert_devices.split(",") if d] or None,
                   expert_slots=a.expert_slots, engram_tokenizer=a.engram_tokenizer or None,
                   graphs=a.graphs, graph_verify=a.graph_verify)
    print(f"engine: {run.device}, engram {'on' if run.engram else 'off'}, expert tier {run.expert_slots} slots"
          + (f" over {run.experts.layer_devices[0]}..{run.experts.layer_devices[-1]}"
             if getattr(run.experts, 'layer_devices', None) else "")
          + (f", decode graphs {'on' if run.model.graphs else 'off'}" if a.graphs else ""))

    rows, right, total_decode, total_tokens = [], 0, 0.0, 0
    for i, q in enumerate(questions, 1):
        ids = render(tokenizer, q["question"], a.thinking, a.effort)
        state = run.state(a.max_context)
        text, prefill_s, decode_s, n_tok = generate(run.model, tokenizer, state, ids, a.max_tokens)
        got, want = answer_of(text), gold_of(q)
        ok = got is not None and want is not None and abs(got - want) < 1e-4
        right += ok
        total_decode += decode_s
        total_tokens += n_tok
        rows.append({"i": i, "question": q["question"], "gold": want, "got": got, "ok": ok,
                     "prompt_tokens": len(ids), "tokens": n_tok,
                     "prefill_s": round(prefill_s, 2), "decode_s": round(decode_s, 2),
                     "text": text})
        mark = "ok " if ok else "BAD"
        tps = n_tok / decode_s if decode_s > 0 else 0.0
        print(f"  [{mark}] {i:2d}: gold {want} got {got} | {len(ids)} prompt tok, {n_tok} gen tok, "
              f"{decode_s:.0f}s decode ({tps:.2f} tok/s), {prefill_s:.0f}s prefill", flush=True)
        if a.show or not ok:
            print("       " + text.replace("\n", "\n       ")[-1200:], flush=True)

    acc = right / max(1, len(questions))
    tps = total_tokens / total_decode if total_decode > 0 else 0.0
    elapsed_s = time.perf_counter() - t_start
    captured = getattr(getattr(run.model, "_decode_graphs", None), "captured", None)
    print(f"\naccuracy: {right}/{len(questions)} = {acc:.0%}")
    print(f"speed   : {total_tokens} tokens in {total_decode:.0f}s decode = {tps:.2f} tok/s; "
          f"{sum(r['prefill_s'] for r in rows) / len(rows):.1f}s mean prefill")
    print(f"wall    : {elapsed_s:.0f}s from build to last answer"
          + (f"; {captured} graphs live, graphs {'still on' if run.model.graphs else 'turned OFF'}"
             if a.graphs else ""))
    if a.out:
        Path(a.out).write_text(json.dumps(
            {"model": a.model, "n": len(questions), "seed": a.seed, "accuracy": acc, "tok_s": tps,
             "graphs": a.graphs, "graph_captures": captured, "graphs_still_on": bool(run.model.graphs),
             "engram": bool(run.engram), "elapsed_s": round(elapsed_s, 1), "rows": rows}, indent=1))
        print(f"report: {a.out}")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
