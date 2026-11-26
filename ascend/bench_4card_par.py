"""ascend/bench_4card_par.py - the N-way 2 000-token arm that `bench_serve.py`'s CLI cannot ask for.

`bench_serve.py`'s default scenarios call `concurrent_prefill_decode(..., a.eos)` and take that
function's default ``n=4``; ``--parallel`` only reaches the serve-engine arms.  So `--only concurrent
--parallel 8` runs four conversations under an eight label (measured: ``"requests": 4`` in the JSON).
This driver builds the same eager engine `rt.build` builds and calls the same
`bench_serve.concurrent_prefill_decode` with ``n`` set, so the measurement code, the counters and the
fields are unchanged - only the thread count the CLI ignored.  Nothing else from bench_serve is
modified.

    python -m ascend.bench_4card_par --model ... --tokenizer ... --cards npu:0,npu:1,npu:2,npu:3 \\
        --parallel 8 --prompt-tokens 2000 --gen 32 --max-context 2048 \\
        --hbm-share 0.5 --expert-reserve-gib 13 --out ascend/results/serve-4card-par8-2k.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .bench_serve import (GIB, _io_counters, _io_delta, _release, concurrent_prefill_decode,
                          make_prompt)
from .dsv41 import runtime as rt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--cards", default="npu:0,npu:1,npu:2,npu:3")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-reserve-gib", type=float, default=13.0)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--eos", type=int, default=1)
    ap.add_argument("--prompt-source", default="README.md")
    ap.add_argument("--label", default="")
    ap.add_argument("--jsonl", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    print(f"cards: logical {cards} (layers on {a.device}); ASCEND_RT_VISIBLE_DEVICES={visible!r}",
          flush=True)
    print(f"budget: hbm_share {a.hbm_share}, expert reserve {a.expert_reserve_gib} GiB on {a.device}, "
          f"max_context {a.max_context}, prompt {a.prompt_tokens}, gen {a.gen}, parallel {a.parallel}",
          flush=True)

    t_run = time.perf_counter()
    t0 = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_devices=cards, expert_reserve_gib=a.expert_reserve_gib,
                   expert_batch=a.expert_batch, io_threads=a.io_threads, graphs=False)
    build_s = time.perf_counter() - t0
    per_expert = 3 * run.cfg.moe_intermediate_size * run.cfg.hidden_size
    print(f"engine: resident {run.weights.resident_bytes() / GIB:.2f} GiB, tier {run.expert_slots} slots "
          f"({run.expert_slots * per_expert / GIB:.2f} GiB) over "
          f"{sorted(set(run.experts.layer_devices or [run.device]))}, built in {build_s:.0f}s",
          flush=True)
    ids = make_prompt(tok, a.prompt_tokens, Path(a.prompt_source))
    print(f"prompt: {len(ids)} tokens from {a.prompt_source}, {a.parallel} conversation(s)", flush=True)

    tier_slots = run.expert_slots
    io0 = _io_counters()
    label = a.label or f"concurrent-2k-n{a.parallel}"
    try:
        rec = concurrent_prefill_decode(run, ids, a.gen, a.max_context, devices, label, a.eos,
                                        n=a.parallel)
    finally:
        io = _io_delta(io0, _io_counters())
        run.model.release_graphs()
        run.close()
        _release(devices)
    rec.update({"cards": cards, "device": a.device, "tier_slots": tier_slots, "io": io})
    summary = {"run_s": round(time.perf_counter() - t_run, 1), "cards": cards, "visible": visible,
               "device": a.device, "mode": "concurrent", "parallel": a.parallel,
               "hbm_share": a.hbm_share, "expert_reserve_gib": a.expert_reserve_gib,
               "max_context": a.max_context, "prompt_tokens": a.prompt_tokens, "gen": a.gen,
               "scenarios": 1, "graph": False}
    if a.jsonl:
        Path(a.jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(a.jsonl, "a") as f:
            f.write(json.dumps(rec) + "\n")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "records": [rec]}, indent=1))
        print(f"report: {a.out}", flush=True)
    print(f"\n{label:16s} wall {rec['wall_s']:.1f}s TTFT {rec['ttft_s']:.1f}s "
          f"decode {rec['decode_tok_s']:.2f} tok/s hit {100 * rec['tier']['hit_rate']:.1f}% "
          f"read {rec['tier']['gb_read']:.1f} GB io {io['read_gb']:.1f}/{io['rchar_gb']:.1f} GB "
          f"peak {rec['peak_gib']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
