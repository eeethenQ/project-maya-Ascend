"""ascend/run_dsv41.py - DeepSeek-V4.1-Flash W8A8 on an Ascend 910B, from a text prompt.

    python -m ascend.run_dsv41 --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \\
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \\
        --chat "The capital of France is" --gen 16 --expert-slots 200

This is the engine's first real run: the 40-layer backbone with per-request state, the routed experts
streamed from the checkpoint's page cache into whatever HBM is left, and greedy decoding.  It is the
same code the serve engine will drive; only the sampling loop is local here.

Printed, in this order: what the model directory looks like, what stays resident and what it costs to get
there, the prompt's tokens, then one line per generated token with the per-token latency.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from .dsv41.checkpoint import Checkpoint
from .dsv41.config import TextConfig, build_layer_plan
from .dsv41.model import Dsv41, ExpertStreamer, State, Weights

GIB = 1 << 30


def pick_device(requested: str, share: float) -> str:
    """The first card of `ASCEND_RT_VISIBLE_DEVICES`, inside this process's share of its HBM."""
    import torch_npu  # noqa: F401 - registers the backend

    visible = [int(x) for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    if not visible:
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES names no card")
    index = 0 if requested in ("", "npu", "npu:0") else int(requested.split(":")[-1])
    torch.npu.set_device(index)
    torch.npu.set_per_process_memory_fraction(share, index)
    return f"npu:{index}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="the W8A8 checkpoint directory")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json (this checkpoint ships none)")
    ap.add_argument("--chat", help="a user message, rendered as V4.1's non-thinking prompt")
    ap.add_argument("--text", help="raw text, tokenized as is")
    ap.add_argument("--ids", help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=8, help="greedy tokens to generate")
    ap.add_argument("--max-context", type=int, default=4096)
    ap.add_argument("--expert-slots", type=int, default=0,
                    help="routed experts kept resident on the card (33.8 MiB each)")
    ap.add_argument("--hbm-share", type=float, default=0.5, help="this process's fraction of one card")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="", help="cards to spread the expert tier over")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--engram-tokenizer", default="",
                    help="enables Engram (hash n-grams through this Hugging Face tokenizer.json)")
    ap.add_argument("--no-experts", action="store_true", help="skip the routed branch (a shape check)")
    a = ap.parse_args()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    device = pick_device(a.device, a.hbm_share)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[a.dtype]

    model_dir = Path(a.model)
    cfg = TextConfig.from_json(model_dir / "config.json")
    print(f"model: {cfg.num_hidden_layers} layers, hidden {cfg.hidden_size}, "
          f"{cfg.n_routed_experts} experts x top-{cfg.num_experts_per_tok}", flush=True)
    t0 = time.time()
    ckpt = Checkpoint(model_dir)
    print(f"index: {ckpt.summary()}  ({time.time() - t0:.1f}s)", flush=True)

    from .dsv41 import runtime as rt
    t0 = time.time()
    run = rt.build(model=str(model_dir), device=device, dtype=a.dtype,
                   expert_slots=a.expert_slots,
                   expert_devices=[d for d in a.expert_devices.split(",") if d] or None,
                   engram_tokenizer=a.engram_tokenizer or None,
                   no_experts=a.no_experts)
    weights, net, src = run.weights, run.model, run.experts
    load_s = time.time() - t0
    resident = weights.resident_bytes()
    print(f"resident: {resident / GIB:.2f} GiB on {device} in {load_s:.1f}s "
          f"({resident / load_s / 1e9:.2f} GB/s)", flush=True)
    per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
    routed = cfg.num_hidden_layers * cfg.num_experts_per_tok
    print(f"expert tier: {run.expert_slots} slots"
          + (f" over {len(set(src.layer_devices))} cards" if getattr(src, 'layer_devices', None) else "")
          + f"; a decode step routes {routed} experts = {routed * per_expert / 1e9:.2f} GB", flush=True)
    print(f"engram: {'on' if run.engram else 'off'}", flush=True)

    state = run.state(a.max_context)

    if a.ids:
        ids = [int(x) for x in a.ids.split(",")]
    elif a.chat:
        prompt = "<｜begin▁of▁sentence｜><｜User｜>" + a.chat + "<｜Assistant｜></think>"
        ids = tok.encode(prompt, add_special_tokens=False).ids
    else:
        ids = tok.encode(a.text or "The capital of France is", add_special_tokens=False).ids
    print(f"prompt: {len(ids)} tokens {ids[:40]}", flush=True)

    t0 = time.time()
    logits = net.forward(ids, state)
    prefill_s = time.time() - t0
    print(f"prefill: {len(ids)} tokens in {prefill_s:.2f}s ({len(ids) / prefill_s:.2f} tok/s)", flush=True)
    _report_top(tok, logits, 8)

    out = list(ids)
    eos = tok.token_to_id("<｜end▁of▁sentence｜>")
    for step in range(a.gen):
        nxt = int(logits[0].argmax())
        out.append(nxt)
        text = tok.decode([nxt], skip_special_tokens=False)
        t0 = time.time()
        if nxt == eos:
            print(f"gen: {step + 1}: {text!r} (eos)", flush=True)
            break
        logits = net.forward([nxt], state)
        dt = time.time() - t0
        print(f"gen: {step + 1}: {text!r}  {dt * 1e3:.0f} ms ({1 / dt:.2f} tok/s)", flush=True)
    print("--- text ---", flush=True)
    print(tok.decode(out[len(ids):], skip_special_tokens=False), flush=True)
    st = src.stats()
    print(f"experts: {st.get('expert_hits', 0)} hits, {st.get('expert_misses', 0)} misses, "
          f"{st.get('expert_bytes', 0) / 1e9:.1f} GB read, {st.get('expert_slots', 0)} resident", flush=True)
    return 0


def _report_top(tok, logits: torch.Tensor, n: int) -> None:
    lp = torch.log_softmax(logits[0].float(), -1)
    top = lp.topk(n)
    print("next:", [(tok.decode([int(i)]), round(float(v), 3))
                    for v, i in zip(top.values.tolist(), top.indices.tolist())], flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
