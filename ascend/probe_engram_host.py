"""ascend/probe_engram_host.py - what Engram actually costs a decode token: the host gather and the wall.

The operator-list entry for Engram said "~0.2-0.3 ms of device time, and its E8M0 dequantisation is already
host-side numpy, so a device kernel would add work".  The estimate priced the arithmetic and not what the
gather *does*: `EngramTable.rows_at` touches 24 rows of a 94 GiB table through an `mmap`, at 24 random
offsets a layer - and `/mnt/weights` is mounted with a 4 MiB readahead window
(`/sys/block/sda/queue/read_ahead_kb = 4096`), so each 4 KiB page fault pulls 4 MiB.  This probe measures
that on the real checkpoint, in one process:

1. **the host gather isolated**, 24 random rows a draw, and the same mapping again after
   ``madvise(MADV_RANDOM)`` - the read amplification is visible in ``/proc/self/io`` as well as in the time;
2. **the host gather in situ**, through the captured decode: `Engram.gather` / `Engram.embedding` wrapped
   with `time.perf_counter` for whole steps, once with the mount's readahead and once with
   `MADV_RANDOM` (a fresh State and a fresh capture an arm, the same prompt and token stream);
3. **the captured device block alone** (`apply_device` replayed inside its own graph, with the
   did-it-capture guard on): the part a Triton kernel could touch, and its share of the step.  On the
   real model (`--device-only`) this is 0.66 ms a layer; under `bench_profile --graphs --msprof` the
   same block is 2 `MatMulV2` + ~24 small ops a layer;
4. **the wall with Engram on and off**, paired in this process on a fresh State and a fresh capture;
5. **`--dog`**, the known probe: `p(" dog")` after `"The quick brown fox jumps over the lazy"`.

Measured on physical npu:2 (2026-11-24, the array shared with other jobs): gather 662.7 ms/token with
the mount's readahead, 237.4 ms with `MADV_RANDOM` in situ; 0.35 ms a layer for the dequantisation once
the pages are in; device block 1.32 ms/token (2 layers), of which the two matmuls are 1.11.

    ASCEND_RT_VISIBLE_DEVICES=2,3 python -m ascend.probe_engram_host --model /mnt/weights/... \\
        --engram-tokenizer /mnt/weights/.../tokenizer.json --expert-devices npu:0,npu:1
"""
from __future__ import annotations

import argparse
import json
import mmap as _mmap
import statistics
import time

import numpy as np
import torch

from .dsv41 import runtime as rt
from .dsv41.graph_runner import GraphRunner


def set_readahead(eng, random: bool) -> None:
    """Put the table mappings back into the fault policy under test (`random=False` = the old mount default).

    `EngramTable.rows_at` now asks for `MADV_RANDOM` itself (it is the right policy for a random gather), so
    an A/B has to undo that on the private mapping rather than rely on the default.
    """
    flag = _mmap.MADV_RANDOM if random else _mmap.MADV_NORMAL
    for table in eng.tables.values():
        table.shard.mmap().madvise(flag)
        table.random_reads = random
        table._policy_set = True


def _read_bytes() -> int:
    with open("/proc/self/io", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("read_bytes"):
                return int(line.split()[1])
    return 0


def isolated(eng, draws: int) -> dict:
    """Cold and warm `rows_at`, before and after MADV_RANDOM, with the bytes the kernel actually reads."""
    rng = np.random.default_rng(0)
    table = eng.tables[min(eng.tables)]
    out: dict[str, object] = {"layer": table.layer, "draws": draws, "cols": eng.cols}

    def draw() -> tuple[float, int]:
        rows = rng.integers(0, table.rows, size=(1, eng.cols))
        before = _read_bytes()
        t0 = time.perf_counter()
        eng.gather(table.layer, rows)
        return (time.perf_counter() - t0) * 1e3, _read_bytes() - before

    def sample(label: str, n: int) -> None:
        got = [draw() for _ in range(n)]
        warm = [draw() for _ in range(2)]           # the same rows a moment later are in the page cache
        del warm
        ms = [g[0] for g in got]
        read = [g[1] for g in got]
        out[label] = {"median_ms": round(statistics.median(ms), 2),
                      "p90_ms": round(sorted(ms)[int(0.9 * (len(ms) - 1))], 2),
                      "median_read_kb": round(statistics.median(read) / 1024, 1)}

    set_readahead(eng, False)
    sample("mmap_default", draws)
    set_readahead(eng, True)                                # a host-side mapping flag; no graph involved
    sample("madvise_random", draws)
    # the same rows twice, to separate the fault from the dequantisation
    rows = rng.integers(0, table.rows, size=(1, eng.cols))
    t0 = time.perf_counter(); eng.gather(table.layer, rows); cold = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter(); eng.gather(table.layer, rows); warm = (time.perf_counter() - t0) * 1e3
    out["cold_then_warm_ms"] = [round(cold, 2), round(warm, 3)]
    return out


def device_block(eng, mm, dev: str, layer: int, iters: int = 80) -> dict:
    """`apply_device` alone, captured and replayed: the op sequence an Engram layer contributes to g_pre."""
    dev = str(dev)
    hc, d = eng.cfg.hc_mult, eng.cfg.hidden_size
    eng.warm(dev)
    rows = eng.hashes([3, 5, 7, 11])[-1:, eng.slot[layer]]
    e = eng.embedding(layer, eng.gather(layer, rows))
    e32 = torch.zeros(1, eng.cols * eng.head_dim, dtype=torch.float32, device=dev)
    e32.copy_(e.view_as(e32))
    h = (torch.randn(1, hc, d, device=dev) * 0.02).float()
    runner = GraphRunner(dev)

    def body():
        return eng.apply_device(layer, h, e32, mm)

    runner.capture("engram_block", body, inputs={"h": h, "e": e32}, verify=True)
    runner.replay("engram_block", sync=True)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        runner.replay("engram_block")
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) / iters * 1e3
    runner.close()
    return {"layer": layer, "captured_verify": True, "ms_per_call": round(ms, 4)}


def in_situ(run, prompt: list[int], tokens: list[int], readahead: bool | None = None) -> dict:
    """Prefill + capture, then whole decode steps with the host Engram block timed.

    ``readahead`` chooses the mapping policy for the whole arm: ``False`` is the mount's 4 MiB window (the
    tree's behaviour before `MADV_RANDOM`), ``True`` the random policy the gather now asks for.
    """
    model, eng = run.model, run.engram
    acc: dict[str, float] = {"gather_ms": 0.0, "embedding_ms": 0.0, "calls": 0.0}
    original_gather, original_embedding = eng.gather, eng.embedding

    def gather(layer, rows):
        t0 = time.perf_counter()
        try:
            return original_gather(layer, rows)
        finally:
            acc["gather_ms"] += (time.perf_counter() - t0) * 1e3
            acc["calls"] += 1

    def embedding(layer, gathered):
        t0 = time.perf_counter()
        try:
            return original_embedding(layer, gathered)
        finally:
            acc["embedding_ms"] += (time.perf_counter() - t0) * 1e3

    eng.gather, eng.embedding = gather, embedding
    if readahead is not None:
        set_readahead(eng, readahead)
    state = run.state(2048)
    model.forward(prompt, state, compute_logits=True)          # capture happens on the first decode step
    rows, wall = [], []
    try:
        for k, tok in enumerate(tokens):
            before = dict(acc)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            logits = model.forward([tok], state, logits_last_only=True)
            tok = int(logits[0].argmax())
            torch.npu.synchronize()
            wall.append((time.perf_counter() - t0) * 1e3)
            if k:                                               # step 0 is the capture
                rows.append({"step": k, **{name: round(acc[name] - before[name], 2) for name in acc}})
    finally:
        eng.gather, eng.embedding = original_gather, original_embedding
    body = [r for r in rows if r["calls"]]
    return {"wall_ms": [round(w, 1) for w in wall], "per_step": body,
            "gather_median_ms": statistics.median(r["gather_ms"] for r in body) if body else None,
            "embedding_median_ms": statistics.median(r["embedding_ms"] for r in body) if body else None,
            "wall_median_ms": round(statistics.median(wall[1:]), 1) if len(wall) > 1 else None,
            "graphs": None if model._decode_graphs is None else model._decode_graphs.describe()}


def device_only(a) -> int:
    """The captured `apply_device` block on its own: no model, no tier, just the two layers' projections."""
    from pathlib import Path

    from .dsv41.checkpoint import Checkpoint
    from .dsv41.config import TextConfig
    from .dsv41.engram import Engram

    ckpt = Checkpoint(a.model)
    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    eng = Engram(ckpt, cfg, a.engram_tokenizer, a.device, torch.bfloat16)
    for layer in cfg.engram_layer_ids:
        base = f"layers.{layer}.engram."
        wkv = torch.from_numpy(np.ascontiguousarray(ckpt.numpy(base + "wkv.weight"))).to(a.device)
        q = torch.from_numpy(np.ascontiguousarray(ckpt.numpy(base + "q_weight"))).float().to(a.device)
        k = torch.from_numpy(np.ascontiguousarray(ckpt.numpy(base + "k_weight"))).float().to(a.device)
        eng.attach(layer, wkv, q, k)
    mm = lambda x, w: torch.nn.functional.linear(x.to(w.dtype), w).float()   # noqa: E731 - Dsv41.mm
    out = {str(lid): device_block(eng, mm, a.device, lid, a.iters) for lid in sorted(eng.tables)}
    print(json.dumps({"device_block": out}, indent=1), flush=True)
    print(f"engram block (2 matmuls + gate) device time, both layers: "
          f"{sum(v['ms_per_call'] for v in out.values()):.3f} ms/token", flush=True)
    ckpt.close()
    return 0


def dog_probe(run, tokenizer_path: str, prompt: str = "The quick brown fox jumps over the lazy",
              token: str = " dog") -> dict:
    """The known-correct probe: p(" dog") after that prompt, through the captured decode.

    The prompt is prefilled up to its last token and that token is then fed as a one-token step, so the
    logits come off the captured path (`g_pre`/`g_post` with Engram inside `g_pre`) - the path the
    documented p~1.000 refers to.  A whole-prompt chunk is also reported: the eager chunk uses the host
    numpy mHC branch where the capture uses the device one, and that difference is worth ~7 points of
    p(" dog") here, which is why the probe is quoted from the decode.
    """
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(tokenizer_path))
    ids = tok.encode(prompt).ids
    want = tok.encode(token).ids
    out: dict[str, object] = {"prompt_tokens": len(ids), "token_ids": want}

    def read(logits) -> dict:
        probs = torch.softmax(logits.float().reshape(-1), dim=-1)
        best = int(torch.argmax(probs))
        return {"p_token": round(float(probs[want[0]]), 6), "argmax_id": best,
                "argmax_text": tok.decode([best]), "argmax_p": round(float(probs[best]), 6)}

    state = run.state(2048)                      # at least the KV window (128), or State refuses
    out["chunk"] = read(run.model.forward(ids, state))
    state = run.state(2048)
    run.model.forward(ids[:-1], state, compute_logits=False)
    out["captured_decode"] = read(run.model.forward([ids[-1]], state))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--engram-tokenizer", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0,npu:1")
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--off-steps", type=int, default=4)
    ap.add_argument("--draws", type=int, default=8)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--dog", action="store_true", help="the known p(\" dog\") probe after the arms")
    ap.add_argument("--dog-only", action="store_true",
                    help="only the known p(\" dog\") probe: one prefill, no tier capture, for a quick check")
    ap.add_argument("--device-only", action="store_true",
                    help="only the captured apply_device block, no model and no tier (fast; for msprof)")
    a = ap.parse_args()
    import torch_npu  # noqa: F401 - registers the backend
    if a.device_only:
        return device_only(a)

    devices = [d for d in a.expert_devices.split(",") if d] or None
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", expert_devices=devices,
                   graphs=True, engram_tokenizer=a.engram_tokenizer)
    print(f"tier: {run.expert_slots} slots over {devices}", flush=True)
    report: dict[str, object] = {}
    try:
        if a.dog_only:
            report["dog"] = dog_probe(run, a.engram_tokenizer)
            print(json.dumps({"dog": report["dog"]}, indent=1), flush=True)
            return 0
        if a.draws:
            report["isolated_host"] = isolated(run.engram, a.draws)
            print(json.dumps({"isolated_host": report["isolated_host"]}, indent=1), flush=True)
        prompt = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
        tokens = [(100 + i * 13) % 40000 for i in range(max(a.steps, a.off_steps) + 1)]
        report["engram_on_default"] = in_situ(run, prompt, tokens[:a.steps + 1], readahead=False)
        print(json.dumps({"engram_on_default": report["engram_on_default"]}, indent=1), flush=True)
        report["engram_on_madvise"] = in_situ(run, prompt, tokens[:a.steps + 1], readahead=True)
        print(json.dumps({"engram_on_madvise": report["engram_on_madvise"]}, indent=1), flush=True)
        report["device_block"] = {str(lid): device_block(run.engram, run.model.mm, run.device, lid)
                                  for lid in sorted(run.engram.tables)}
        print(json.dumps({"device_block": report["device_block"]}, indent=1), flush=True)
        saved, run.model.engram = run.model.engram, None
        try:
            report["engram_off"] = in_situ(run, prompt, tokens[:a.off_steps + 1])
        finally:
            run.model.engram = saved
        print(json.dumps({"engram_off": report["engram_off"]}, indent=1), flush=True)
        on = report["engram_on_madvise"]["wall_median_ms"]
        off = report["engram_off"]["wall_median_ms"]
        if on and off:
            print(f"\nwall: engram on {on} ms/token, off {off} ms/token, delta {on - off:+.1f} ms "
                  f"({on / off:.3f}x)")
        for arm in ("engram_on_default", "engram_on_madvise"):
            print(f"host gather in situ ({arm}): {report[arm]['gather_median_ms']} ms/token, "
                  f"embedding (H2D) {report[arm]['embedding_median_ms']} ms/token, "
                  f"wall {report[arm]['wall_median_ms']} ms/token")
        print("wall engram off: " + str(report["engram_off"]["wall_median_ms"]) + " ms/token")
        if a.dog:
            report["dog"] = dog_probe(run, a.engram_tokenizer)
            print(json.dumps({"dog": report["dog"]}, indent=1), flush=True)
    finally:
        run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
