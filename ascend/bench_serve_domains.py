"""ascend/bench_serve_domains.py - N conversations through the **serve path**, graphs on against off.

This is the serve-path counterpart of ``bench_4card_graphs.py``: it builds a real `serve_engine.Engine`
(``--parallel N``, ``--graphs on|off``), starts its scheduler workers, and submits N requests at the same
instant exactly as `serve/server.py` does (``GEN <max_new> rid=<id> ...``).  The default is one shared
2 000-token prompt, the direct-drive arm's workload (``--distinct-prompts`` gives each conversation its own
cut of the document, a KV/contamination check).  Every timing field
comes from the protocol's own lines: TTFT is the first ``T`` line minus the submit, wall is the last
``DONE``, decode is tokens over the decode window (see `bench_serve._serve_rec`).  No model call in this
script; the engine is the thing under test.

The capture question is answered from the model's own registry after the arm: how many conversations hold
a live domain (``model._graph_domains``), each domain's graph count, and each slot's ``graph_steps`` (the
one-token steps it ran while its own domain was alive, which the engine counts).  A conversation past the
bound is also in the engine log, naming the bound and the per-domain cost.

One arm per process, because the two builds do not fit together (each engine holds ~15 GiB of weights):
interleave the arms from the shell, ON, OFF, ON, OFF, ... - an eager arm drifts by seconds between runs.

    # graphs on, four 2 000-token conversations, four cards, the engine's 0.5 HBM share
    python3 -m ascend.bench_serve_domains --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \\
        --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \\
        --cards npu:0,npu:1,npu:2,npu:3 --graphs on --parallel 4 --prompt-tokens 2000 --gen 32 \\
        --max-context 2048 --hbm-share 0.5 --expert-reserve-gib 13 \\
        --out ascend/results/serve-domain-par4-2k-on-1.json

    # the paired control
    ... --graphs off --out ascend/results/serve-domain-par4-2k-off-1.json

    # a distinct-prompt arm (the contamination check), still through the serve path
    ... --graphs on --parallel 4 --prompt-tokens 2000 --gen 32 --distinct-prompts --out ...

    # the bound, past four: eight short conversations, four capture and four run eagerly (with the log)
    ... --graphs on --parallel 8 --prompt-tokens 8 --gen 8 --max-context 2048 \\
        --out ascend/results/serve-domain-par8-short-on.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

from . import serve_engine
from .bench_serve import (GIB, _allocated_gib, _io_counters, _io_delta, _peak_gib, _release, _reset_peak,
                          _tier, _tier_delta, distinct_prompts, make_prompt, serve_parallel)

# a fixed line the shell/report can grep: the arm is done and this is what it measured
RESULT_PREFIX = "SERVE-DOMAINS"


def _engine(a, cards: list[str], graphs: bool):
    """Build the engine the server would start (all the arguments `Engine` reads, nothing else)."""
    ns = argparse.Namespace(
        model=a.model, tokenizer=a.tokenizer, device=a.device, dtype="bfloat16",
        hbm_share=a.hbm_share, expert_slots=None, io_threads=a.io_threads,
        expert_batch=a.expert_batch, expert_reserve_gib=a.expert_reserve_gib,
        expert_devices=",".join(cards), engram_tokenizer="", verbose=False,
        max_context=a.max_context, prefill_chunk=a.prefill_chunk, serve=True, graphs=graphs,
        graph_verify=False, graph_max_conversations=a.graph_max_conversations, parallel=a.parallel,
        parallel_streams=a.parallel_streams, prefix_reuse="always", prefix_reuse_mode="always")
    t0 = time.perf_counter()
    engine = serve_engine.Engine(ns)
    build_s = time.perf_counter() - t0
    per_expert = 3 * engine.cfg.moe_intermediate_size * engine.cfg.hidden_size
    print(f"engine: built in {build_s:.0f}s, resident {engine.rt.weights.resident_bytes() / GIB:.2f} GiB, "
          f"tier {engine.rt.expert_slots} slots "
          f"({engine.rt.expert_slots * per_expert / GIB:.2f} GiB) over "
          f"{sorted(set(engine.rt.experts.layer_devices or [engine.rt.device]))}, graphs={graphs}, "
          f"bound {engine._graph_max}", flush=True)
    return engine, build_s


def _capture_accounting(engine) -> dict:
    """What the captured policy actually did, from the model's registry and the engine's per-slot counters."""
    model = engine.rt.model
    domains = dict(getattr(model, "_graph_domains", {}) or {})
    slots = []
    for s in engine.slots:
        state = getattr(s, "state", None)
        dom = domains.get(id(state)) if state is not None else None
        if dom is not None and getattr(dom, "state", None) is not state:
            dom = None
        slots.append({
            "slot": getattr(s, "name", ""),
            "has_state": state is not None,
            "captured": dom is not None,
            "domain_graphs": int(getattr(dom, "captured", 0)) if dom is not None else 0,
            "domain_gib": round(dom.bytes() / GIB, 3) if dom is not None else 0.0,
            "graph_steps": int(getattr(s, "graph_steps", 0)),
            "graph_notes": sorted(getattr(s, "graph_notes", set())),
        })
    return {
        "graph_max_conversations": int(getattr(model, "graph_max_conversations", 0)),
        "serve_graph_max": int(getattr(engine, "_graph_max", 0)),
        "domains_alive": len(domains),
        "domains_leaked": int(getattr(model, "_graph_leaked", 0) or 0),
        "graphs_alive": sum(int(getattr(d, "captured", 0)) for d in domains.values()),
        "captured_conversations": sum(1 for s in slots if s["captured"]),
        "failed_conversations": sum(1 for s in slots if "failed" in s["graph_notes"]),
        "refused_conversations": sum(1 for s in slots if "refused" in s["graph_notes"]),
        "model_graphs": bool(getattr(model, "graphs", False)),
        "graph_failures": list(getattr(model, "_graph_failures", []) or []),
        "slots": slots,
    }


def run(a) -> int:
    cards = [d for d in a.cards.split(",") if d]
    devices = [a.device] + [d for d in cards if d != a.device]
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    graphs = a.graphs == "on"
    label = a.label or f"serve-domains-par{a.parallel}-{a.prompt_tokens}tok-graphs-{a.graphs}"
    print(f"cards: {cards} (layers on {a.device}); ASCEND_RT_VISIBLE_DEVICES={visible!r}", flush=True)
    print(f"arm: {label}, graphs {a.graphs}, parallel {a.parallel}, prompt {a.prompt_tokens}, gen {a.gen}, "
          f"max_context {a.max_context}, hbm_share {a.hbm_share}, reserve {a.expert_reserve_gib} GiB",
          flush=True)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    src = Path(a.prompt_source)
    t_run = time.perf_counter()
    engine, build_s = _engine(a, cards, graphs)
    rec: dict | None = None
    error = None
    io0 = _io_counters()
    tier0 = _tier(engine.rt)
    try:
        engine.say_ready()
        engine.start_workers()
        if a.distinct_prompts:
            prompts = distinct_prompts(tok, a.parallel, a.prompt_tokens, src)
            print(f"prompt: {a.parallel} distinct prompts of {len(prompts[0])} tokens from {src}", flush=True)
        else:
            # the direct-drive arm's workload: one 2 000-token prompt shared by all conversations, so the
            # tier's expert working set is the same and every greedy stream must agree
            shared = make_prompt(tok, a.prompt_tokens, src)
            prompts = [shared] * a.parallel
            print(f"prompt: one {len(shared)}-token prompt shared by {a.parallel} conversations, from {src}",
                  flush=True)
        rec = serve_parallel(engine, prompts, a.gen, label, devices, timeout=a.timeout)
    except BaseException:                                          # noqa: BLE001 - report, do not hide
        error = traceback.format_exc()
        print(f"arm failed:\n{error}", flush=True)
    finally:
        try:
            engine.stop_workers()
        finally:
            io = _io_delta(io0, _io_counters())
            hbm = {"peak_gib": _peak_gib(devices), "allocated_gib": _allocated_gib(devices)}
            graphs_probe = _capture_accounting(engine)
        engine.shutdown()
        _release(devices)
    if rec is None:
        rec = {"scenario": label, "errors": [error] if error else [], "succeeded": 0}
    rec.update({"cards": cards, "device": a.device, "graphs_config": a.graphs,
                "build_s": round(build_s, 1), "tier_slots": int(engine.rt.expert_slots),
                "tier_before_arm": tier0, "io": io, "hbm": hbm, "graph_domains": graphs_probe,
                "label": label})
    summary = {"run_s": round(time.perf_counter() - t_run, 1), "cards": cards, "visible": visible,
               "device": a.device, "mode": "serve-domains", "parallel": a.parallel,
               "prompt_tokens": a.prompt_tokens, "gen": a.gen, "hbm_share": a.hbm_share,
               "expert_reserve_gib": a.expert_reserve_gib, "max_context": a.max_context,
               "graphs_config": a.graphs, "graph": graphs, "build_s": round(build_s, 1),
               "serve_graph_max": graphs_probe["serve_graph_max"],
               "captured_conversations": graphs_probe["captured_conversations"],
               "domains_alive": graphs_probe["domains_alive"],
               "scenarios": 1}
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "records": [rec]}, indent=1))
        print(f"report: {a.out}", flush=True)
    if rec.get("ttft_s") is not None:
        tiers = rec.get("tier") or {}
        print(f"{RESULT_PREFIX} {label} wall {rec['wall_s']:.1f}s TTFT {rec['ttft_s']:.1f}s "
              f"spread {rec.get('ttft_spread_s')} decode {rec.get('decode_tok_s')} tok/s "
              f"per-req {(rec.get('decode_tok_s_per_request') or [])} "
              f"hit {100 * (tiers.get('hit_rate') or 0):.2f}% read {tiers.get('gb_read')} GB "
              f"peak {rec.get('peak_gib')} captured {graphs_probe['captured_conversations']}/"
              f"{a.parallel} domains {graphs_probe['domains_alive']} "
              f"graphs {graphs_probe['graphs_alive']}", flush=True)
    print(f"{RESULT_PREFIX}-SUMMARY " + json.dumps(summary), flush=True)
    return 0 if rec.get("succeeded") == a.parallel else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--cards", default="npu:0,npu:1,npu:2,npu:3")
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-reserve-gib", type=float, default=13.0,
                    help="expert reserve on the compute card (default 13: four concurrent one-forward "
                         "2 000-token prefills need ~2.7 GiB of activation each)")
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=4, help="conversations served at once (default 4)")
    ap.add_argument("--prompt-tokens", type=int, default=2000)
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--prefill-chunk", type=int, default=0,
                    help="--prompt-chunk for the engine (default 0: the whole prompt in one forward)")
    ap.add_argument("--graphs", choices=("on", "off"), default="on")
    ap.add_argument("--graph-max-conversations", type=int, default=None,
                    help="cap the engine's captured-conversation bound (default: the engine's own, 4)")
    ap.add_argument("--distinct-prompts", action="store_true",
                    help="give every conversation a different prompt cut from the same document (a KV / "
                         "output-contamination check); default is one shared prompt, the direct-drive arm's "
                         "workload, so all greedy streams must agree")
    ap.add_argument("--parallel-streams", action="store_true")
    ap.add_argument("--prompt-source", default="README.md")
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    try:
        return run(a)
    except BaseException:                                          # noqa: BLE001 - a record, not a trace only
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
