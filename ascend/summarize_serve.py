"""ascend/summarize_serve.py - the derived tables for the four-card serve measurements.

Reads the raw result JSONs `bench_serve` wrote and prints (and optionally writes) the comparison the
report quotes:

* the 4 x 2 000-token and 4 x short arms on the four-card tier against the same arms on the three-card
  tier measured the same day (the only same-regime comparison) and against the committed three-card
  record (which was storage-bound, so its TTFT is not a card-count comparison);
* GB per decode step, derived as ``(tier GB per request - a warm single prefill's GB) / generated
  tokens`` - a derived figure, labelled as such in the output;
* the graph-lease arms;
* the multi-turn reuse comparison: tokens identical per turn and TTFT per arm.

Usage:

    python3 -m ascend.summarize_serve \
        --two-k ascend/results/bench_serve-2k-4card.json \
        --two-k-3card ascend/results/bench_serve-2k-3card-day.json \
        --two-k-recorded ascend/results/bench_serve-2k-3card.json \
        --lease ascend/results/serve-lease-4card.json \
        --multiturn-safe ascend/results/serve-multiturn-4card-safe.json \
        --multiturn-always ascend/results/serve-multiturn-4card-always.json \
        --out ascend/results/bench-4card-derived.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str | None) -> dict:
    if not path:
        return {}
    return json.loads(Path(path).read_text())


def records(doc: dict) -> dict[str, dict]:
    return {r.get("scenario", ""): r for r in doc.get("records", [])}


def arm(doc: dict, name: str) -> dict:
    return records(doc).get(name, {})


def two_k_row(doc: dict) -> dict:
    single = [r for r in doc.get("records", []) if r.get("scenario") == "single"]
    warm = single[-1] if single else {}
    arm2k = arm(doc, "concurrent-2k")
    short = arm(doc, "control-short")
    prefill_gb = (warm.get("tier") or {}).get("gb_read")
    per_req_gb = (arm2k.get("tier") or {}).get("gb_read_per_request")
    per_req_steps = arm2k.get("gen_tokens") or 0
    gb_step = (round((per_req_gb - prefill_gb) / per_req_steps, 2)
               if per_req_gb is not None and prefill_gb is not None and per_req_steps else None)
    return {
        "cards": doc.get("summary", {}).get("cards") or arm2k.get("cards"),
        "visible": doc.get("summary", {}).get("visible"),
        "run_s": doc.get("summary", {}).get("run_s"),
        "tier_slots_build": (warm.get("tier_slots") or arm2k.get("tier_slots")),
        "single_cold_ttft_s": single[0].get("ttft_s") if single else None,
        "single_warm_ttft_s": warm.get("ttft_s"),
        "single_warm_gb": prefill_gb,
        "two_k": {
            "wall_s": arm2k.get("wall_s"),
            "ttft_s_list": arm2k.get("ttft_s_list"),
            "ttft_median_s": arm2k.get("ttft_median_s"),
            "ttft_spread_s": arm2k.get("ttft_spread_s"),
            "decode_tok_s": arm2k.get("decode_tok_s"),
            "decode_tok_s_per_request": arm2k.get("decode_tok_s_per_request"),
            "decode_ms_median_per_request": arm2k.get("decode_ms_median_per_request"),
            "hit_rate": (arm2k.get("tier") or {}).get("hit_rate"),
            "gb_per_request": per_req_gb,
            "gb_per_step": gb_step,
            "peak_gib": arm2k.get("peak_gib"),
            "tokens_identical": arm2k.get("tokens_identical"),
        },
        "short": {
            "wall_s": short.get("wall_s"),
            "ttft_s_list": short.get("ttft_s_list"),
            "ttft_median_s": short.get("ttft_median_s"),
            "decode_tok_s": short.get("decode_tok_s"),
            "decode_tok_s_per_request": short.get("decode_tok_s_per_request"),
            "hit_rate": (short.get("tier") or {}).get("hit_rate"),
            "gb_per_request": (short.get("tier") or {}).get("gb_read_per_request"),
            "peak_gib": short.get("peak_gib"),
        },
    }


def ratio(a, b):
    return round(a / b, 2) if a and b else None


def _scratch(summary: dict, turn, key: str):
    for m in (summary.get("scratch") or {}).get("per_turn", []):
        if m.get("turn") == turn:
            return m.get(key)
    return None


def multiturn_row(doc: dict) -> dict:
    s = arm(doc, "multiturn-summary")
    turns = [r for r in doc.get("records", []) if r.get("scenario", "").startswith("turn-")]
    return {
        "cards": doc.get("summary", {}).get("cards"),
        "run_s": doc.get("summary", {}).get("run_s"),
        "mode": s.get("reuse_mode"),
        "prefill_chunk": s.get("prefill_chunk"),
        "turns": s.get("turns"),
        "ttft_s_list": s.get("ttft_s_list"),
        "median_ttft_s": s.get("median_ttft_s"),
        "reused": s.get("reused"),
        "reused_exact_all": s.get("reused_exact_all"),
        "tokens_identical_all": s.get("tokens_identical_all"),
        "first_differing_turn": (s.get("scratch") or {}).get("first_differing_turn"),
        "per_turn": [{"turn": r.get("turn"), "prompt_tokens": r.get("prompt_tokens"),
                      "reused": r.get("reused"), "ttft_s": r.get("ttft_s"),
                      "scratch_ttft_s": _scratch(s, r.get("turn"), "scratch_ttft_s"),
                      "tokens_identical": _scratch(s, r.get("turn"), "tokens_identical"),
                      "ttft_ratio": ratio(_scratch(s, r.get("turn"), "scratch_ttft_s"),
                                          r.get("ttft_s"))}
                     for r in turns],
    }


def lease_row(doc: dict) -> dict:
    return arm(doc, "lease-summary")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--two-k", required=True)
    ap.add_argument("--two-k-3card", default="")
    ap.add_argument("--two-k-recorded", default="")
    ap.add_argument("--lease", default="")
    ap.add_argument("--multiturn-safe", default="")
    ap.add_argument("--multiturn-always", default="")
    ap.add_argument("--multiturn-narrow", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    four = two_k_row(load(a.two_k))
    three_day = two_k_row(load(a.two_k_3card)) if a.two_k_3card else {}
    recorded = two_k_row(load(a.two_k_recorded)) if a.two_k_recorded else {}
    out = {
        "four_card": four,
        "three_card_same_day": three_day,
        "three_card_recorded": recorded,
        "four_vs_three_same_day": {
            "two_k_ttft_speedup": ratio((three_day.get("two_k") or {}).get("ttft_median_s"),
                                        (four.get("two_k") or {}).get("ttft_median_s")),
            "two_k_decode_speedup": ratio((four.get("two_k") or {}).get("decode_tok_s"),
                                          (three_day.get("two_k") or {}).get("decode_tok_s")),
            "short_ttft_speedup": ratio((three_day.get("short") or {}).get("ttft_median_s"),
                                        (four.get("short") or {}).get("ttft_median_s")),
            "short_decode_speedup": ratio((four.get("short") or {}).get("decode_tok_s"),
                                          (three_day.get("short") or {}).get("decode_tok_s")),
        },
        "four_vs_recorded": {
            "two_k_ttft_speedup": ratio((recorded.get("two_k") or {}).get("ttft_median_s"),
                                        (four.get("two_k") or {}).get("ttft_median_s")),
            "two_k_decode_speedup": ratio((four.get("two_k") or {}).get("decode_tok_s"),
                                          (recorded.get("two_k") or {}).get("decode_tok_s")),
            "short_ttft_speedup": ratio((recorded.get("short") or {}).get("ttft_median_s"),
                                        (four.get("short") or {}).get("ttft_median_s")),
        },
        "lease": lease_row(load(a.lease)) if a.lease else {},
        "multiturn_safe": multiturn_row(load(a.multiturn_safe)) if a.multiturn_safe else {},
        "multiturn_always": multiturn_row(load(a.multiturn_always)) if a.multiturn_always else {},
        "multiturn_safe_large_chunk": multiturn_row(load(a.multiturn_narrow)) if a.multiturn_narrow else {},
    }
    text = json.dumps(out, indent=1)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
        print(f"\nwrote {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
