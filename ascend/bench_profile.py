"""ascend/bench_profile.py - what a decode step spends its time on, operation by operation (Ascend profiler).

`bench_step` says how many milliseconds a step costs and which phase they belong to.  It cannot say whether
that time is one slow operator or a thousand tiny ones paying the launch overhead - and that distinction
decides what to fix.  For that, the Ascend profiler (`msprof`) collects per-operator device time.

    python -m ascend.bench_profile --model /mnt/weights/... --msprof

`--msprof` re-runs this module under `msprof --application=...`, then reads the `op_summary_*.csv` it wrote
and prints the operators by device time.  Without the flag it just times the steps (torch's own profiler
does not report NPU time on this stack: `ProfilerActivity.PrivateUse1` accepts the setting and returns
nothing).  Output goes under `.ascend-work/`, which is gitignored.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

from .dsv41 import runtime as rt

WORK = Path(".ascend-work/profile")
TIME_COLUMNS = ("Total Time(us)", "Task Duration(us)", "Device Duration(us)", "Duration(us)")


def build_and_run(a, steps: int, timed: bool) -> None:
    """Load, prefill, then `steps` decode steps - the workload both modes measure."""
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16",
                   expert_devices=[d for d in a.expert_devices.split(",") if d] or None,
                   engram_tokenizer=a.engram_tokenizer or None, graphs=a.graphs)
    model = run.model
    if a.fused_quant:
        model.fused_quant, model.fused_swiglu = True, False
    if a.torch_mhc:
        model.fused_mhc = False
    if timed:
        print(f"engine: {run.device}, engram {'on' if run.engram else 'off'}, "
              f"{run.expert_slots} expert slots", flush=True)
    state = run.state(a.max_context)
    ids = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    model.forward(ids, state, compute_logits=True)
    tok = 3
    times = []
    for _ in range(steps):
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        t0 = time.perf_counter()
        logits = model.forward([tok], state, logits_last_only=True)
        tok = int(logits[0].argmax())
        if str(run.device).startswith("npu"):
            torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    if timed:
        print(f"{steps} steps: {sum(times) / len(times):.1f} ms/token "
              f"(min {min(times):.0f}, max {max(times):.0f})", flush=True)
    run.close()


def read_ops(directory: Path) -> tuple[list[dict], str]:
    """The operators out of msprof's `op_summary_*.csv`: grouped by type, sorted by device time."""
    files = sorted(directory.rglob("op_summary*.csv"))
    if not files:
        return [], ""
    by_type: dict[str, dict] = {}
    column = ""
    for path in files:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            column = next((c for c in TIME_COLUMNS if c in fields), column)
            if not column:
                column = next((c for c in fields if "Time" in c and "Total" in c), "")
            name_col = "OP Type" if "OP Type" in fields else ("Op Name" if "Op Name" in fields else fields[0])
            if not column:
                continue
            for row in reader:
                try:
                    micros = float(row.get(column) or 0.0)
                except (TypeError, ValueError):
                    continue
                entry = by_type.setdefault(row.get(name_col, "?"), {"op": row.get(name_col, "?"),
                                                                    "count": 0, "us": 0.0})
                entry["count"] += 1
                entry["us"] += micros
    return sorted(by_type.values(), key=lambda r: -r["us"]), column


def run_msprof(a) -> int:
    out = WORK
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    inner = [sys.executable, "-m", "ascend.bench_profile", "--model", a.model, "--device", a.device,
             "--prompt-tokens", str(a.prompt_tokens), "--steps", str(a.steps),
             "--max-context", str(a.max_context), "--under-msprof"]
    if a.expert_devices:
        inner += ["--expert-devices", a.expert_devices]
    if a.engram_tokenizer:
        inner += ["--engram-tokenizer", a.engram_tokenizer]
    if a.graphs:
        inner += ["--graphs"]
    if a.fused_quant:
        inner += ["--fused-quant"]
    if a.torch_mhc:
        inner += ["--torch-mhc"]
    command = ["msprof", f"--output={out}", "--task-time=on", "--ai-core=on", "--runtime-api=on",
               "--application=" + " ".join(inner)]
    print("running:", " ".join(command[:5]), "... (the profiler makes it several times slower)")
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-2000:], done.stderr[-2000:], sep="\n")
        return done.returncode
    rows, column = read_ops(out)
    if not rows:
        print(f"no op_summary*.csv under {out}; msprof said:\n{done.stdout[-1500:]}")
        return 1
    total = sum(r["us"] for r in rows)
    launches = sum(r["count"] for r in rows)
    # The profile covers the prompt read *and* the decode steps, so a "per step" figure counted the
    # prompt's operators as if they belonged to one decode step - it made a change look like an 18%
    # regression when its own signature (half the Cos/Sin) was right.  Divide by tokens processed instead.
    tokens = max(1, a.prompt_tokens + a.steps)
    steps = tokens
    print(f"\n1 prompt of {a.prompt_tokens} tokens + {a.steps} decode steps = {tokens} tokens profiled")
    print(f"{'operator':44s} {'calls/tok':>10s} {'ms/tok':>9s} {'share':>7s} {'us/call':>9s}")
    for r in rows[: a.top]:
        print(f"{r['op'][:44]:44s} {r['count'] / steps:10.1f} {r['us'] / 1e3 / steps:9.2f} "
              f"{100 * r['us'] / max(total, 1e-9):6.1f}% {r['us'] / max(1, r['count']):9.1f}")
    print(f"\n{launches / steps:.0f} operator calls per token over {len(rows)} distinct operators; "
          f"{total / 1e3 / steps:.1f} ms of device time per token (column {column!r})")
    tiny = [r for r in rows if r["us"] / max(1, r["count"]) < 100]
    print(f"under 100 us per call: {sum(r['count'] for r in tiny) / steps:.0f} calls/token, "
          f"{sum(r['us'] for r in tiny) / 1e3 / steps:.1f} ms/token "
          f"({100 * sum(r['us'] for r in tiny) / max(total, 1e-9):.0f}% of device time)")
    print(f"trace and CSVs: {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--max-context", type=int, default=1024)
    ap.add_argument("--engram-tokenizer", default="")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--graphs", action="store_true",
                    help="run the decode steps as captured per-layer graphs (see ascend/run_dsv41.py)")
    ap.add_argument("--fused-quant", action="store_true",
                    help="use npu_dynamic_quant for the expert activation quantisation instead of the torch "
                         "per_token_int8 (off by default: -2 240 operators/step and -3 ms of device time, "
                         "but a numeric variant - see Dsv41.__init__ - and no effect on the decode step)")
    ap.add_argument("--msprof", action="store_true", help="profile with msprof and report per-operator time")
    ap.add_argument("--torch-mhc", action="store_true",
                    help="A/B the fused mHC: run the torch Sinkhorn instead of CANN's MhcPreSinkhorn")
    ap.add_argument("--under-msprof", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    import torch_npu  # noqa: F401 - registers the backend
    if a.msprof:
        return run_msprof(a)
    build_and_run(a, a.steps, timed=True)
    print("\nfor per-operator device time: add --msprof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
