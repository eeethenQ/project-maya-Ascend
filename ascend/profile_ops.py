"""ascend/profile_ops.py - per-operator device time from two msprof runs, differenced per decode step.

`bench_profile --msprof` divides its totals by ``prompt_tokens + steps`` because its CSV covers the prompt
read and the decode steps alike, and it groups by operator *type* only.  For a targeting document that is
wrong twice: a prefill operator is not a decode operator, and a type is not a shape.

So this reads two msprof output directories of the *same* workload differing only in the number of decode
steps - ``steps_b`` - ``steps_a`` - and subtracts them.  The prompt and the one-off graph capture are in
both, so they cancel; what is left is exactly the replay steps, per operator **and per shape**:

    python -m ascend.profile_ops --a .ascend-work/profile_a --b .ascend-work/profile_b --steps 16

``--json`` writes the full (operator, shape) table so it can be classified and quoted without another run.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

TIME_COLUMNS = ("Total Time(us)", "Task Duration(us)", "Device Duration(us)", "Duration(us)")
SHAPE_COLUMNS = ("Shape", "Input Shapes", "Input Shape", "Input Dims", "Dims")


def load(directory: Path) -> tuple[dict, dict]:
    """(op, shape) -> [count, us], plus the column names it found - grouped, not per task."""
    rows: dict[tuple[str, str], list] = {}
    meta: dict = {"files": [], "columns": {}}
    for path in sorted(directory.rglob("op_summary*.csv")):
        meta["files"].append(str(path))
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if not meta["columns"]:
                meta["columns"] = {"fields": fields}
            column = next((c for c in TIME_COLUMNS if c in fields), "")
            if not column:
                continue
            shape_col = next((c for c in SHAPE_COLUMNS if c in fields), "")
            op_col = "OP Type" if "OP Type" in fields else ("Op Name" if "Op Name" in fields else fields[0])
            meta["columns"].update({"time": column, "shape": shape_col, "op": op_col})
            for row in reader:
                try:
                    us = float(row.get(column) or 0.0)
                except (TypeError, ValueError):
                    continue
                key = (row.get(op_col, "?"), (row.get(shape_col, "") or "?") if shape_col else "?")
                entry = rows.setdefault(key, [0, 0.0])
                entry[0] += 1
                entry[1] += us
    return rows, meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="msprof output dir of the shorter run")
    ap.add_argument("--b", required=True, help="msprof output dir of the longer run")
    ap.add_argument("--steps", type=int, required=True, help="steps_b - steps_a (the differenced steps)")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--json", default="", help="write the full (operator, shape) table here")
    a = ap.parse_args()
    ra, ma = load(Path(a.a))
    rb, mb = load(Path(a.b))
    print(json.dumps({"columns": ma["columns"], "files_a": ma["files"], "files_b": mb["files"]}, indent=1))
    missing = [k for k in ra if k not in rb]
    if missing:
        print(f"warning: {len(missing)} (op, shape) keys in A not in B (capture/prefill residue)")

    diff: dict[tuple[str, str], list] = {}
    for key in set(ra) | set(rb):
        ca, ua = ra.get(key, [0, 0.0])
        cb, ub = rb.get(key, [0, 0.0])
        c, u = cb - ca, ub - ua
        if c < 0 or u < 0:
            print(f"warning: negative diff for {key}: {c} calls {u:.1f} us")
        diff[key] = [c / a.steps, u / a.steps]
    total_us = sum(v[1] for v in diff.values())
    total_calls = sum(v[0] for v in diff.values())

    by_type: dict[str, list] = {}
    for (op, shape), (c, u) in diff.items():
        e = by_type.setdefault(op, [0.0, 0.0])
        e[0] += c
        e[1] += u
    print(f"\nper decode step (differenced over {a.steps} steps): {total_calls:.0f} calls, "
          f"{total_us / 1e3:.2f} ms device\n")
    print(f"{'operator':34s} {'calls/step':>10s} {'ms/step':>8s} {'share':>6s} {'us/call':>8s}")
    for op, (c, u) in sorted(by_type.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{op[:34]:34s} {c:10.1f} {u / 1e3:8.2f} {100 * u / max(total_us, 1e-9):5.1f}% {u / max(c, 1e-9):8.2f}")
    print(f"\n{'operator / shape':70s} {'calls/step':>10s} {'ms/step':>8s} {'us/call':>8s}")
    for (op, shape), (c, u) in sorted(diff.items(), key=lambda kv: -kv[1][1])[: a.top]:
        print(f"{op[:34]:34s} {shape[:35]:35s} {c:10.2f} {u / 1e3:8.3f} {u / max(c, 1e-9):8.2f}")

    if a.json:
        out = [{"op": op, "shape": shape, "calls_per_step": c, "ms_per_step": u / 1e3,
                "us_per_call": u / max(c, 1e-9)} for (op, shape), (c, u) in
               sorted(diff.items(), key=lambda kv: -kv[1][1])]
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
