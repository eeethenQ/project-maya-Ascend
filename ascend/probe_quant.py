"""ascend/probe_quant.py - the per-token int8 quantiser as one Triton kernel, A/B against the torch path.

`per_token_int8` is ten dispatches on the card and runs 280 times a decode step (40 layers x one for the
layer's activation + one per expert's SwiGLU output).  The vendor one-op replacement rounds through bf16
first and moves int8 values; this kernel does not round at all before quantising, so the bar is **exact**:

    q      bit-identical int8
    scale  bit-identical fp32

at the model's real shapes - ``[1, 5120]`` (the layer activation), ``[1, 2304]`` (an expert's gate output)
and ``[T, 5120]`` (a prefill chunk) - over realistic magnitudes and the two cases where a formula
difference would show up first: exact ties (``x = k + 0.5`` with ``scale = 1``) and an all-zero row.

    python -m ascend.probe_quant                       # exactness + capture + timing
    python -m ascend.probe_quant --msprof              # op counts and device time of 280 calls/step
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

WORK = ".ascend-work/quant"


def torch_ref(x):
    """`model.per_token_int8`'s torch form (`kernel=False`), the shipped reference."""
    from ascend.dsv41.model import per_token_int8

    return per_token_int8(x, fused=False, kernel=False)


def cases(device, gen):
    """(name, x) at the model's shapes and the magnitudes the step actually feeds this function."""
    out = []
    for rows, cols in ((1, 5120), (1, 2304), (7, 5120), (64, 5120), (257, 5120)):
        tag = f"[{rows},{cols}]"
        out.append((f"{tag} randn .05", (torch.randn(rows, cols, generator=gen) * 0.05).to(device)))
        out.append((f"{tag} randn*24", (torch.randn(rows, cols, generator=gen) * 24.0).to(device)))
        out.append((f"{tag} uniform 100", ((torch.rand(rows, cols, generator=gen) * 2 - 1) * 100).to(device)))
    # the residual-stream magnitudes the real checkpoint reaches (|x|max ~ 24 by layer 79)
    out.append(("zeros [4, 5120]", torch.zeros(4, 5120, device=device)))
    out.append(("tiny 1e-7 [4, 5120]", (torch.randn(4, 5120, generator=gen) * 1e-7).to(device)))
    out.append(("bf16 [1, 5120]", (torch.randn(1, 5120, generator=gen) * 0.5).to(torch.bfloat16).to(device)))
    # exact ties: amax = 127 makes scale exactly 1.0, so every half-integer is a round-half-to-even tie
    half = (torch.randint(-254, 255, (8, 5120), generator=gen).float() + 0.5).to(device)
    half[0, 0] = 127.0
    half[1, 0] = -127.0
    out.append(("ties scale==1 [8, 5120]", half))
    # near-ties around a non-power-of-two scale: x built from the fp32 quotient
    base = (torch.randn(4, 2304, generator=gen).abs() * 30).to(device)
    base[:, 0] = base.abs().max(-1).values * 1.0 + 1e-3           # amax is column 0
    sc = base[:, :1].abs().clamp_min(1e-5) / 127.0
    k = torch.randint(-126, 127, (4, 2304), generator=gen).float().to(device) + 0.5
    near = (k * sc).to(device)
    near[:, 0] = base[:, 0]
    out.append(("near-ties [4, 2304]", near))
    return out


def exactness(a) -> int:
    from ascend.dsv41 import quant_triton

    gen = torch.Generator(device="cpu").manual_seed(11)
    bad = 0
    for name, x in cases(a.device, gen):
        want_q, want_s = torch_ref(x)
        got_q, got_s = quant_triton.per_token_int8(x)
        dq = (got_q != want_q)
        ds = (got_s.view(torch.int32) != want_s.view(torch.int32))
        nq, ns = int(dq.sum()), int(ds.sum())
        if nq or ns:
            bad += 1
            rows = ds.nonzero().flatten()[:4].tolist()
            am = x.float().abs().amax(dim=-1)
            scale_rows = [(r, hex(int(am[r].view(torch.int32))), hex(int(want_s[r, 0].view(torch.int32))),
                           hex(int(got_s[r, 0].view(torch.int32)))) for r in rows]
            idx = dq.nonzero()[:4].tolist()
            samples = [(idx[i], float(x[idx[i][0], idx[i][1]]), int(want_q[idx[i][0], idx[i][1]]),
                        int(got_q[idx[i][0], idx[i][1]])) for i in range(len(idx))]
            print(f"  {name:26s} EXACT-FAIL q {nq}/{x.numel()} scale {ns}/{x.shape[0]}: "
                  f"rows(amax,want,got)={scale_rows} q={samples[:3]}")
        else:
            print(f"  {name:26s} exact  q {x.numel()} values, scale {x.shape[0]} rows, "
                  f"amax {float(want_s.max()) * 127:.3g}")
    # how many of the tie case's quotients are exactly .5 (a kernel that rounds half-away would differ)
    x = dict(cases(a.device, gen))["ties scale==1 [8, 5120]"]
    q = x.float()
    exact_ties = int(((q - torch.floor(q)) == 0.5).sum())
    print(f"  exact .5 quotients in the tie case: {exact_ties} of {x.numel()}")
    return bad


def capture(a) -> None:
    from ascend.dsv41.graph_runner import GraphRunner
    from ascend.dsv41 import quant_triton

    gen = torch.Generator(device="cpu").manual_seed(3)
    x = (torch.randn(1, 5120, generator=gen) * 6).to(a.device)
    runner = GraphRunner(a.device)
    try:
        def fn():
            return quant_triton.per_token_int8(x)

        runner.capture("quant", fn, inputs={"x": x}, verify=True)
        report = runner.report("quant")
        print(f"  capture verified={report.verified} ok={report.ok} mode={report.mode} {report}")
        before = quant_triton.calls
        runner.replay("quant", sync=True)
        print(f"  replay issued {quant_triton.calls - before} kernel call(s)")
    finally:
        runner.close()


def timing(a) -> None:
    from ascend.dsv41 import quant_triton

    gen = torch.Generator(device="cpu").manual_seed(5)
    x = (torch.randn(1, 5120, generator=gen) * 6).to(a.device)
    h = (torch.randn(1, 2304, generator=gen) * 0.5).to(a.device)

    def bench(fn, n, tag):
        for _ in range(5):
            fn()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            fn()
        torch.npu.synchronize()
        us = (time.perf_counter() - t0) * 1e6 / n
        print(f"  {tag:22s} {us:8.1f} us/call  ({us * 280 / 1e3:6.2f} ms for 280 calls)")
        return us

    print("  per-call, eager:")
    for n, t in ((5120, x), (2304, h)):
        bench(lambda t=t: quant_triton.per_token_int8(t), 200, f"triton [{1},{n}]")
        bench(lambda t=t: torch_ref(t), 200, f"torch  [{1},{n}]")
    # what a step actually does: the 280 calls inside one captured graph
    from ascend.dsv41.graph_runner import GraphRunner

    xs5 = [torch.randn(1, 5120, device=a.device) for _ in range(5)]
    xs2 = [torch.randn(1, 2304, device=a.device) for _ in range(6)]
    for tag, impl in (("triton", lambda t: quant_triton.per_token_int8(t)), ("torch", torch_ref)):
        def step():
            out = None
            for i in range(4):                       # 4 layers' activation quantisers
                out = impl(xs5[i])
                for j in range(6):                   # six experts' gate-output quantisers per layer
                    out = impl(xs2[j])
            return out                               # capture verify needs a tensor out of the graph
        runner = GraphRunner(a.device)
        try:
            runner.capture(tag, step, inputs={"a0": xs5[0], "a1": xs5[1], "a2": xs5[2], "a3": xs5[3],
                                              "h0": xs2[0], "h1": xs2[1], "h2": xs2[2], "h3": xs2[3],
                                              "h4": xs2[4], "h5": xs2[5]}, verify=(tag == "triton"))
            ms = bench(lambda: runner.replay(tag, sync=True), a.steps, f"{tag} 24-call replay")
            print(f"  {tag:8s} captured 24 calls -> {ms * 280 / 24 / 1e3:6.2f} ms per 280-call step")
        finally:
            runner.close()


def run_msprof(a) -> int:
    import shutil
    import subprocess
    from pathlib import Path

    out = Path(WORK)
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    inner = [sys.executable, "-m", "ascend.probe_quant", "--device", a.device, "--steps", str(a.steps),
             "--timing"]
    command = ["msprof", f"--output={out}", "--task-time=on", "--ai-core=on",
               "--application=" + " ".join(inner)]
    print("running:", " ".join(command[:4]), "...", flush=True)
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        print(done.stdout[-3000:], done.stderr[-3000:], sep="\n")
        return done.returncode
    from ascend.bench_profile import read_ops

    rows, column = read_ops(out)
    if not rows:
        print(f"no op_summary under {out}:\n{done.stdout[-2000:]}")
        return 1
    total = sum(r["us"] for r in rows)
    n = 280 * (2 * (a.steps + 6) + 5 * 2 + 2 * 200 * 2)   # rough: not used for the per-call columns
    print(f"\n{'operator':40s} {'calls':>8s} {'ms':>9s} {'us/call':>9s}")
    for r in rows[: a.top]:
        print(f"{r['op'][:40]:40s} {r['count']:8d} {r['us'] / 1e3:9.3f} {r['us'] / max(1, r['count']):9.2f}")
    print(f"\n{sum(r['count'] for r in rows)} operator calls, {total / 1e3:.2f} ms of device time "
          f"(column {column!r}); {n} calls in the workload")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--exact", action="store_true", default=True)
    ap.add_argument("--no-exact", dest="exact", action="store_false")
    ap.add_argument("--capture", action="store_true", default=True)
    ap.add_argument("--no-capture", dest="capture", action="store_false")
    ap.add_argument("--timing", action="store_true", default=True)
    ap.add_argument("--no-timing", dest="timing", action="store_false")
    ap.add_argument("--msprof", action="store_true")
    a = ap.parse_args()
    if a.msprof:
        return run_msprof(a)
    import torch_npu  # noqa: F401 - registers the device

    torch.npu.set_device(a.device)
    from ascend.dsv41 import quant_triton

    print(f"device {a.device}, triton available={quant_triton.available()} error={quant_triton.error()!r}")
    if not quant_triton.available():
        return 1
    quant_triton.warm(a.device)
    bad = 0
    if a.exact:
        print("exactness vs per_token_int8(fused=False):")
        bad = exactness(a)
    if a.capture:
        print("capture:")
        capture(a)
    if a.timing:
        print("timing:")
        timing(a)
    print("EXACT" if not bad else f"{bad} CASE(S) NOT EXACT")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
