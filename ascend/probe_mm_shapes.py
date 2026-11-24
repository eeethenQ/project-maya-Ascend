"""ascend/probe_mm_shapes.py - is a decode-step `npu_quant_matmul` call overhead or weight bandwidth?

The decode step issues 720 of these (40 layers x 6 experts x 3 projections) and the profiler says 19.8 us
each.  The plan on the table is to fold the six experts into one call per projection, on the theory that
those 19.8 us are launch overhead (M is 1).  But 11.8 MB / 19.8 us is 596 GB/s, which is this card's
class of HBM number, and at M=1 the multiply is *only* a weight read: 11.8 M MACs is 0.04 us of Cube.

So this measures the cost as a function of **call count at constant bytes** and as a function of **bytes at
constant call count**, inside a captured graph (where a launch costs 1.88 us, not the eager 47):

    python -m ascend.probe_mm_shapes --device npu:0

If ``18 calls x N=2304`` and ``1 call x N=41472`` cost the same, the call count is not what a decode step
pays for and batching cannot pay.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch_npu  # noqa: F401

from .dsv41.graph_runner import GraphRunner

INTER, HIDDEN = 2304, 5120


def timed(runner: GraphRunner, name: str, rounds: int) -> float:
    """Median ms per replay, one sync per replay (the step does one sync a layer)."""
    times = []
    for _ in range(rounds):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        runner.replay(name)
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.npu.set_device(a.device)
    dev = a.device
    gen = torch.Generator(device="cpu").manual_seed(a.seed)

    def q(rows: int, cols: int) -> torch.Tensor:
        return torch.randint(-127, 128, (rows, cols), dtype=torch.int8, generator=gen).to(dev)

    xq = q(1, HIDDEN)
    tiny = q(1, HIDDEN)
    runner = GraphRunner(dev)
    cases: list[tuple[str, str, str]] = []

    # -- the shape of the cost: N at a fixed call count ---------------------------------------
    for n in (2304, 4608, 13824, 41472):
        w = q(n, HIDDEN)
        s = (torch.rand(n, generator=gen) * 0.01).to(dev)
        out = torch.empty(1, n, dtype=torch.bfloat16, device=dev)

        def body(w=w, s=s, out=out):
            out.copy_(torch_npu.npu_quant_matmul(xq, w.T, s, output_dtype=torch.bfloat16))
        name = f"n{n}"
        runner.capture(name, body, inputs={"w": w, "s": s})
        cases.append((name, f"1 call, N={n}", f"{n * HIDDEN / 1e6:.1f} MB"))

    # -- constant bytes, 18 calls against 1 (the batching under test) --------------------------
    ws = [q(INTER, HIDDEN) for _ in range(18)]
    ss = [(torch.rand(INTER, generator=gen) * 0.01).to(dev) for _ in range(18)]
    sink = torch.empty(1, INTER, dtype=torch.float32, device=dev)

    def body18():
        acc = sink
        for i in range(18):
            acc.add_(torch_npu.npu_quant_matmul(xq, ws[i].T, ss[i], output_dtype=torch.bfloat16))
    runner.capture("c18", body18, inputs={"xq": xq})
    cases.append(("c18", "18 calls, N=2304 each", f"{18 * INTER * HIDDEN / 1e6:.1f} MB"))

    W = torch.stack(ws).reshape(18 * INTER, HIDDEN)
    S = torch.cat(ss)
    out1 = torch.empty(1, 18 * INTER, dtype=torch.bfloat16, device=dev)

    def body1():
        out1.copy_(torch_npu.npu_quant_matmul(xq, W.T, S, output_dtype=torch.bfloat16))
    runner.capture("c1", body1, inputs={"W": W, "S": S})
    cases.append(("c1", "1 call, N=41472", f"{18 * INTER * HIDDEN / 1e6:.1f} MB"))

    # -- the down projection's six calls against one 6x-N call, same bytes ----------------------
    xs6 = q(6, INTER)
    w2s = [q(HIDDEN, INTER) for _ in range(6)]
    s2s = [(torch.rand(HIDDEN, generator=gen) * 0.01).to(dev) for _ in range(6)]
    y6 = torch.empty(6, HIDDEN, dtype=torch.bfloat16, device=dev)

    def body6():
        for i in range(6):
            y6[i:i + 1].copy_(torch_npu.npu_quant_matmul(xs6[i:i + 1], w2s[i].T, s2s[i],
                                                   output_dtype=torch.bfloat16))
    runner.capture("d6", body6, inputs={"xs6": xs6})
    cases.append(("d6", "down: 6 calls, N=5120", f"{6 * INTER * HIDDEN / 1e6:.1f} MB"))

    W2 = torch.cat([w2s[i].T.contiguous() for i in range(6)], dim=1)   # [INTER, 6*HIDDEN]
    S2 = torch.cat(s2s)
    out6 = torch.empty(6, 6 * HIDDEN, dtype=torch.bfloat16, device=dev)

    def body6x():
        # the diagonal form: M=6 rows, one concatenated weight, the wanted blocks on the diagonal
        out6.copy_(torch_npu.npu_quant_matmul(xs6, W2, S2, output_dtype=torch.bfloat16))
    runner.capture("d6x", body6x, inputs={"xs6": xs6})
    cases.append(("d6x", "down: 1 call, N=6x5120", f"{6 * INTER * HIDDEN / 1e6:.1f} MB (6x the MACs)"))

    # -- the fixed cost: a call that reads almost nothing --------------------------------------
    w0 = q(64, HIDDEN)
    s0 = (torch.rand(64, generator=gen) * 0.01).to(dev)
    o0 = torch.empty(1, 64, dtype=torch.bfloat16, device=dev)

    def body0():
        o0.copy_(torch_npu.npu_quant_matmul(xq, w0.T, s0, output_dtype=torch.bfloat16))
    runner.capture("tiny", body0, inputs={"w0": w0, "s0": s0})
    cases.append(("tiny", "1 call, N=64", f"{64 * HIDDEN / 1e6:.3f} MB"))

    print(f"{'case':24s} {'median ms':>10s} {'GB/s':>7s}  bytes")
    for name, label, note in cases:
        ms = timed(runner, name, a.rounds)
        mib = {"n2304": 2304 * HIDDEN, "n4608": 4608 * HIDDEN, "n13824": 13824 * HIDDEN,
               "n41472": 41472 * HIDDEN, "c18": 18 * INTER * HIDDEN, "c1": 18 * INTER * HIDDEN,
               "d6": 6 * INTER * HIDDEN, "d6x": 6 * INTER * HIDDEN,
               "tiny": 64 * HIDDEN}[name]
        print(f"{label:24s} {ms:10.3f} {mib / ms / 1e6:7.0f}  {note}")
    runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
