"""ascend/bench_experts.py - what one decode step's routed experts cost, three ways.

A decode step routes top-6 experts in each of 40 layers; each expert is 3 x 2304 x 5120 int8 = 33.8 MiB,
so a token needs ~8.1 GB moved from the checkpoint's page cache into HBM.  On this machine host->HBM
measured ~20 GB/s and a single-threaded host copy ~9 GB/s, so *how* the bytes are read and copied decides
the tokens per second, and nothing else in the engine comes close.

    python -m ascend.bench_experts --model /mnt/weights/... --experts 6 --layers 8

Prints, for each strategy, the host read and the host->device copy as separate numbers, because they are
different bottlenecks: the first is host memory bandwidth, the second the PCIe link.
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .dsv41.checkpoint import Checkpoint
from .dsv41.config import TextConfig

GIB = 1 << 30


def strategy_views(ckpt: Checkpoint, layer: int, experts: list[int]) -> dict[str, np.ndarray]:
    """Zero-copy page-cache views stacked on the host: one memcpy per projection."""
    out = {}
    for p in ("w1", "w2", "w3"):
        parts = []
        for e in experts:
            base = f"layers.{layer}.ffn.experts.{e}.{p}.weight"
            info = ckpt.info(base)
            parts.append(np.frombuffer(ckpt.raw(base), dtype=np.int8).reshape(info.shape))
        out[p] = np.stack(parts)
    return out


def open_shards(ckpt: Checkpoint) -> dict[str, int]:
    """One descriptor per shard, kept open for the whole run (an expert read is a pread, not an open)."""
    return {name: os.open(str(shard.path), os.O_RDONLY) for name, shard in ckpt.shards.items()}


def read_expert(ckpt: Checkpoint, fds: dict[str, int], layer: int, e: int,
                buffers: dict[str, np.ndarray], slot: int) -> None:
    """`preadv` one expert's three matrices straight into buffer slot `slot`: one copy, no stack."""
    for p in ("w1", "w2", "w3"):
        info = ckpt.info(f"layers.{layer}.ffn.experts.{e}.{p}.weight")
        shard = ckpt.shards[info.shard]
        os.preadv(fds[info.shard], [memoryview(buffers[p][slot]).cast("B")], shard.data_start + info.begin)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--experts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--device", default="npu:0")
    a = ap.parse_args()

    cfg = TextConfig.from_json(Path(a.model) / "config.json")
    ckpt = Checkpoint(a.model)
    import torch_npu  # noqa: F401
    torch.npu.set_device(0)
    torch.npu.set_per_process_memory_fraction(0.5, 0)
    device = a.device

    experts = list(range(a.experts))
    out_dim, in_dim = cfg.moe_intermediate_size, cfg.hidden_size
    per_expert = 3 * out_dim * in_dim
    total = a.layers * per_expert * a.experts
    print(f"{a.layers} layers x {a.experts} experts x {per_expert / (1 << 20):.1f} MiB "
          f"= {total / 1e9:.2f} GB, as one decode step of {cfg.num_hidden_layers} layers would route "
          f"{cfg.num_hidden_layers * a.experts * per_expert / 1e9:.2f} GB")

    # 1. page-cache views + np.stack, then one copy
    t0 = time.perf_counter()
    stacked = [strategy_views(ckpt, layer, experts) for layer in range(a.layers)]
    host_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for layer, blocks in enumerate(stacked):
        for p in ("w1", "w2", "w3"):
            torch.from_numpy(blocks[p]).to(device, non_blocking=True)
    torch.npu.synchronize()
    h2d_s = time.perf_counter() - t0
    del stacked
    torch.npu.empty_cache()
    print(f"views+stack : host {host_s * 1e3:7.1f} ms ({total / host_s / 1e9:5.1f} GB/s)  "
          f"h2d {h2d_s * 1e3:7.1f} ms ({total / h2d_s / 1e9:5.1f} GB/s)")
    scale = cfg.num_hidden_layers / a.layers
    print(f"              a full 40-layer step: {(host_s + h2d_s) * scale:.2f} s "
          f"-> {1 / ((host_s + h2d_s) * scale):.2f} tok/s")

    # 2. pread into a preallocated buffer, one thread
    buf = {p: np.empty((len(experts), out_dim if p != "w2" else in_dim,
                        in_dim if p != "w2" else out_dim), dtype=np.int8)
           for p in ("w1", "w2", "w3")}
    fds = open_shards(ckpt)
    t0 = time.perf_counter()
    for layer in range(a.layers):
        for i, e in enumerate(experts):
            read_expert(ckpt, fds, layer, e, buf, i)
    host_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(a.layers):
        for p in ("w1", "w2", "w3"):
            torch.from_numpy(buf[p]).to(device, non_blocking=True)
    torch.npu.synchronize()
    h2d_s = time.perf_counter() - t0
    print(f"pread 1 thr : host {host_s * 1e3:7.1f} ms ({total / host_s / 1e9:5.1f} GB/s)  "
          f"h2d {h2d_s * 1e3:7.1f} ms ({total / h2d_s / 1e9:5.1f} GB/s)")

    # 3. pread with a thread per expert
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.threads) as pool:
        for layer in range(a.layers):
            list(pool.map(lambda e, L=layer: read_expert(ckpt, fds, L, e, buf, e),
                          experts))
    host_s = time.perf_counter() - t0
    print(f"pread {a.threads} thr : host {host_s * 1e3:7.1f} ms ({total / host_s / 1e9:5.1f} GB/s)  "
          f"a full step {(host_s + h2d_s) * scale:.2f} s -> {1 / ((host_s + h2d_s) * scale):.2f} tok/s")

    # 4. the same with pinned staging: the copy then overlaps the device work
    try:
        pin = torch.from_numpy(buf["w1"]).pin_memory()
        print(f"pinned staging: available ({pin.numel() * pin.element_size() / (1 << 20):.0f} MiB)")
    except Exception as e:                                                    # noqa: BLE001
        print(f"pinned staging: not available ({e})")
    for fd in fds.values():
        os.close(fd)
    ckpt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
