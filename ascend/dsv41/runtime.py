"""ascend/dsv41/runtime.py - building an engine from arguments: device, dtype, checkpoint, weights.

Both front ends (``ascend.run_dsv41`` for one prompt, ``ascend.serve_engine`` for the line protocol) need
the same four decisions, and neither should have to know how a card is claimed or how the expert tier is
sized.  This module makes them in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import Checkpoint
from .config import TextConfig, build_layer_plan
from .model import Dsv41, ExpertSource, ExpertStreamer, NullExpertSource, State, Weights

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
GIB = 1 << 30


@dataclass
class Runtime:
    cfg: TextConfig
    ckpt: Checkpoint
    weights: Weights
    model: Dsv41
    device: str
    dtype: torch.dtype
    experts: ExpertSource

    def state(self, max_context: int, batch: int = 1) -> State:
        return State(self.cfg, build_layer_plan(self.cfg), self.device, self.dtype,
                     max_context=max_context, batch=batch)

    def close(self) -> None:
        self.experts.close()
        self.ckpt.close()


def resolve_device(requested: str, share: float) -> str:
    """``cpu`` for tests, else the first visible card, inside this process's share of its HBM."""
    if requested == "cpu":
        return "cpu"
    import torch_npu  # noqa: F401 - registers the backend

    visible = [int(x) for x in os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    if not visible:
        raise SystemExit("ASCEND_RT_VISIBLE_DEVICES names no card")
    index = int(str(requested).split(":")[-1]) if ":" in str(requested) else 0
    torch.npu.set_device(index)
    # the card is shared: 0.5 of 64 GB keeps this process inside 32 GB whatever else is running
    torch.npu.set_per_process_memory_fraction(share, index)
    return f"npu:{index}"


def free_hbm_gib(device: str) -> float:
    """What is left of this process's share, in GiB."""
    if device == "cpu" or not hasattr(torch, "npu"):
        return 0.0
    index = int(device.split(":")[-1])
    free, _total = torch.npu.mem_get_info(index)
    return free / GIB


def build(*, model: str, device: str = "npu:0", dtype: str = "bfloat16", hbm_share: float = 0.5,
          expert_slots: int | None = None, io_threads: int = 6, expert_batch: int = 8,
          no_experts: bool = False, verbose: bool = False) -> Runtime:
    """Load the checkpoint's resident weights and size the expert tier to what is left of the card."""
    model_dir = Path(model)
    cfg = TextConfig.from_json(model_dir / "config.json")
    dev = resolve_device(device, hbm_share)
    dt = DTYPES[dtype]
    ckpt = Checkpoint(model_dir)
    if no_experts:
        src: ExpertSource = NullExpertSource(ckpt, cfg, dt)
    else:
        src = ExpertStreamer(ckpt, cfg, dev, dt, cache_slots=0, io_threads=io_threads, max_batch=expert_batch)
    weights = Weights(ckpt, cfg, dev, dt, expert_source=src, verbose=verbose).load()
    if not no_experts and expert_slots is None:
        # size the tier from what is *actually* left once the resident weights are on the card, less the
        # staging ring and a reserve for activations, KV and the allocator's own fragmentation
        per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        # a 5 GiB reserve: the head's matmul, the prompt's activations and the allocator's own padding all
        # come out of what is left, and an expert tier sized to the last byte OOMs at the first token
        budget = max(0.0, (free_hbm_gib(dev) if dev != "cpu" else 8 * GIB) - src.staging_bytes() - 5.0 * GIB)
        expert_slots = int(budget // per_expert)
    if not no_experts:
        # never more slots than the model has experts: on the host the budget formula would allow millions
        expert_slots = min(int(expert_slots), cfg.num_hidden_layers * cfg.n_routed_experts)
        src.cache_slots = expert_slots
    model_obj = Dsv41(cfg, weights, dev, dt)
    return Runtime(cfg=cfg, ckpt=ckpt, weights=weights, model=model_obj, device=dev, dtype=dt, experts=src)
