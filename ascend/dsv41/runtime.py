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
# the HBM share this process claimed per device, so the budget can be computed from it
_SHARES: dict[int, float] = {}


@dataclass
class Runtime:
    cfg: TextConfig
    ckpt: Checkpoint
    weights: Weights
    model: Dsv41
    device: str
    dtype: torch.dtype
    experts: ExpertSource
    engram: object | None = None          # Engram, when the checkpoint's 203 GB of tables are present

    @property
    def expert_slots(self) -> int:
        return int(getattr(self.experts, "cache_slots", 0))

    def decode_working_set(self) -> int:
        """Experts one decode step routes: below this the tier thrashes and every step re-reads them."""
        return self.cfg.num_hidden_layers * self.cfg.num_experts_per_tok

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
    _SHARES[index] = share
    return f"npu:{index}"


def free_hbm_bytes(device: str) -> int:
    """What is left of this process's *share*, in **bytes**.

    ``mem_get_info`` answers for the allocator, not for the share: on a 64 GB card claiming 0.5 it can
    report tens of GiB after this process has already taken half.  The share is what the cap enforces, so
    the budget is computed from it and from what the allocator holds.
    """
    if device == "cpu" or not hasattr(torch, "npu"):
        return 0
    index = int(device.split(":")[-1])
    # the card's own capacity, not the share-limited figure get_device_properties may report
    _free_physical, total = torch.npu.mem_get_info(index)
    allowed = float(total) * _SHARES.get(index, 0.5)
    # what this process actually holds: reserved includes blocks the allocator grew during loading and no
    # longer needs, which would wrongly shrink the tier to nothing
    used = float(torch.npu.memory_allocated(index))
    # bytes, not GiB: returning a "gib" figure once made the tier arithmetic divide a gigabyte count by a
    # byte count and quietly size the tier to zero
    return int(max(0.0, allowed - used))


def build(*, model: str, device: str = "npu:0", dtype: str = "bfloat16", hbm_share: float = 0.5,
          expert_slots: int | None = None, io_threads: int = 6, expert_batch: int = 8,
          expert_reserve_gib: float = 6.0, expert_devices: "list[str] | None" = None,
          engram_tokenizer: str | None = None, no_experts: bool = False, verbose: bool = False) -> Runtime:
    """Load the checkpoint's resident weights and size the expert tier to what is left of the cards.

    `expert_devices` spreads the routed experts' tier over several cards, a contiguous block of layers per
    card.  The layers themselves stay on `device`: the tier is what needs room (a decode step routes 240
    experts, 8.5 GB, and the ones a token routes change from token to token, so the set a conversation
    touches is many times that), while the attention state and the resident weights need very little.
    """
    model_dir = Path(model)
    cfg = TextConfig.from_json(model_dir / "config.json")
    dev = resolve_device(device, hbm_share)
    dt = DTYPES[dtype]
    # claim the share of every card the experts will use, and record which layers go where
    pool = [d for d in (expert_devices or []) if d != "cpu"]
    for d in pool:
        resolve_device(d, hbm_share)
    layer_devices = None
    if pool and len(pool) > 1:
        n = cfg.num_hidden_layers
        layer_devices = [pool[layer * len(pool) // n] for layer in range(n)]
    ckpt = Checkpoint(model_dir)
    if no_experts:
        src: ExpertSource = NullExpertSource(ckpt, cfg, dt)
    else:
        src = ExpertStreamer(ckpt, cfg, dev, dt, cache_slots=0, io_threads=io_threads,
                             max_batch=expert_batch, layer_devices=layer_devices)
    weights = Weights(ckpt, cfg, dev, dt, expert_source=src, verbose=verbose).load()
    if not no_experts:
        per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        if expert_slots is None:
            # The reserve is for the activations and temporaries of the card the layers run on; a card that
            # only holds experts needs room for little else.  The staging ring is *host* pinned memory and
            # does not come out of HBM at all.  Sized per card, so no card can be pushed past its share by
            # the others - the tier is HBM, and the cap is per card.
            reserve = int(expert_reserve_gib * GIB)
            if dev == "cpu":
                per_card = {dev: 8 * GIB}
            else:
                per_card = {dev: max(0, free_hbm_bytes(dev) - reserve)}
                for extra in dict.fromkeys(pool):
                    if extra != dev:
                        # a card that only holds experts still needs room for the batch in flight: the
                        # staged copy, the slots it is cloned into, and the allocator's own padding.  At
                        # 1 GiB the tier filled the card and the next insert ran it out of memory.
                        per_card[extra] = max(0, free_hbm_bytes(extra) - int(5.0 * GIB))
            # never more slots on a card than the model has experts
            cap = cfg.num_hidden_layers * cfg.n_routed_experts
            src.set_slots({d: min(int(b // per_expert), cap) for d, b in per_card.items()})
        else:
            src.set_slots(total=min(int(expert_slots), cfg.num_hidden_layers * cfg.n_routed_experts))
    engram = None
    if engram_tokenizer and cfg.engram_layer_ids and f"layers.{cfg.engram_layer_ids[0]}.engram.embed.weight" in ckpt:
        from .engram import Engram                      # only when the 203 GB of tables are actually present
        engram = Engram(ckpt, cfg, engram_tokenizer, dev, dt)
        for layer in cfg.engram_layer_ids:
            proj = weights.engram_projection(layer)
            engram.attach(layer, proj["wkv"], proj["q_weight"], proj["k_weight"])
        if verbose:
            print(f"  engram: layers {list(cfg.engram_layer_ids)}, "
                  f"{engram.cols} rows per token, tables mapped not loaded", flush=True)
    elif engram_tokenizer:
        print("  engram: the checkpoint has no tables for this config; disabled", flush=True)
    model_obj = Dsv41(cfg, weights, dev, dt, engram=engram)
    return Runtime(cfg=cfg, ckpt=ckpt, weights=weights, model=model_obj, device=dev, dtype=dt, experts=src,
                   engram=engram)
