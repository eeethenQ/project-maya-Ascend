"""ascend/dsv41/config.py - the checkpoint's config.json as a validated model description, and the layer plan.

DeepSeek-V4.1-Flash is not 40 copies of one layer.  It is 40 layers of four kinds, wired by three static
sharing graphs (CSA2):

* a *KV source* (layers 2, 8, 14, 20) owns a compressor and one compressed-KV cache; every later layer up to
  the next source attends over that same cache instead of building its own.  So the model keeps 4 caches,
  not 40.
* an *index source* (2, 8, 14, 20, 24, 28, 32, 36) owns the TopK(512) selection; the layers after it reuse
  that selection.  Layer 20 is additionally the *candidate* source: it narrows the future to 2048 blocks of
  8 positions, and the later index sources re-score only inside that pool.
* a layer with ``compress_ratio == 0`` (0 and 1) has no compressed branch at all, only the 128-token window.

`build_layer_plan` reproduces that graph from config alone; it is the single source of truth for which
tensors a layer must have, which caches the reader must allocate, and what order a forward pass runs.
This module imports nothing beyond the standard library so the plan can be tested anywhere.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The three shapes of attention layer, from which everything else follows.
SWA_ONLY, FULL, REINDEX, REUSE = "swa", "full", "reindex", "reuse"


@dataclass(frozen=True)
class LayerRole:
    """One backbone layer's place in the attention sharing graph."""

    layer_idx: int
    compress_ratio: int
    kv_source_layer: int | None
    index_source_layer: int | None
    is_kv_source: bool
    is_index_source: bool
    is_candidate_source: bool
    uses_candidate_filter: bool

    @property
    def has_long_context(self) -> bool:
        """True for every layer that has the compressed (global) branch at all."""
        return self.compress_ratio > 0

    @property
    def mode(self) -> str:
        if not self.has_long_context:
            return SWA_ONLY
        if self.is_kv_source:
            return FULL
        if self.is_index_source:
            return REINDEX
        return REUSE


@dataclass(frozen=True)
class Topology:
    """The validated, model-wide sharing graph."""

    layers: tuple[LayerRole, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_topk: int

    def layer(self, layer_idx: int) -> LayerRole:
        return self.layers[layer_idx]

    def kv_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(r.layer_idx for r in self.layers if r.kv_source_layer == source_layer)

    def index_consumers(self, source_layer: int) -> tuple[int, ...]:
        return tuple(r.layer_idx for r in self.layers if r.index_source_layer == source_layer)


def _latest_source(layer_idx: int, sources: tuple[int, ...]) -> int | None:
    """The newest source at or below `layer_idx` - the cache a layer reads, if it has one."""
    return next((s for s in reversed(sources) if s <= layer_idx), None)


def build_layer_plan(cfg: "TextConfig") -> Topology:
    """Build and validate the V4.1 layer-sharing graph from the text config.

    Only the first ``num_hidden_layers`` entries of ``compress_ratios`` describe the backbone: the config
    also carries ratios for DSpark's speculative layers, which this engine does not run.
    """
    num_layers = int(cfg.num_hidden_layers)
    ratios = tuple(cfg.compress_ratios)
    if len(ratios) < num_layers:
        raise ValueError(f"compress_ratios has {len(ratios)} entries for {num_layers} layers")
    ratios = ratios[:num_layers]
    kv_sources = tuple(cfg.kv_source_layer_ids)
    index_sources = tuple(cfg.index_source_layer_ids)
    candidate_source = int(cfg.candidate_source_layer_id)

    if any(r < 0 for r in ratios):
        raise ValueError(f"compress_ratios must be >= 0, got {ratios}")
    for name, ids in (("kv_source_layer_ids", kv_sources), ("index_source_layer_ids", index_sources)):
        bad = [i for i in ids if not 0 <= i < num_layers]
        if bad:
            raise ValueError(f"{name} names layers the backbone does not have: {bad}")
        if list(ids) != sorted(ids):
            raise ValueError(f"{name} must be ascending: {ids}")
        # A source must actually have a compressed branch, or the cache it owns has no meaning.
        for i in ids:
            if ratios[i] <= 0:
                raise ValueError(f"{name} names layer {i}, whose compress_ratio is 0")
    if not 0 <= candidate_source < num_layers:
        raise ValueError(f"candidate_source_layer_id {candidate_source} is not a layer")
    if ratios[candidate_source] <= 0:
        raise ValueError(f"candidate source layer {candidate_source} has no compressed branch")

    roles: list[LayerRole] = []
    for layer_idx, ratio in enumerate(ratios):
        # The sources are only meaningful for a layer that has the compressed branch (upstream sets them
        # only when ratio > 0), so a ratio-0 layer never reads a shared cache.
        has = ratio > 0
        kv_source = _latest_source(layer_idx, kv_sources) if has else None
        index_source = _latest_source(layer_idx, index_sources) if has else None
        roles.append(
            LayerRole(
                layer_idx=layer_idx,
                compress_ratio=ratio,
                kv_source_layer=kv_source,
                index_source_layer=index_source,
                is_kv_source=layer_idx in kv_sources,
                is_index_source=layer_idx in index_sources,
                is_candidate_source=layer_idx == candidate_source,
                # A consumer inherits its index source's selection policy: layer 26 reuses layer 24's
                # TopK, and that TopK was itself restricted to layer 20's candidate blocks.
                uses_candidate_filter=index_source is not None and index_source > candidate_source,
            )
        )
    return Topology(
        layers=tuple(roles),
        kv_source_layer_ids=kv_sources,
        index_source_layer_ids=index_sources,
        candidate_source_layer_id=candidate_source,
        candidate_topk_blocks=int(cfg.candidate_topk_blocks),
        candidate_block_size=int(cfg.candidate_block_size),
        index_topk=int(cfg.index_topk),
    )


@dataclass(frozen=True)
class RopeParams:
    """RoPE as the model uses it.  Two bases: the backbone's and the compressed entries'."""

    theta: float
    compress_theta: float
    scale_factor: float
    beta_fast: float
    beta_slow: float
    original_max_position: int
    # The compressed branch always applies YaRN (the upstream passes use_yarn=role.has_long_context);
    # the window branch applies it once the position passes the original training length.
    yarn_on_compressed: bool = True
    yarn_on_window: bool = False


@dataclass(frozen=True)
class TextConfig:
    """The text model's geometry, read from config.json's `text_config` (the real names are kept)."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    qk_rope_head_dim: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    moe_intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    hidden_act: str
    swiglu_limit: float
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    topk_method: str
    norm_topk_prob: bool
    routed_scaling_factor: float
    sliding_window: int
    compress_ratios: tuple[int, ...]
    kv_source_layer_ids: tuple[int, ...]
    index_source_layer_ids: tuple[int, ...]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    max_position_embeddings: int
    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float
    rope: RopeParams
    engram_layer_ids: tuple[int, ...] = ()
    num_nextn_predict_layers: int = 0
    tie_word_embeddings: bool = False
    image_token_id: int | None = None
    # Not in this checkpoint; upstream defaults it to 0 (hash-based expert routing is a V4 feature).
    num_hash_layers: int = 0
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @property
    def nope_head_dim(self) -> int:
        return self.head_dim - self.qk_rope_head_dim

    @property
    def n_heads_per_group(self) -> int:
        """Heads sharing one output group in wo_a (64 / 8 = 8)."""
        if self.num_attention_heads % self.o_groups:
            raise ValueError(f"{self.num_attention_heads} heads do not split into {self.o_groups} groups")
        return self.num_attention_heads // self.o_groups

    @property
    def wo_a_in(self) -> int:
        """wo_a's input width: one group's heads x head_dim (8 x 512 = 4096 in the checkpoint)."""
        return self.n_heads_per_group * self.head_dim

    @property
    def wo_a_out(self) -> int:
        return self.o_groups * self.o_lora_rank

    @classmethod
    def from_json(cls, path: str | Path) -> "TextConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> "TextConfig":
        text = doc.get("text_config", doc)
        rope_scaling = text.get("rope_scaling") or {}
        rope = RopeParams(
            theta=float(text.get("rope_theta", 10000.0)),
            compress_theta=float(text.get("compress_rope_theta", text.get("rope_theta", 10000.0))),
            scale_factor=float(rope_scaling.get("factor", 1.0)),
            beta_fast=float(rope_scaling.get("beta_fast", 32.0)),
            beta_slow=float(rope_scaling.get("beta_slow", 1.0)),
            original_max_position=int(rope_scaling.get("original_max_position_embeddings",
                                                        text.get("max_position_embeddings", 65536))),
        )
        cfg = cls(
            hidden_size=int(text["hidden_size"]),
            num_hidden_layers=int(text["num_hidden_layers"]),
            num_attention_heads=int(text["num_attention_heads"]),
            head_dim=int(text["head_dim"]),
            qk_rope_head_dim=int(text["qk_rope_head_dim"]),
            q_lora_rank=int(text["q_lora_rank"]),
            o_lora_rank=int(text["o_lora_rank"]),
            o_groups=int(text["o_groups"]),
            moe_intermediate_size=int(text["moe_intermediate_size"]),
            vocab_size=int(text["vocab_size"]),
            rms_norm_eps=float(text["rms_norm_eps"]),
            hidden_act=str(text.get("hidden_act", "silu")),
            swiglu_limit=float(text["swiglu_limit"]),
            n_routed_experts=int(text["n_routed_experts"]),
            n_shared_experts=int(text.get("n_shared_experts", 1)),
            num_experts_per_tok=int(text["num_experts_per_tok"]),
            scoring_func=str(text.get("scoring_func", "softmax")),
            topk_method=str(text.get("topk_method", "greedy")),
            norm_topk_prob=bool(text.get("norm_topk_prob", True)),
            routed_scaling_factor=float(text.get("routed_scaling_factor", 1.0)),
            sliding_window=int(text["sliding_window"]),
            compress_ratios=tuple(int(x) for x in text["compress_ratios"]),
            kv_source_layer_ids=tuple(int(x) for x in text["kv_source_layer_ids"]),
            index_source_layer_ids=tuple(int(x) for x in text["index_source_layer_ids"]),
            candidate_source_layer_id=int(text["candidate_source_layer_id"]),
            candidate_topk_blocks=int(text["candidate_topk_blocks"]),
            candidate_block_size=int(text["candidate_block_size"]),
            index_n_heads=int(text["index_n_heads"]),
            index_head_dim=int(text["index_head_dim"]),
            index_topk=int(text["index_topk"]),
            max_position_embeddings=int(text["max_position_embeddings"]),
            hc_mult=int(text.get("hc_mult", 4)),
            hc_sinkhorn_iters=int(text.get("hc_sinkhorn_iters", 20)),
            hc_eps=float(text.get("hc_eps", 1e-6)),
            rope=rope,
            engram_layer_ids=tuple(int(x) for x in text.get("engram_layer_ids", ())),
            num_nextn_predict_layers=int(text.get("num_nextn_predict_layers", 0)),
            tie_word_embeddings=bool(doc.get("tie_word_embeddings", text.get("tie_word_embeddings", False))),
            image_token_id=doc.get("image_token_id"),
            num_hash_layers=int(text.get("num_hash_layers", 0)),
            raw=doc,
        )
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.hidden_act != "silu":
            raise ValueError(f"hidden_act is {self.hidden_act}; this engine implements silu")
        if self.topk_method != "noaux_tc":
            raise ValueError(f"topk_method is {self.topk_method}; this engine implements noaux_tc")
        if self.scoring_func != "sqrtsoftplus":
            raise ValueError(f"scoring_func is {self.scoring_func}; this engine implements sqrtsoftplus")
        if self.o_lora_rank * self.o_groups != self.wo_a_out:
            raise ValueError("o_groups x o_lora_rank is not wo_a's output width")
        if self.head_dim - self.qk_rope_head_dim <= 0:
            raise ValueError("qk_rope_head_dim must be smaller than head_dim")
        if self.index_head_dim % 2:
            raise ValueError("index_head_dim must be even for RoPE")
        if self.hc_mult < 1:
            raise ValueError("hc_mult must be at least 1")
        if not 0 < self.swiglu_limit:
            raise ValueError("swiglu_limit must be positive")
        if self.n_shared_experts != 1:
            # The shared expert is a single SwiGLU at moe_intermediate_size in this checkpoint.  More than
            # one would mean a wider shared FFN; the tensor shapes would have to change with it.
            raise ValueError(f"n_shared_experts is {self.n_shared_experts}; this engine implements 1")
        # The sharing graph is part of the config's meaning: a source list that cannot be laid out over these
        # layers is a broken checkpoint, and it must fail here rather than at the first forward pass.
        build_layer_plan(self)
