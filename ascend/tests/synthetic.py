"""ascend/tests/synthetic.py - a checkpoint small enough to build in a test, of the real model's shape.

Every tensor name, dtype and shape relation the engine relies on is generated here, at a geometry small
enough to run on a laptop: four layers that cover all three attention modes (window-only, full with a
compressor, reuse, and a re-scoring index source), two streams of mHC, two heads, and eight experts.  A
test that runs this model exercises the same code path as the real one, in under a second.
"""
from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

import numpy as np

from ascend.dsv41.config import TextConfig, build_layer_plan


def tiny_doc() -> dict:
    """A config that is the real one shrunk, with every structural feature still present."""
    return {
        "architectures": ["DeepseekV41ForCausalLM"],
        "model_type": "deepseek_v41",
        "bos_token_id": 0,
        "eos_token_id": 1,
        "pad_token_id": 2,
        "quantization_config": {"quant_method": "ascend"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "vocab_size": 32,
            "hidden_size": 64,
            "moe_intermediate_size": 16,
            "num_hidden_layers": 4,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "qk_rope_head_dim": 2,
            "q_lora_rank": 8,
            "o_lora_rank": 4,
            "o_groups": 2,
            "hidden_act": "silu",
            "swiglu_limit": 10.0,
            "rms_norm_eps": 1e-20,
            "use_cache": True,
            "tie_word_embeddings": False,
            "max_position_embeddings": 64,
            "rope_theta": 10000,
            "rope_scaling": {"rope_type": "yarn", "factor": 16, "beta_fast": 32, "beta_slow": 1,
                             "original_max_position_embeddings": 65536},
            "n_routed_experts": 8,
            "n_shared_experts": 1,
            "num_experts_per_tok": 2,
            "scoring_func": "sqrtsoftplus",
            "topk_method": "noaux_tc",
            "norm_topk_prob": True,
            "routed_scaling_factor": 1.5,
            "sliding_window": 4,
            # layer 0 window-only; 1 full (owns both a compressor and the index keys); 2 reuse; 3 re-scores
            "compress_ratios": [0, 2, 2, 2],
            "compress_rope_theta": 160000,
            "kv_source_layer_ids": [1],
            "index_source_layer_ids": [1, 3],
            "index_n_heads": 2,
            "index_head_dim": 8,
            "index_topk": 4,
            "candidate_source_layer_id": 1,
            "candidate_topk_blocks": 4,
            "candidate_block_size": 2,
            "hc_mult": 2,
            "hc_sinkhorn_iters": 20,
            "hc_eps": 1e-6,
            "engram_layer_ids": [],
            "num_nextn_predict_layers": 0,
        },
    }


def tiny_config() -> TextConfig:
    return TextConfig.from_dict(tiny_doc())


def _f32(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(x, dtype=np.float32).tobytes()


def _bf16(x: np.ndarray) -> bytes:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    r = ((u >> 16) & 1) + 0x7FFF
    return ((u + r) >> 16).astype(np.uint16).tobytes()


def _i8(x: np.ndarray) -> bytes:
    return np.ascontiguousarray(x, dtype=np.int8).tobytes()


def write_safetensors(path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> None:
    header, offset = {}, 0
    for name, (dtype, shape, blob) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + len(blob)]}
        offset += len(blob)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(t[2] for t in tensors.values()))


def make_tiny_checkpoint(root: Path, cfg: TextConfig | None = None, seed: int = 11) -> TextConfig:
    """Write a complete, self-consistent W8A8 checkpoint for `cfg` (two shards, so the reader is exercised)."""
    cfg = cfg or tiny_config()
    plan = build_layer_plan(cfg)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(cfg.raw, indent=1))
    rng = np.random.default_rng(seed)
    H, V = cfg.hidden_size, cfg.vocab_size
    ih, ihd = cfg.index_n_heads, cfg.index_head_dim
    t1: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    t2: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    def bf(dst, name, shape, scale=0.02):
        dst[name] = ("BF16", shape, _bf16(rng.standard_normal(shape) * scale))

    def f32(dst, name, shape, scale=0.02):
        dst[name] = ("F32", shape, _f32(rng.standard_normal(shape) * scale))

    def ones(dst, name, shape):
        dst[name] = ("BF16", shape, _bf16(np.ones(shape, np.float32)))

    def q8(dst, name, shape, scale=0.02):
        """An int8 weight with its per-output-channel scale (and a zero offset, as symmetric quant is)."""
        out = shape[0]
        dst[name] = ("I8", shape, _i8(rng.integers(-127, 128, shape)))
        dst[name + "_scale"] = ("F32", (out, 1),
                                _f32((rng.random((out, 1)) * 0.01 + 1e-3).astype(np.float32), ))
        dst[name + "_offset"] = ("F32", (out, 1), _f32(np.zeros((out, 1), np.float32)))

    bf(t1, "embed.weight", (V, H), 0.05)
    bf(t2, "head.weight", (V, H), 0.05)
    ones(t2, "norm.weight", (H,))
    for li in range(cfg.num_hidden_layers):
        role = plan.layer(li)
        r = role.compress_ratio
        B = f"layers.{li}."
        d = t1 if li % 2 == 0 else t2
        bf(d, B + "attn.wq_a.weight", (cfg.q_lora_rank, H))
        ones(d, B + "attn.q_norm.weight", (cfg.q_lora_rank,))
        q8(d, B + "attn.wq_b.weight", (cfg.num_attention_heads * cfg.head_dim, cfg.q_lora_rank))
        bf(d, B + "attn.wkv.weight", (cfg.head_dim, H))
        ones(d, B + "attn.kv_norm.weight", (cfg.head_dim,))
        bf(d, B + "attn.wo_a.weight", (cfg.wo_a_out, cfg.wo_a_in))
        bf(d, B + "attn.wo_b.weight", (H, cfg.wo_a_out))
        bf(d, B + "attn.attn_sink", (cfg.num_attention_heads,), 0.1)
        ones(d, B + "attn_norm.weight", (H,))
        ones(d, B + "ffn_norm.weight", (H,))
        n_mix = (2 + cfg.hc_mult) * cfg.hc_mult
        for part in ("attn", "ffn"):
            f32(d, B + f"hc_{part}_fn", (n_mix, cfg.hc_mult * H), 0.02)
            f32(d, B + f"hc_{part}_base", (n_mix,), 0.05)
            f32(d, B + f"hc_{part}_scale", (3,), 1.0)
        if role.is_kv_source:
            bf(d, B + "attn.compressor.wkv.weight", (cfg.head_dim, H))
            ones(d, B + "attn.compressor.norm.weight", (cfg.head_dim,))
            if r > 1:
                bf(d, B + "attn.compressor.wgate.weight", (cfg.head_dim, H))
        if role.is_index_source:
            if role.is_kv_source:
                bf(d, B + "attn.indexer.wk.weight", (ihd, cfg.head_dim))
                ones(d, B + "attn.indexer.k_norm.weight", (ihd,))
            q8(d, B + "attn.indexer.wq_b.weight", (ih * ihd, cfg.q_lora_rank))
            bf(d, B + "attn.indexer.weights_proj.weight", (ih, H))
        bf(d, B + "ffn.gate.weight", (cfg.n_routed_experts, H))
        bf(d, B + "ffn.gate.bias", (cfg.n_routed_experts,), 0.05)
        for p in ("w1", "w2", "w3"):
            q8(d, B + f"ffn.shared_experts.{p}.weight",
               (cfg.moe_intermediate_size, H) if p != "w2" else (H, cfg.moe_intermediate_size))
        for e in range(cfg.n_routed_experts):
            for p in ("w1", "w2", "w3"):
                q8(d, B + f"ffn.experts.{e}.{p}.weight",
                   (cfg.moe_intermediate_size, H) if p != "w2" else (H, cfg.moe_intermediate_size))
    write_safetensors(root / "quant_model_weights-00001-of-00002.safetensors", t1)
    write_safetensors(root / "quant_model_weights-00002-of-00002.safetensors", t2)
    return cfg


class TinyModel:
    """A temp directory holding the checkpoint, and the reader over it."""

    def __init__(self, seed: int = 11):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "tiny-dsv41"
        self.cfg = make_tiny_checkpoint(self.root, seed=seed)

    def close(self):
        self._tmp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
