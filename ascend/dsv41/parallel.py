"""ascend/dsv41/parallel.py - one request across several cards: tensor parallelism (dense) + expert
parallelism (routed experts), driven from one process.

The engine's existing multi-card layout splits the **experts by layer**: card 0 runs every layer's dense
work and each card holds ten layers' routed experts.  This module is the other split, the one a single
request needs:

* **TP** - the dense weights are sharded `W/4`: attention is column-parallel over heads (`wq_b` rows,
  `wo_a` rows, `wo_b` columns, `attn_sink` entries) and row-parallel at the output, so each card computes
  sixteen heads and a partial output; the shared expert is column-parallel in `w1/w3`, row-parallel in
  `w2`, so each card computes a partial shared-expert output.  `wq_a`/`wkv` are replicated (a 5-13 MB
  weight every head needs whole), the router and the norms are replicated (they are elementwise or tiny
  and every card must select the same experts), and the KV planes are **replicated** - MLA's KV latent is
  shared by all heads, so every card builds the same latent and holds the same cache.  That is what keeps
  attention itself collective-free.
* **EP** - a routed expert `e` is owned by card `e % W` (strided, see `Dsv41._ep_owns`: a learned router
  need not be uniform over ids, and a contiguous block can hand one card every layer's hot experts).  Each
  card fetches and multiplies only its own experts, so each card's tier is 40 layers x 96 experts instead
  of 10 layers x 384 - the same 3 840 (layer, expert) candidates per card.

Two collectives per layer, one after attention and one after the MoE.  They are hand-rolled from
``copy_`` because a ``torch.distributed`` HCCL group wants one process per card while this engine is one
process driving four cards (see `ascend/bench_hccs.py`, which also measures what they cost: a 20 KB
all-reduce is ~1.5 ms, so 80 of them are ~117 ms of a 409.6 ms decode step).  Only ``copy_`` may cross a
card boundary - a kernel cannot take a cross-device operand (that fault is recorded in bench_hccs.py).

Correctness, honestly: the sums are fp32 and the split changes only the *order* of the additions, but the
engine promises 2e-3 and the split does **not** hold it on the real bf16 checkpoint.  On the tiny fp32 model
(``ascend/tests/test_parallel.py``, TP/EP=2 against TP/EP=1 on CPU, whole chunk and one token at a time) the
split is within 2e-3.  On the 40-layer model at bf16, `bench_tp --parity` measures 8.1e-3 relative for the
dense path alone and 3.1e-1 with the routed experts: reshaping a 8192-column bf16 matmul into four
2048-column ones changes the fp32 accumulator order, and at 40 layers that is enough to flip a router top-k
tie.  Meeting 2e-3 would need fp32 matmuls.  That is one of the two independent reasons TP is not the
engine's path (the other is §18.4-18.5 of docs/ASCEND-DSV41.md: it is slower at T=1 and at prefill).

No graph capture here.  A captured body bakes its card and its addresses and a cross-card copy inside
capture is a hard error, so the captured decode stays single-card; a captured TP step would need one
graph per card per half-layer plus 80 eager collectives between replays, which is exactly the 117 ms
above plus the graph launch cost.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from .checkpoint import Checkpoint
from .config import TextConfig, build_layer_plan
from .model import Dsv41, ExpertStreamer, State, Weights, rms

GIB = 1 << 30


class TPComm:
    """All-reduces over a set of cards in one process: sum on rank 0, then broadcast.

    Every cross-card transfer is a ``copy_`` into a local buffer and the addition runs on the card that
    owns the sum.  That is not a limitation of this code: a kernel whose operands live on two cards takes
    an ``MTE accesses an invalid GM address`` fault, measured on this box (``bench_hccs.py``).

    The sum is in rank order, fp32 in, fp32 out (callers pass fp32), and the result lands on every rank.
    On ``cpu`` (the tests) every "card" is the same device and the copies are local.
    """

    def __init__(self, devices: list[str]):
        self.devices = [str(d) for d in devices]
        self.size = len(self.devices)
        self._indices = [int(d.split(":")[-1]) if ":" in d else 0 for d in self.devices]
        self._scratch: dict[int, torch.Tensor] = {}
        self.collectives = 0
        self.bytes_moved = 0

    @contextlib.contextmanager
    def _on(self, rank: int):
        """Make `rank`'s card current for allocations and stream queries; restore afterwards."""
        if not hasattr(torch, "npu") or not self.devices[rank].startswith("npu"):
            yield
            return
        previous = torch.npu.current_device()
        if previous != self._indices[rank]:
            torch.npu.set_device(self._indices[rank])
        try:
            yield
        finally:
            if previous != self._indices[rank]:
                torch.npu.set_device(previous)

    def _buffer(self, rank: int, like: torch.Tensor) -> torch.Tensor:
        buf = self._scratch.get(rank)
        if buf is None or buf.shape != like.shape or buf.dtype != like.dtype or buf.device != like.device:
            buf = torch.empty_like(like)
            self._scratch[rank] = buf
        return buf

    def all_reduce(self, parts: list[torch.Tensor]) -> list[torch.Tensor]:
        """Sum the per-rank tensors and leave the sum on every rank.  Returns the same list, filled."""
        n = len(parts)
        if n == 1:
            return parts
        if len(parts) != self.size:
            raise ValueError(f"{len(parts)} parts for a group of {self.size}")
        base = parts[0]
        self.collectives += 1
        for r in range(1, n):
            buf = self._buffer(r, base)
            with self._on(0):
                buf.copy_(parts[r])            # the one cross-card move of this term
                base.add_(buf)                 # the addition runs where `base` lives
            self.bytes_moved += base.numel() * base.element_size() * 2
        for r in range(1, n):
            with self._on(r):
                parts[r].copy_(base)            # broadcast the sum back
            self.bytes_moved += base.numel() * base.element_size()
        return parts

    def gather_rows(self, parts: list[torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        """Concatenate per-rank vocabulary shards along the last axis into ``out`` (on rank 0)."""
        width = out.shape[-1] // self.size
        with self._on(0):
            out[..., :width].copy_(parts[0])
            for r in range(1, self.size):
                out[..., r * width:(r + 1) * width].copy_(parts[r])
        self.bytes_moved += out.numel() * out.element_size() * 2
        return out


class ParallelState:
    """One conversation's KV, replicated: one `State` per rank, and the host cursors the forward reads.

    The KV planes are per card, so a TP/EP conversation owns one per rank; they carry the same values
    because every card builds the same latent and the attention state is written identically.  The mirror
    fields (`next_pos`, `token_ids`, `tokens`) are what the forward and the caller read on the host.
    """

    def __init__(self, cfg: TextConfig, devices: list[str], dtype, max_context: int = 8192):
        plan = build_layer_plan(cfg)
        self.cfg = cfg
        self.states = [State(cfg, plan, d, dtype, max_context=max_context) for d in devices]
        self.next_pos = 0
        self.tokens = 0
        self.token_ids: list[int] = []
        self.topk = None
        self.candidates = None

    @property
    def pos(self):
        return self.states[0].pos

    @property
    def kv(self):
        return self.states[0].kv

    def rewind(self, n: int = 1) -> None:
        if n < 0 or n > self.next_pos:
            raise ValueError(f"cannot rewind {n} of {self.next_pos} tokens")
        for st in self.states:
            st.rewind(n)
        self.next_pos -= n
        self.tokens -= n
        del self.token_ids[len(self.token_ids) - n:]

    def advance(self, ids: list[int]) -> None:
        for st in self.states:
            st.token_ids.extend(ids)
            st.pos += len(ids)
            st.next_pos += len(ids)
            st.kv.advance(len(ids))
            st.tokens += len(ids)
        self.token_ids.extend(ids)
        self.next_pos += len(ids)
        self.tokens += len(ids)


class TPEngine:
    """A TP+EP group: one `Dsv41` per card, stepped together, one collective after each layer half."""

    def __init__(self, cfg: TextConfig, ranks: list[Dsv41], devices: list[str]):
        self.cfg = cfg
        self.ranks = ranks
        self.devices = [str(d) for d in devices]
        self.comm = TPComm(self.devices)
        self.device = self.devices[0]                  # the caller's device: rank 0's
        self.dtype = ranks[0].dtype
        self.graphs = False
        self.timers: dict[str, float] | None = None
        self.host_mixes = True
        self.expert_cards = None

    # -- the interface the engine's callers use ------------------------------------------------
    def state(self, max_context: int = 8192) -> ParallelState:
        return ParallelState(self.cfg, self.devices, self.dtype, max_context=max_context)

    def release_graphs(self) -> None:
        for r in self.ranks:
            r.release_graphs()

    def graph_summary(self, state) -> str:
        return "graphs: off (TP/EP runs eager; a capture cannot hold a cross-card collective)"

    def expert_stats(self) -> dict[str, int]:
        total: dict[str, int] = {}
        for r in self.ranks:
            stats = r.weights.experts.stats() if hasattr(r.weights.experts, "stats") else {}
            for k, v in stats.items():
                total[k] = total.get(k, 0) + int(v)
        return total

    # -- the forward ----------------------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, ids, state: ParallelState, logits_last_only: bool = True,
                compute_logits: bool = True):
        """One chunk (a prompt, or a single decode token) through 40 layers on all ranks."""
        cfg, ranks = self.cfg, self.ranks
        W = len(ranks)
        host_ids = [int(t) for t in (ids.tolist() if torch.is_tensor(ids) else ids)]
        T = len(host_ids)
        start = state.next_pos
        host_positions = list(range(start, start + T))
        positions = [torch.tensor(host_positions, device=r.device, dtype=torch.long) for r in ranks]

        # -- embedding, sharded by vocabulary: each rank looks up the ids it owns into zero rows and the
        # all-reduce sums the one non-zero contribution per token.  One collective a chunk.
        rows_v = cfg.vocab_size // W
        h = []
        for r, model in enumerate(ranks):
            lo, hi = r * rows_v, (r + 1) * rows_v
            ids_r = torch.tensor(host_ids, device=model.device, dtype=torch.long)
            mask = (ids_r >= lo) & (ids_r < hi)
            hr = torch.zeros(T, cfg.hidden_size, device=model.device, dtype=torch.float32)
            if bool(mask.any()):
                hr[mask] = model.weights.embed[ids_r[mask] - lo].float()
            h.append(hr)
        h = self.comm.all_reduce(h)
        h = [hr.unsqueeze(1).repeat(1, cfg.hc_mult, 1) for hr in h]

        # -- Engram is a host lookup keyed by the token history: hashed once, applied per rank with the
        # same rows (its projection weights are replicated, the tables are mapped not loaded).
        engram_rows = None
        if ranks[0].engram is not None:
            history = state.token_ids + host_ids
            engram_rows = ranks[0].engram.hashes(history)[len(state.token_ids):]

        pre = [torch.zeros(T, cfg.hc_mult, device=r.device, dtype=torch.float32) for r in ranks]
        for p in pre:
            p[:, 0] = 1.0

        for li in range(cfg.num_hidden_layers):
            # first half: mHC, norm, attention - each rank on its own heads, partial output
            halves = [r._attn_half(li, h[i], pre[i], state.states[i], positions[i], host_positions,
                                   engram_rows) for i, r in enumerate(ranks)]
            o = self.comm.all_reduce([hd[0] for hd in halves])
            # second half: post-MHC, the FFN pre-mix and norm
            ffns = []
            for i, r in enumerate(ranks):
                oi, res, a_pre, a_post, a_comb = halves[i]
                ffns.append(r._ffn_half(li, o[i], res, a_pre, a_post, a_comb))
            # experts: each rank its own 96-of-384 subset plus its shared-expert shard
            y = self.comm.all_reduce([r.moe(li, f[0]) for i, (r, f) in enumerate(zip(ranks, ffns))])
            nxt = []
            for i, r in enumerate(ranks):
                _x, res, f_pre, f_post, f_comb = ffns[i]
                with r.phase("moe"):
                    nxt.append(r.hc_post(y[i], res, f_post, f_comb))
                pre[i] = f_pre
            h = nxt

        state.advance(host_ids)
        if not compute_logits:
            return None
        # -- final norm and the head, sharded by vocabulary; rank 0 returns the full row.  The norm is
        # replicated but the result must live on the card whose head shard multiplies it, so each rank
        # normalises its own copy of the same row (same values, four small elementwise calls).
        sl = slice(-1, None) if logits_last_only else slice(None)
        xs = [rms(r.hc_pre(h[i][sl], pre[i][sl]), ranks[i].weights.final_norm, ranks[i].eps)
              for i, r in enumerate(ranks)]
        logits = [r.mm(xs[i], r.weights.head) for i, r in enumerate(ranks)]
        out = torch.empty(xs[0].shape[0], cfg.vocab_size, device=ranks[0].device, dtype=torch.float32)
        return self.comm.gather_rows(logits, out)

    def close(self) -> None:
        for r in self.ranks:
            try:
                r.weights.experts.close()
            except Exception:                                   # noqa: BLE001 - teardown is best effort
                pass
        self.ranks.clear()


@dataclass
class ParallelRuntime:
    cfg: TextConfig
    engine: TPEngine
    ckpt: Checkpoint
    dtype: torch.dtype
    devices: list[str]

    @property
    def model(self) -> TPEngine:
        return self.engine

    def state(self, max_context: int = 8192) -> ParallelState:
        return self.engine.state(max_context)

    def expert_slots(self) -> int:
        return sum(getattr(r.weights.experts, "cache_slots", 0) for r in self.engine.ranks)

    def resident_bytes(self) -> int:
        return sum(r.weights.resident_bytes() for r in self.engine.ranks)

    def close(self) -> None:
        self.engine.close()
        self.ckpt.close()


def build(*, model: str, devices: list[str], hbm_share: float = 0.5, dtype: str = "bfloat16",
          expert_slots_per_card: int | None = None, expert_reserve_gib: float = 6.0,
          io_threads: int = 6, expert_batch: int = 8, verbose: bool = False,
          no_experts: bool = False) -> ParallelRuntime:
    """Build a TP/EP group: one sharded `Weights` and one expert streamer per card.

    ``expert_slots_per_card`` pins the tier (an A/B needs the two arms to see the same tier); left None,
    each card's tier is sized from what is left of its HBM share after its dense shard, exactly as
    `runtime.build` does for one card.
    """
    from . import runtime as rt

    model_dir = __import__("pathlib").Path(model)
    cfg = TextConfig.from_json(model_dir / "config.json")
    dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    W = len(devices)
    # claim the share of every card first, so the tier arithmetic sees the same numbers one card would
    resolved = [rt.resolve_device(d, hbm_share) for d in devices]
    ckpt = Checkpoint(model_dir)
    ranks: list[Dsv41] = []
    for r in range(W):
        dev = resolved[r]
        if no_experts:
            from .model import NullExpertSource
            src = NullExpertSource(ckpt, cfg, dt)
        else:
            src = ExpertStreamer(ckpt, cfg, dev, dt, cache_slots=0, io_threads=io_threads,
                                 max_batch=expert_batch, layer_devices=None, landing_blocks=False)
        weights = Weights(ckpt, cfg, dev, dt, expert_source=src, verbose=verbose,
                          shard=(r, W)).load()
        if expert_slots_per_card is None:
            per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
            free = rt.free_hbm_bytes(dev) - int(expert_reserve_gib * GIB)
            slots = min(int(max(0, free) // per_expert), cfg.num_hidden_layers * cfg.n_routed_experts // W)
        else:
            slots = int(expert_slots_per_card)
        if not no_experts:
            src.set_slots({dev: slots})
        ranks.append(Dsv41(cfg, weights, dev, dt, graphs=False, tp_rank=r, tp_size=W))
        if verbose:
            print(f"  tp rank {r} on {dev}: {slots} tier slots", flush=True)
    return ParallelRuntime(cfg=cfg, engine=TPEngine(cfg, ranks, resolved), ckpt=ckpt, dtype=dt,
                           devices=resolved)
