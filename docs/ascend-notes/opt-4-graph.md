# opt-4 — Cutting the 32 242 operator calls a decode step with per-layer NPU graphs

**Verdict: available, proven on this stack, and worth ~2.0x on the token.**
`torch.npu.graph` / `torch.npu.NPUGraph` / `torch.npu.graph_pool_handle` exist in the installed
`torch_npu 2.10.0.post7`, capture+replay works on CANN 9.2.1 / driver 26.0.rc1 (verified, §2), the hot
`npu_quant_matmul` op captures cleanly, and the vendor's own vLLM-Ascend (`v0.30.0`) uses exactly this API
for its decode graph — not raw `aclmdlRICapture`. What it cannot do is remove the 606 ms expert fetch: that
is host `preadv` and must be moved *outside* the captured region and then overlapped (§6, Phase 2).

Read-only probes only. No model was built, no tokens generated, no graph of the real model captured.
All device work below was 4–1024 element tensors.

---

## 1. The number this direction is aiming at

The step's own arithmetic says the host, not the device, is the engine:

| | value | source |
|---|---:|---|
| wall per decode token | 1 661 ms | task facts |
| device time per step | 164 ms | `msprof`, `docs/ASCEND-DSV41.md:277-278` |
| operator calls per step | 32 242 | " |
| **host cost implied per call** | **46.4 us** | (1661-164)/32 242 |

That 46 us is not a guess about this model — it reproduces independently on the same container with
200 trivial elementwise ops: **47.4 us/op** eager, **1.88 us/op** inside a replay, one launch of 376 us
for the whole 200-op graph (25.2x, §2.4). The engine is paying a PyTorch dispatcher + ACL launch toll
32 242 times per token, and a graph is the only mechanism that removes it without changing arithmetic.

Projection, using the measured per-node and per-launch costs:

| component | today | captured (Phase 1) | note |
|---|---:|---:|---|
| host dispatch of 32 242 ops | 891 ms | ~61 ms | 80 replays x 15.7 us + 32 242 x 1.88 us |
| expert fetch (host `preadv`) | 606 ms | 606 ms | **not removable by capture** |
| device time | 164 ms | 164 ms | unchanged; Phase 3 attacks it |
| **total** | **1 661 ms** | **~831 ms** | **~2.0x** |

Phase 2 (overlap the fetch behind the previous layer's replay) then takes the floor to
`max(606, 164+)` ≈ 700 ms → ~2.4x. Phase 3 (`super_kernel_optimize`) is the only one of the three that
touches the 164 ms device term.

---

## 2. What is actually available, with evidence

### 2.1 The API surface, imported from the container

    $ docker exec maya-ascend-dev python3 -c "import torch, torch_npu; ..."
    torch 2.10.0+cpu     torch_npu 2.10.0.post7
    hasattr(torch.npu, 'graph') -> True
    ['NPUGraph', 'NpuGraphOpHandler', '_npugraph_handlers', 'graph', 'graph_pool_handle',
     'graph_task_group_begin', 'graph_task_group_end', 'graph_task_update_begin',
     'graph_task_update_end', 'graphs', 'make_graphed_callables', 'npugraph_ex',
     'register_npu_graph_handler']

Exact names, all live:

| API | where | signature / note |
|---|---|---|
| `torch.npu.graph` | `torch_npu/npu/graphs.py:591` region; class `torch_npu.npu.graphs.graph` | `(npu_graph, pool=None, stream=None, auto_dispatch_capture=False, capture_error_mode='global')` |
| `torch.npu.NPUGraph` | `graphs.py:591`, base `torch_npu._C._NPUGraph` | `capture_begin, capture_end, replay, reset, pool, update, debug_dump, super_kernel_optimize` |
| `torch.npu.graph_pool_handle` | `graphs.py:76` | `()` → opaque pool token |
| `torch.npu.is_current_stream_capturing` | `graphs.py:67` | `()` → bool |
| `torch.npu.graph_task_group_begin/end` | `graphs.py:87,91` | host-updatable region inside a capture |
| `torch.npu.graph_task_update_begin/end` | `graphs.py:95,99` | re-bind an updatable region without recapture |
| `torch.npu.super_kernel_scope_begin/end` | `graphs.py:403,407` | fuse scope for the capture; also `torch.ops.npu.super_kernel_scope_begin` |
| `torch.npu.make_graphed_callables` | `graphs.py` | module-level helper, unused by the vendor |

`capture_error_mode` is documented as "specifies the `aclmdlRICaptureMode` for the graph capture stream"
(`graphs.py:615-619`) — this *is* ACL's `aclmdlRICapture*` underneath, wrapped by torch_npu rather than called
directly; there is no Python-visible `aclmdlRICapture` symbol, and the wrapper is the supported entry point.

### 2.2 The vendor's own reference: `torch.npu.graph`, not `aclmdlRICapture`

`grep -rn "torch.npu.graph\|NPUGraph\|aclmdlRICapture" /vllm-workspace/vllm-ascend/vllm_ascend/` returns
30 hits and **zero** `aclmdlRICapture`. The ones that define the pattern:

- `compilation/acl_graph.py:58` — `aclgraph: torch.npu.NPUGraph | None = None`
- `compilation/acl_graph.py:117` — `self.graph_pool = current_platform.get_global_graph_pool()`
- `compilation/acl_graph.py:210-212` — `with torch.npu.graph(aclgraph, pool=self.graph_pool):` and inside it
  `with super_kernel_scope("full_model", self.enable_super_kernel):`
- `compilation/acl_graph.py:247` — `aclgraph.super_kernel_optimize(optimize_options={"dcci_after_kernel_end": [".*"]})`
- `compilation/updatable_graph.py:120-135` — `register_task()`: `graph_task_group_begin(stream)` → run op →
  `graph_task_group_end(stream)` → handle, later re-driven by `apply()` (`:73-76`) via
  `torch.npu.graph_task_update_begin(update_stream, handle)`
- `worker/v2/utils.py:195-210` — `torch_npu_graph_wrapper`, `with torch.npu.graph(*args, **kwargs), super_kernel_scope("full_model", ...)`
- `worker/encoder_acl_graph.py:297,301` — the minimal form: `graph = torch.npu.NPUGraph()` … `with torch.npu.graph(graph, self.graph_pool):`
- `_310p/attention/attention_v1.py:193-200` — the vendor's own guard: *"Pageable H2D is illegal under NPU
  GLOBAL ACLGraph capture … move it outside torch.npu.graph()"*, keyed on `torch.npu.is_current_stream_capturing()`

The stack is therefore supported at the level this engine needs, the vendor ships decode-graph capture through
it, and `graph_task_group_*` exists precisely for this model's case: an op whose host-side arguments change
per replay.

### 2.3 Capture and replay work here (probe)

    x = ones(4); g = torch.npu.NPUGraph()
    with torch.npu.graph(g): y = (x+1)*2
    CAPTURE OK; y = [0,0,0,0]        <-- capture does NOT execute
    x.fill_(5); g.replay()
    REPLAY OK; y = [12,12,12,12]     <-- correct

Two facts change the engineering: **capture does not run the work** (the first replay is warm-up; a captured
but never-replayed layer silently produces nothing), and replay reads whatever is in the input buffers at
replay time — which is how the expert weights get in.

### 2.4 The per-call cost, measured on this container

    200 tiny ops (512-elem add+mul), eager:   9 479 / 9 539 / 11 497 us  -> 47.4 us/op
    same 200 ops, captured and replayed:        375.8 / 375.8 / 382.0 us  -> 1.88 us/node
    2-node graph, replay:                        15.7 us/launch (4000 replays)
    40 graphs sharing one pool: captured in 0.25 s, replayed fine

1.88 us/node is in-graph node dispatch, not arithmetic (that is nanoseconds here); 15.7 us is the acl graph
launch. Both sit ~25x below the 47.4 us eager path — and 47.4 matches the 46.4 us the engine's profile
implies. The model is at the same 47 us/call toll, 32 242 times.

### 2.5 What capture does to the four primitives this code needs — probed

| probe | result | consequence for this engine |
|---|---|---|
| `.item()` / `.tolist()` inside capture | **RuntimeError** `AclrtSynchronizeStreamWithTimeout … 107027`, `Stream (stream_id=44) during the capture stage is not supported`; DFX hint: *default stream, CPU ops, multi-device tensors, dynamic control flow on tensor values* | `model.py:965`, `:942` are hard capture errors |
| pinned `copy_(non_blocking=True)` H2D inside capture | **allowed**, replays correctly (`z=21.0` after refilling the host buffer with 7) | expert H2D can live *inside* the graph if the source pinned address and the device landing address are fixed |
| `torch.npu.Event().synchronize()` inside capture | **no error raised** | the staging-ring safety net at `model.py:430-433` silently becomes a no-op — dangerous, keep it outside |
| `torch.zeros()` / allocations inside capture | **allowed**, replays correctly | the `torch.zeros` at `model.py:967` and the `float()` temporaries are fine |
| host int used as a slice index at capture, changed before replay | **silently wrong**: captured with `idx=3`, replayed after `idx=9`, the write still landed in columns 3-6 | `model.py:830` (`swa[:, host_positions[t]] = kv[t]`) and every other host-positioned store are *silently* baked to the capture-time position |
| replay after an input resize | **no error, no effect** — old shape replayed, values stale | dynamic shapes (`entries`, `topk=min(index_topk,entries)`) will not be caught by the runtime |
| `npu:0 -> npu:1` copy inside capture | **RuntimeError** (multi-device tensors listed as a common cause) | with `--expert-devices`, the `x.to(dev)` hops at `model.py:928-930` cannot be inside a graph |
| `torch_npu.npu_quant_matmul` inside capture | **allowed**, replays | the hot expert op is capturable |
| `super_kernel_optimize()` on a 20-node graph | works, result identical before/after (22.025 → 23.027 after changing input) | Phase 3 is usable |

The three silent-failure rows are the ones that matter: this stack does **not** protect you — a wrong graph
produces plausible numbers, not an exception.

---

## 3. Capture-safe vs capture-hostile, line by line in `Dsv41.forward`

`forward` is `model.py:988-1040`; the layer loop is `:1012-1030`. Per layer, 806 of the 32 242 calls live.

### 3.1 Safe as-is (pure device ops, shape-stable at T=1)

| region | lines | note |
|---|---|---|
| `mm` (`F.linear`) | `:653-660` | fixed shapes; fp32 upcast |
| `rms`, `hc_pre`, `hc_post` | `:149-153`, `:688-695` | 5 ops; fixed [1,4,d] |
| `hc_mixes`, **device branch** | `:676-686` | 20 Sinkhorn iterations, ~4x4 |
| `rope.apply`, `sparse_attn`, `out_projection` | `:102-123`, `:790-798`, `:854-862` | fixed shapes; `T=1` |
| `swiglu`, `expert_mm_q`, `expert_mm`, `dequant`, `per_token_int8` | `:865-906`, `:57-64` | `npu_quant_matmul` verified capturable |
| `softplus`/`topk`/`gather` router | `:918-921` | top-6 is a fixed count |
| all allocations | `:967`, `:941` | verified |

### 3.2 Hostile — must be removed or restructured before anything is captured

1. **`idx[0].tolist()`** `:965` — per-layer device->host sync, 40 per step, hard capture error. This is the
   single line that makes the current MoE uncapturable. It exists because `fetch_int8` takes `list[int]`.
2. **`torch.unique(idx).tolist()`** `:942` — the T>1 path; same failure, but decode never takes it.
3. **`fetch_int8`** `:458-503` in full: `os.preadv` (`:401`), `self._events[slot].synchronize()` (`:433`),
   `.numpy()` on the staging buffer (`:434`), `torch.from_numpy(ss).to(device)` from a *temporary* numpy
   array (`:451-452`), LRU dict mutation (`:498-502`), `torch.npu.empty_cache()` on the OOM path (`:496`) —
   `empty_cache` is in the class of actions the `capture_begin` docstring (`graphs.py:611-617`) calls unsafe.
   The 606 ms lives here; it is host work, cannot be captured, and must be *moved* and *overlapped*.
4. **`hc_mixes` host branch** `:668-675` — `got.cpu().numpy()` (`:671`) and `from_numpy(...).to(dev)` (`:674`)
   is a pageable H2D, the exact pattern the vendor's `attention_v1.py:193-200` refuses under capture.
   4 host round-trips per layer, 160 per step. **This optimisation must be switched off inside the graph**
   (`self.host_mixes = False`, `:622`) — see Phase 0b.
5. **Host-positioned state writes**: `state.comp_kv_raw[li][:, positions] = kv` (`:718-719`),
   `state.index_k[li][:, g0:g0+k]` (`:753`), `state.long_kv[li][:, g0:g0+lat]` (`:826`),
   `state.swa[li][:, host_positions[t]] = kv[t]` (`:829-830`). Probe D proves these are *silently* baked.
6. **Compress/indexer shapes**: `entries = (host_positions[-1]+1)//r` (`:758`), `topk=min(index_topk, entries)`
   (`:771`), `candidate_blocks` width/`nb` (`:780-787`). Shapes grow with context; probe F says the runtime
   will not complain.
7. **Cross-device** `:928-930` — `x.to(dev)`, `.to(x.device)`. Probe G: capture raises.
8. **`engram.embedding`** `engram.py:185-189` — `torch.from_numpy(gathered).to(device)` per Engram layer per
   token: pageable H2D, illegal in a capture; the lazy `self._block_on[dev]` at `engram.py:206-208` must be
   warmed first. **`phase()`** `:632-650` does `torch.npu.synchronize()` (`:643`/`:649`) whenever `timers` is
   on, so capture must run with `self.timers is None`. `state.token_ids.extend` `:1033` and `engram.hashes`
   `:1007-1008` are host bookkeeping and fine *outside*.

**Answer to "whole layer or only parts": neither exactly — the layer splits in two at the routing read, and
both halves are separately capturable.** That seam is forced by the fetch and is the same seam the CUDA engine
uses (`include/strata/core/session.hpp:116-131`: `pre[l]` up to `moe_route`, `post[l]` = `moe_finish` +
`gr_write`, "launched AFTER the host has run the pool"). What is *not* capturable as one unit is the MoE with
the fetch inside it, nor the indexer at growing context unless bucketed.

## 4. Phase 0 — the three changes that make a layer capturable

These are prerequisites, not optimisations, and each is independently defensible.

**0a. Device-side state addressing.** `state.pos` already exists as a device tensor (`model.py:570`) and is
maintained (`:1034`) but never used for addressing — every store indexes with a host int. Replace:
`swa[:, p] = kv` → `swa.index_copy_(1, pos_idx, kv)`; the window *read* `swa[:, w0:p+1]` → a device-side
gather with `(pos - win + 1 + arange(win)) % max_context` computed from `state.pos`; the `long_kv`/`index_k`
stores likewise. Cost: one extra gather of ≤512 KiB per layer, which is nothing next to the 20 KB activation.
Calls: +2/layer, and it removes the class of silent-wrong-result bugs that probe D demonstrates.

**0b. Device mHC path, with the two mixes batched.** `host_mixes` (`:622`) was the right call before graphs
and is wrong inside one: it *adds* 160 host round-trips a step and cannot be captured at all. Inside the
graph, set `host_mixes=False` and stack the attention and FFN mixes into one `[2, 4*d] x [4*d, 24]` matmul, so
one batched Sinkhorn replaces two (9 000 in-graph nodes → ~4 500, i.e. ~8 ms of replay dispatch at 1.88 us).
No numerics change: the same formula on two rows.

**0c. Hoist what is position-only.** `Rope.angles` is rebuilt per `rope.apply` call (4x per layer: `:811`,
`:813`, `:824`, `:851`); compute the step's angles once outside the loop — ~1 600 calls/step. The routing ids
cannot be hoisted; they depend on layer `l`'s activation.

After Phase 0 a layer is: device-only ops, fixed shapes, device-side state addressing, one host sync (the
routing read) — still 806 calls, but *capturable*.

## 5. Phase 1 — two graphs per layer, one fixed expert landing zone

**Shape of the capture.** 40 layers x 2 graphs = 80 graphs sharing one `graph_pool_handle()` — the vendor does
the same (`acl_graph.py:117`). Capturing 40 graphs of this size took 0.25 s (probe), so ~1 s per process.

    for li in layers:
        g_pre[li].replay()                       # mHC + attention + router; writes x_li and idx_li
        torch.npu.synchronize()                  # the ONE host break, was 806 dispatches + 1 sync
        ids = idx_buf[li].cpu().tolist()         # 6 ints, fixed device address
        blocks = streamer.fetch_int8(li, ids)    # pread -> pinned ring -> H2D into landing[parity]
        g_post[li].replay()                      # 6 expert matmuls + shared expert + hc_post

**The landing zone is the whole trick.** `g_post[li]` bakes the address it reads, so all 40 `g_post` graphs
must read the *same* device buffer: a `[6, 3, 2304, 5120]` int8 slab = **203 MB**, double-buffered to
**406 MB** because `g_post[li]` is still reading parity 0 while the host fills parity 1 for layer `li+1`. For
scale, the pinned staging ring already costs `3 x 8 x 3 x 34 MB = 2.4 GB` of host RAM (`model.py:354`,
`:505-508`). If HBM is tight: one 203 MB buffer, no prefetch until the previous `g_post` drains.

**Why the expert count being variable is harmless.** For T=1 the routed set is `topk` of 384 with distinct
indices (`model.py:919`, `_routed_one` `:958-965`), so the count is **always 6** — the variability is *which*
experts, never how many. Unrolling six matmul triples at capture is shape-correct forever. This is the one
place where the model's own structure already gives the capture what it needs.

**What must be true, enumerated.** (i) T=1, always — prompt chunks take the eager path. (ii) `expert_batch`
fixed at 6 and `max_batch>=6` (`:350`) so `_ensure_staging` never reallocates (`:366-385`). (iii) The pinned
staging tensors keep their addresses for the life of the graphs. (iv) The routing ids arrive at a fixed device
address, written by `g_pre`, read by `cpu()` between replays. (v) `capture_error_mode="relaxed"`: the
streamer runs on 6 IO threads (`:360`) and device work from those threads is "an action in another thread"
that `global` mode errors on (`graphs.py:611-617`). (vi) `self.timers is None` during capture. (vii) All lazy
host->device materialisation is warmed first (`engram._block_on`, `:206-208`; the `rope` tables `:74-100`).
(viii) `entries` is handled by capture buckets — next paragraph.

**The one shape that still moves: `entries`.** The indexer computes over `entries = (pos+1)//r` compressed
entries (`:758`) and takes `topk=min(index_topk, entries)` (`:771`), so a graph captured at context 512 is
wrong at 4 096. Two honest options: (a) capture per power-of-two bucket of `entries` — over an 8 192 context
that is 11 buckets x 40 layers = 440 graphs, 440 captures at ~6 ms = 2.6 s amortised over thousands of
tokens, affordable but ugly; (b) leave the indexer out of `g_pre`, which costs 60 x 47 us x 40 = 113 ms/step —
worse than (a). Prefer (a), with the bucket table an explicit, loggable config.


## 6. Phase 2 — overlap the 606 ms, which is the bigger prize once the split exists

Today the fetch cannot overlap anything, because the host is the critical path for the whole step
(`fetch_int8` runs *inside* the layer, `model.py:972`). After Phase 1 the layer is two replays with a host gap
between them — and that gap is exactly where the next layer's I/O belongs:

    for li: g_pre[li].replay(); ids = ...; prefetch(li+1) into landing[1-p]; g_post[li].replay()

The device work per layer is only ~4 ms (164/40) against a ~15 ms fetch (606/40), so the step becomes
fetch-bound at ~700 ms → ~2.4x. Same structure, and the same conclusion, as the CUDA engine's measured note
that "a per-layer CPU pool cannot be hidden behind a strictly serial residual chain" (`session.hpp:124-126`) —
except here the pool is a disk read and it *can* be hidden, because the read of layer `li+1` does not depend
on the routing of layer `li`. Phase 2 is cheap (a second landing buffer, an `Event` per parity) and belongs in
the same change as Phase 1: it is where the second x lives.

---

## 7. Phase 3 — `super_kernel_optimize`, the only lever on the 164 ms

`NPUGraph.super_kernel_optimize` (`graphs.py:712`) takes `split_mode`, `stream_fusion`, `auto_op_parallel`,
`constant_codegen`, `dcci_before_kernel_start` / `dcci_after_kernel_end`, and
`aggressive_opt_strategies.{event_breaker_bypass, value_breaker_bypass, task_breaker_bypass}`. The vendor
enables it for its full-model decode graph (`acl_graph.py:247-252`, inside `super_kernel_scope("full_model",
...)` at `:212`), and it is on by default in that path.

This converts "32 242 captured nodes dispatched at 1.88 us" into "a few fused device tasks": it attacks the
61 ms of in-graph dispatch *and* the 164 ms of device time by fusing and overlapping chains that are
sequential in the Python source — notably the 4 500 Sinkhorn nodes from Phase 0b. Probed: acceptable and
numerically identical on a 20-node graph (22.025 before and after; 23.027 after changing the input). It is an
optimising rewriter, so parity must be re-gated on logits (§9), not assumed from the probe.

## 8. If capture is refused or too risky: the structural alternatives, ranked

**(a) A fixed-shape expert path that always computes exactly 6 experts — already true; worth 0 calls.**
`_routed_one` (`:958`) already walks exactly `cfg.num_experts_per_tok` ids, with no `unique`, no `where`, one
activation quantisation. The shape is fixed today. This is a *precondition* for capture, not an alternative
to it. Its only residual variability is the id values, which is what the landing zone absorbs.

**(b) Hoisting host work out of the loop — worth ~1 600 of 32 242 calls (5%), low risk.**
RoPE angles (4 calls/layer, §4.0c), `positions`/`pre`/`host_positions` bookkeeping (already cheap). The
routing `.tolist()` cannot be hoisted: it needs layer `l`'s activation. A 5% lever, strictly subsumed by
Phase 0c — do it there and count it once.

**(c) The dense 40 layers as one big fused sequence — not possible, and the arithmetic says why.**
A single whole-model graph is possible only if no layer needs the host, i.e. all 240 experts of the step are
resident: 240 x 33.8 MB = **8.1 GB** for one token's set, against a tier of 284 slots (9.4 GiB) on one card at
46-54% hit or 2 967 slots (98 GiB) on four cards at 74% (`docs/ASCEND-DSV41.md:255-256`). Even at 74% the
probability that all 40 layers are simultaneously hot is ~0, and a conversation's working set grows toward all
15 360 experts = 519 GB. Fusing across the residual chain is impossible anyway: the MoE sits *between* the two
dense halves of every layer (`:1017-1029`), so no contiguous run of dense work is longer than one layer's
attention+mHC. "One big fused sequence" is not a shortcut around capture — it is capture's endgame, and it is
out of reach on this memory budget. Same conclusion as the CUDA engine, which chose per-layer graphs
(`session.hpp:103-107`).

**(d) Batching the two mHC mixes — worth ~4 500 in-graph nodes (~8 ms) and 80 host round-trips, low risk.**
Do it *inside* the capture (Phase 0b), not instead of it. As a standalone change (no graphs) it would save
~14 device calls/layer but leave the 160 host round-trips that dominate — i.e. ≈0 on the median step, because
the round-trip is a sync, not the arithmetic. Its real value only appears at 1.88 us/node.

**Ranking: (d) and (b) as Phase 0 chores → Phase 1 capture → Phase 2 overlap → Phase 3 super-kernel.
(a) is a precondition; (c) is a dead end on 128 GB of HBM.**

## 9. Measurement plan

The machine drifts up to 50% between runs (`docs/ASCEND-DSV41.md:235`, `:299`), so an unpaired before/after is
worthless. Extend the harness that already knows this — `ascend/bench_step.py --ab` (`:208-235`) interleaves
arms in one process, 3 alternations, `--ab-steps`, greedy chaining, same-percentile comparison:

1. **One process, both arms, alternating, ABBA order.** `capture on` / `capture off` per repeat, 6 repeats x
   8 steps = 48 samples per arm, sharing one model, one `State`, one expert tier, one prompt
   (`(i*7+3) % 40000` as at `bench_step.py:211`) — only the graph wrapper differs.
2. **Paired estimator, not two medians.** Ratio within each adjacent A/B pair; report the median of the 48
   ratios with a bootstrap 95% CI. Drift common to a pair cancels; marginal medians do not, and that is how a
   1.42x was once read as 0.92x (`ASCEND-DSV41.md:230-232`).
3. **Warm-up that is not counted**: 3 prompt tokens + 3 decode steps discarded per arm — capture does not
   execute (probe §2.3), so a discarded replay is mandatory.
4. **Two drift-free counters as a second signal.** `msprof`'s `op_summary` op count and device time are
   absolute; assert them separately (op count *should* stay ~32 242 — the ops still execute, they are just no
   longer dispatched by the host — and device time must not regress).
5. **Capture budget**: ≤2 s for 80 graphs, ≤64 MB pool growth per graph beyond the 406 MB landing zone, one
   recapture per `entries` bucket crossing (logged).

**Acceptance criteria — all must hold, or the change does not land:**

- **Correctness (gate, not a metric):** 64 greedy tokens produce **identical token ids** to the eager path
  from the same prompt; `max|Δ| ≤ 2e-2` on per-step logits with 100% top-1 agreement over 32 steps; layer
  0/19/39 activations within 1e-2 relative. One mismatch fails, including after `super_kernel_optimize` — the
  toy-graph probe is not evidence for 800-node graphs.
- **Speed:** median paired ratio ≤ 0.60 (≥1.67x) for Phase 1+2, bootstrap CI upper ≤ 0.70. Phase 1 alone
  should land ~0.50 (831/1661); 0.60 is the pass line so the claim survives a bad window.
- **No silent-shape regression:** replay each graph at two `entries` buckets and assert the outputs differ
  appropriately — probe F shows a wrong-shape replay does not raise.
- **Tier behaviour unchanged:** identical hit/miss counts, `oom_skips` and bytes read versus the eager arm
  (`:510-513`), because the fetch is untouched.
- **Memory:** peak HBM growth ≤ +1 GiB (graph pools + 406 MB landing zone), per the engine's own budget
  (`runtime.py:70`).


## 10. What would falsify this

1. **A real layer does not capture** — the probes cover `npu_quant_matmul`, allocations, pinned H2D and
   shapes, not `F.linear` at 5120x2304, `out_projection`'s grouped einsum (`:861`), or `sparse_attn` at
   `[1,16,512]`. Capture layer 0 alone as a spike before building 80.
2. **The 1.88 us/node is optimistic for Cube ops.** At 3-4 us/node Phase 1 lands at ~870-960 ms instead of
   831 — still ~1.8x. The direction survives; the pass line moves.
3. **`entries` bucketing needs more buckets**, so the indexer stays eager and loses 113 ms/step — still
   clears 1.6x, but Phase 2 becomes mandatory.
4. **The multi-card layout cannot host the graphs** — a cross-device copy cannot be captured (probe G), so
   with `--expert-devices npu:0..3` (74% hit, `ASCEND-DSV41.md:255-256`) `g_post` must be captured *on the
   expert card*, and replaying a device-1 graph from a process whose current device is 0 is untested here.
   That layout has the best hit rate; probe it before freezing the design. Single-card is the safe target.
5. **Stream/context exhaustion** — the vendor hits `Alloc sq cq fail` and mitigates by *reducing the number of
   captured graphs* (`acl_graph.py:231-243`); 80-120 graphs is far more than a typical vLLM capture set.
   Mitigation: capture lazily per layer, fall back to eager for any layer that fails, log it.

---

## 11. Ranked summary

| # | change | calls saved/step | expected | risk |
|---|---|---:|---|---|
| 0a | device-side state addressing (`state.pos` exists at `:570`) | +80 | prerequisite; removes silent-wrong-KV class | low |
| 0b | device mHC + batched mixes (`host_mixes=False`) | −160 host syncs, +4 500 in-graph nodes | prerequisite; ~8 ms | low |
| 0c | hoist RoPE angles out of the loop | −1 600 | ~2% | low |
| **1** | **80 per-layer graphs, fixed 203/406 MB landing zone** | **−32 162** | **831 ms, ~2.0x** | medium |
| 2 | landing-zone double buffer + one-layer-ahead prefetch | 0 | floor → ~700 ms, ~2.4x | medium |
| 3 | `super_kernel_optimize` on the captured graphs | n/a (device) | attacks the 164 ms | medium |
| a | fixed 6-expert shape | 0 — already true (`:958`) | precondition | none |
| b | hoist host work | −1 600 | 5%, folded into 0c | low |
| d | batch the two mHC mixes | −80 host, −4 500 nodes | folded into 0b | low |
| c | 40 dense layers as one fused sequence | — | **not possible**: 8.1 GB/token vs 9.4 GiB tier at 46% hit | dead end |

**Do 0a/0b/0c, then capture layer 0 as a spike, then all 80. The direction is the right one: it is the only
change on this list that removes the 46 us/call toll 32 242 times instead of shaving it.**

---

### Appendix — the probes, verbatim

    ssh blue-910b3 'docker exec -i maya-ascend-dev bash -lc "python3 - 2>&1"' <<'EOF'
    import torch, torch_npu; dev="npu:0"
    g=torch.npu.NPUGraph(); x=torch.ones(4,device=dev)
    with torch.npu.graph(g): y=(x+1.0)*2.0
    x.fill_(5.0); g.replay(); torch.npu.synchronize(); print(y.cpu().tolist())   # [12,12,12,12]
    EOF

The 25x figure is the same construction at 200 nodes on a 512-element tensor, eager (one
`torch.npu.synchronize()` per iteration) versus one replayed graph, 3 repetitions, minimum reported. The card
is shared: the ratio is the claim, not the microseconds.
