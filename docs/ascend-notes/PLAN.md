# The optimisation work queue

One page for the work the lead agent runs, so progress survives a long session and a second pair of eyes
can see what is in flight, who owns it and what "done" means. The analysis behind the ordering is
[`../ASCEND-OPTIMISATION.md`](../ASCEND-OPTIMISATION.md); this file is only the state.

## Rules the work follows

- **One owner per file.** A specialist may create new modules under `ascend/dsv41/` and their tests; the
  lead is the only one who edits `model.py` and the other existing files, so two agents cannot clobber each
  other. Integration is the lead's job.
- **Only the lead touches the cards.** The four 910B3s are shared with another tenant; a measurement is
  worthless if two of us are on them. Specialists do CPU-only work (and the CPU torch path is available in
  the container), and hand over a script when something must run on the NPU.
- **Every step is a commit of its own** and leaves the tree green (`python3 -m unittest discover -s ascend`).
  A step that cannot be finished is reverted, not left half-applied - the two neutral experiments so far
  (Engram prefetch, per-layer tier quota) were both removed rather than kept behind a flag.
- **A change is only "faster" if an interleaved A/B in one process says so**, because this machine drifts
  by up to 50% between runs. Operator counts and hit rates are the deterministic secondary signal.

## The queue

Done, with the evidence that closed it:

| # | item | evidence |
|---|---|---|
| 0c | the step's RoPE tables, built once per layer | `6966609`: Cos/Sin counts halved in `op_summary`; decode-equals-chunk green |
| 0a | device-side state addressing (the KV planes) | `a61bf27`: `KvPlanes` is a drop-in (18 CPU tests over positions 0..300, wrap, rewind, `-1` padding); one `index_copy_` write, a fixed-width window gather plus mask; the real-model probe still says `" dog"` |
| 0b | the two mHC mixes batched | ruled out: they take different inputs (`h` before and after attention), so one `[2, 4d] x [4d, 24]` matmul cannot serve both; what 0b needed is `host_mixes=False`, already the general path |
| P1 | the capture probe | `67f39fa`: every operator in the forward captures and replays bit-identically; a two-graph `g_pre -> D2H -> g_post` reproduces eager |
| P2a | every layer kind captures | `6dd8061`, `dafe6cb`: a window layer 3 351 -> 606 us (**5.53x for that layer**), a compress layer 8 687 -> 1 100 us (**7.90x for that layer**); all 40 layers capturable. Per-layer, not the step |
| P2b | 80 graphs, single card | `615104c`: 595 against 998 ms/token, paired **0.660 [0.612, 0.738]**; 16 greedy ids identical and logits bit-equal; capture 1.9 s |
| P3 | four-card graphs, a landing zone per expert card | `383ab69`: 324.5 against 753.1 ms, paired **0.481 [0.261, 0.652] = 2.08x**; tier 2 480 slots at **90.9%** and 0.774 GB/step against single-card 254 at 64.6% and 3.004; 80 graphs in 1.8 s; the silent per-card capture-stream bug fixed |
| P4 | the landing-zone copy work | `aca9485`: 452.4 -> 409.6 ms, paired **0.880 [0.773, 0.979]**, -42.8 ms; TensorMove 1 016.5 -> 338.8 calls and 22.46 -> 19.65 ms a step |
| P5 | the GSM8K re-check | `c3a6ac6`: 9/10 at 1.77 tok/s with graphs against 10/10 at 0.83 eager; a control with graphs off and `host_mixes=False` reproduces the graphs-on text token for token - the difference is the mHC branch, not capture |
| P6 | Engram fixed, graphs with Engram on | `cc01d7d`: the contract is restored in `engram.py`; 80 graphs with Engram on, greedy ids identical, logits bit-equal, `" dog"` p≈1.000; 160 tests, 8 skipped |
| R4 | an expert's scales read once and kept on the card | `8d3d9a4`: 1 158 ms cached against 1 125 fresh, **0.97x** - no wall-clock gain. It stays only as a capture enabler, because it removes the synchronous scale copies |
| D1 | device-time cuts a graph cannot remove | `8cfdc55`: measured out, see "Ruled out" below; net effect on the step none measurable (1.011 [0.974, 1.266] over 28 paired samples) |

What is genuinely left:

| # | item | why it is left / acceptance |
|---|---|---|
| G1 | expose `--graphs` in `ascend/serve_engine.py` | `strata-ascend` builds the eager model today, so the server runs 816 ms, not 409.6; `run_dsv41 --graphs` is the only captured path |
| G2 | the recapture cost | 80 graphs take 1.8-1.9 s and belong to a State; every new conversation recaptures, and crossing an indexer bucket costs another 1.8 s. The GSM8K run's last State captured 118 at 2 048 context |
| G3 | make the tier the landing zone | at 90.9% hit the step still copies 8.1 GB device-to-device; `TensorMove` is 19.65 ms, the top device-time item |
| G4 | prefill | `T > 1` is eager and array-bound: 421 s for 823 prompt tokens in the GSM8K re-check; capture does not apply to it |
| G5 | multi-session | the serve engine holds one conversation (`READY ... multi=n` is absent) |

The graph work landed in the six commits from `615104c` to `cc01d7d`: single card 0.660, four cards 0.481,
the copy work 0.880, plus `8cfdc55`'s graph leak and device-time negatives. The re-check and the Engram fix
were not speed steps; they changed what can be claimed about accuracy and correctness.

## The capture probe: every op this engine needs captures, bit-identically

`.ascend-work/probe_capture.py` on `npu:0`, one preallocated input set per case, eager against a replayed
graph (20 iterations):

| case | mode | checksums equal | eager us | graph us |
|---|---|---:|---:|---:|
| `npu_quant_matmul` (the expert multiply) | graph | **yes** | 83.2 | 29.4 |
| `index_copy_` (the KV write) | graph | **yes** | 33.7 | 35.4 |
| `index_select` (the fixed-width window gather) | graph | **yes** | 28.2 | 26.3 |
| `linear` bf16 | graph | **yes** | 34.4 | 25.7 |
| `softmax` | graph | **yes** | 29.1 | 22.6 |
| `silu` / `clamp` | graph | **yes** | 23.6 / 27.9 | 23.1 / 23.6 |
| 3-operand `einsum` | graph | **yes** | 150.3 | 28.5 |
| `topk` | graph | **yes** | 38.0 | 31.5 |
| `g_pre` -> D2H -> `g_post` (two graphs, the round trip between replays) | graph | **yes** | 184.1 | 51.1 |

The per-case speedups are small because each case is one latency-bound call on a tiny tensor; what this
establishes is the thing that could have sunk the whole direction - **no operator in this engine's path
fails to capture, an in-place write captures, and a two-graph pipeline with a host round trip between the
replays reproduces the eager result exactly**. `GraphRunner` (`ascend/dsv41/graph_runner.py`) wraps this with
the input contract (shape, dtype, device, `data_ptr` checked on every replay) and a guard that proves the
graph reads live tensors rather than a baked value.

## Capture of a real layer now works - and the blocker was one host copy

The bisect (`.ascend-work/bisect_capture.py`) named it on the first run: **`Rope`**. Its frequency tables
were built on the **CPU**, so every `Rope.angles` call did a host->device copy of them - 136 a decode step -
and a copy is precisely what a captured graph cannot contain. `read_window`, `sparse_attn` and
`out_projection` captured fine all along; the tables are the only thing that touched the host. They are now
created on the device, which also removes those 136 copies from the eager path.

With that one change, `.ascend-work/probe_layer.py` on layer 1 (window-only):

| | |
|---|---|
| capture | 56 ms, mode `graph` |
| replay at the capture position vs eager | **bit-identical** (max diff 0.0) |
| replay after moving the position tensor in place vs a *fresh eager* run at the new position | **rel 0.0** |
| and the two eager runs do differ (0.543), so the position is not vacuous | |
| one layer, eager vs graph | **3 351 us -> 606 us (5.53x)** |

So the two properties that could have sunk the direction both hold: the graph replays bit-exactly, and it
reads the live position rather than a value baked at capture. Phase 1 is unblocked; what remained for the
per-layer split was the compressor/indexer (layers with `compress_ratio > 0` still derived group indices from
host ints) and the MoE, which must stay outside the graph because it reads the host. Both landed: the
compressor/indexer in `dafe6cb` (one graph per selection bucket), the MoE seam in `615104c`.

## Where the capture stood before this

The individual operators all capture and replay bit-identically (`probe_capture.py`, table above), and 0a
removed the host integers and varying shapes from the layer's KV path. But **capturing the composed layer
still fails**: `.ascend-work/probe_layer.py` captures layer 1 (window-only, so no compressor or indexer)
and `GraphRunner.capture` raises

    capture 'layer1' failed on npu:0: RuntimeError: ACL stream synchronize failed, error code:107027

so something inside `Dsv41.attention` is still not capturable even though each of its operators is - an op
that synchronises internally, or an allocation/in-place pattern ACL dislikes, not a host integer (those were
the easy cases and are gone). The next diagnostic is a bisect, capturing progressively more of the layer in
this order: `Rope.cos_sin` + `apply`; the q path (`wq_a`, `rms`, `wq_b`); the kv path and `write_window`; the
window `read_window`; `sparse_attn` alone; `out_projection`; then the whole `attention`. Whichever step first
raises names the operator to work around. It was run and named **`Rope`**: its tables were built on the CPU,
so every `Rope.angles` call did a host copy. `6dd8061` moved them to the device, and the composed layer then
captured.

## Two bugs the work found

- `run_dsv41`'s `--expert-slots` defaulted to **0**, which meant "no tier" rather than "auto-size".  Every
  `run_dsv41` run since that rewrite was measuring a token that re-read all 240 of its experts - 2.5 s
  against 1.6.  The `bench_step` and GSM8K numbers are unaffected (they size the tier from the HBM left).
- `State` kept two cursors of its own and `KvPlanes` a third; `forward` had to advance all of them, and the
  rewind test is what caught the drift.

## Measured facts the queue is built on

- the captured step is **409.6 ms/token** on four cards with 80 graphs: 324.5 against 753.1 (0.481, 2.08x),
  and 816.5 ms eager. The copy work was 452.4 -> 409.6 (0.880). The absolutes drift; the ratios do not.
- the four-card tier is **2 480 slots, 90.9% hit, 0.774 GB/step**; one card 254 slots, 64.6%, 3.004 GB/step.
- the device time is **137.5 ms** at 29 744 operator calls after the graphs (pre-graph: 164.3 ms at
  32 242); the top device-time item is the landing-zone `TensorMove`, 19.65 ms after the copy work.
- R4's scale cache is worth **0.97x**, i.e. nothing, on wall clock: the analysis's "5-15 ms" did not appear.
- the expert reads are **cold** (`read_bytes` 1.65 GB/step against `rchar` 1.47) and the array saturates at
  **~2.5 GB/s** - threads do not help, which is why R1 is dead and the stream is not on this queue.
- `torch.npu.graph` captures and replays on this stack; 47.4 us a call eager becomes 1.88 us a call
  captured. That ratio is why the call-cutting work (8 832 calls, ~177 ms) was queued *after* the graphs,
  not before: it would be worth ~17 ms once a graph is in place. It was not done.

## Ruled out, with the measurement that ruled it out

A hit-count expert profile (-18-31 points of hit rate against LRU at 2 603 slots); speculative prefetch from
the previous token's routing (3.24x read amplification, net slower); a pinned host arena (`ulimit -l` is
64 MB, and pageable already copies at 20 GB/s); pooling the four cards' HBM over HCCS (0 hit-rate points -
the per-layer capacity is what binds); the Engram prefetch (0.90x); the per-layer tier quota (neutral, 1.07x);
R1 and R3 (the array saturates at ~2.5 GB/s and `hc_post` leaves nothing to overlap); `npu_dynamic_quant`
(-2 642 calls and ~3 ms of device time, but logits move 3.1e-2, outside the 2e-3 promise); `npu_clipped_swiglu`
(+1.9 ms net); and `npu_fusion_attention` (cannot be captured - `AclmdlRICaptureEnd` 107025).
