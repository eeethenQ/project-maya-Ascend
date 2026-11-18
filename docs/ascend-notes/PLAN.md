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

| # | item | owner | state | acceptance |
|---|---|---|---|---|
| 0c | the step's RoPE tables, built once per layer | lead | **done** (`6966609`) | Cos/Sin counts halved in `op_summary`; decode-equals-chunk test green |
| 0a | device-side state addressing (the KV planes) | `kv-device` | in flight | a drop-in plane module with CPU tests proving it equals today's slicing, including early positions, wrap, `-1` padding, and no host ints |
| 0b | mHC on the device path, the two mixes batched | lead | queued | one `[2, 4d] x [4d, 24]` matmul replacing two; equal to the current result to 2e-5 |
| P1 | capture one layer, verify the replay | `capture-harness` → lead ran the probe | **done** (evidence below) | a probe that captures this engine's own ops (`npu_quant_matmul`, `index_copy_`, `index_select`, softmax, topk, two-graph handoff) and gets **bit-identical** results |
| P2 | 80 graphs + one fixed expert landing zone | lead | queued | interleaved A/B, 48 paired samples, median; greedy token ids identical; target 1 661 -> ~831 ms |
| R4 | preload all 15 360 experts' scales and offsets (0.61 GB) | lead | queued | the 120 synchronous scale copies per step go away; no numerics change |
| D1 | device-time cuts a graph cannot remove | unassigned | queued | `npu_fusion_attention` with the sink (~35-45 ms of device time), `npu_dynamic_quant`, `npu_clipped_swiglu` |

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

## Measured facts the queue is built on

- 32 242 operator calls a step, 164 ms of device time, ~1 661 ms of wall clock: **call-bound**.
- the expert reads are **cold** (`read_bytes` 1.65 GB/step against `rchar` 1.47) and the array saturates at
  **~2.5 GB/s** - threads do not help, which is why R1 is dead and the stream is not on this queue.
- `torch.npu.graph` captures and replays on this stack; 47.4 us a call eager becomes 1.88 us a call
  captured. That ratio is why the call-cutting work (8 832 calls, ~177 ms) is queued *after* the graphs: it
  would be worth ~17 ms once a graph is in place.

## Ruled out, with the measurement that ruled it out

A hit-count expert profile (-18-31 points of hit rate against LRU at 2 603 slots); speculative prefetch from
the previous token's routing (3.24x read amplification, net slower); a pinned host arena (`ulimit -l` is
64 MB, and pageable already copies at 20 GB/s); pooling the four cards' HBM over HCCS (0 hit-rate points -
the per-layer capacity is what binds); the Engram prefetch (0.90x); the per-layer tier quota (neutral).
