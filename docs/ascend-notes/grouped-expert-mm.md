# The expert block as two grouped Cube calls

Read/write investigation on `maya-ascend-dev` (CANN 9.2.1, torch_npu 2.10.0.post7, Triton-Ascend 3.6.0,
Ascend910B3), 2026-11-24, **physical npu:2** (shared with a serving process holding ~18 GB).  Nothing here
is committed.  `DSV41_PER_EXPERT_MM=1` puts the per-expert form back for a paired A/B.

## What the 19.8 us a call actually is

`ascend/probe_mm_shapes.py` measures the op inside a captured graph (where a launch costs 1.88 us, not the
eager 47), at constant bytes and at constant call count:

| case | median ms | bytes |
|---|---|---|
| 18 x `[1,5120]x[5120,2304]` | 0.544 | 212 MB |
| 1 x `[1,5120]x[5120,41472]` | 0.289 | 212 MB |
| 6 x down `[1,2304]x[2304,5120]` | 0.207 | 70.8 MB |
| 1 x down, `M=6`, `N=6*5120` (diagonal) | 0.117 | 70.8 MB |
| 1 x `N=64` | 0.094 | 0.3 MB |

The same bytes in one call cost 47 % less, and a call that reads nothing still costs ~94 us of
replay+sync floor, so ~12 us of every call is fixed (tiling, workspace, launch, drain) and the weight read
runs at ~1.1 TB/s.  Six experts * eighteen calls a layer was paying that fixed cost eighteen times.

## The two calls

`_run_experts`, `batched_experts` (default on):

* **gate + up, one call.** At `T == 1` all six experts multiply the same activation row, so the six `w1`
  are one contiguous `[6*inter, hidden]` weight and the six `w3` the next one: one
  `[1, hidden] x [hidden, 2*6*inter]` returns all twelve results side by side (`regs.scale1` is the
  matching `[2*6*inter]` scale vector).
* **down, one call.** The six rows differ per expert, so the group goes on `N`: six rows
  (`hq` `[6, inter]`) against the six `w2` side by side, `[6, 6*hidden]` out, and only the *diagonal*
  blocks are the answer (`torch.diagonal` is a view, and the new `down_sum` kernel takes its row stride
  and sums the six rows into `y` in slot order - the `_down_kernel` calls' order, so bit-identical).

The int8 Cube accumulates in int32 (exact, tiling-independent) and the per-channel scale is one fp32
multiply per output element, so the grouping cannot change a bit - and it does not: on the real
checkpoint, `graphs=True` + `graph_verify=True`, 6 greedy decode steps, logits **bit-identical** between
the two forms (80/80 graphs captured).

## The landing zone had to become projection-major

`[2, 3, K, per]` (gate, up, down) instead of `[2, K, 3, per]`.  An expert is then no longer contiguous, and
the fill cannot be one copy per expert.  `ascend/probe_slab_layout.py` measures the alternatives:

| fill | ms/layer | GB/s |
|---|---|---|
| merged, 6 x 35.4 MB (the old slab) | 0.445 | 477 |
| `copy_` per (expert, projection), 18 x 11.8 MB | 0.746 | 284 |
| strided destination, 6 calls of 3 chunks | 1.351 | 157 |
| `cat` of the 18 projections into the flat slab, 1 call | 0.640 | 332 |
| one permuted copy (only on a full miss) | 0.449 | 472 |

The `cat` ships.  Wall-clock it is 44 % worse than the merged fill, but that is *host* time: on the device
the profiler shows the fill unchanged (`TensorMove` 9.72 -> 5.96 ms/token, `ConcatD` 0.91 -> 4.87), i.e.
+0.2 ms/token - the same 212 MB a layer, one kernel instead of six.

## Measured, `bench_profile --graphs --msprof`, 1 prompt of 8 + 3 decode steps, per token

| | before (HEAD) | per-expert arm | batched |
|---|---|---|---|
| `QuantBatchMatmulV3` calls | 483.3 | 483.3 | **250.5** |
| `QuantBatchMatmulV3` ms | 9.58 | 9.58 | **6.94** (27.7 us/call) |
| `TensorMove` calls / ms | 182.1 / 9.72 | 116.6 / 5.94 | 116.6 / 5.97 |
| `ConcatD` calls / ms | 107.5 / 0.91 | 118.5 / 5.07 | 118.5 / 4.87 |
| all operators, calls | 5802 | 5748 | **5370** |
| all operators, ms | 51.3 | 51.6 | **48.8** |

The profiled run also contains the 8-token prompt (eager, 3156 Cube calls) and 40 capture warm-ups, which
is why the per-token Cube count is not 80/11: the decode step itself is **720 -> 80** `npu_quant_matmul`
calls (18 -> 2 a layer x 40 layers), 240 of the batched arm's 2756.

`bench_profile --steps 16` wall clock could not resolve the change on this card (1666 vs 1496 ms/token,
min 1069 vs 1329): the tier was starved to 254 slots by the other tenant's 18 GB, so a step is
host-streaming-bound and this machine drifts.  Device time is the measurement.

## What the vendor will not take

* a 3-D batched `npu_quant_matmul` with a per-expert scale: `Expected 1 dimension input, but got x2Scale_
  with sizes [4, 32]` - the batched form requires one 1-D scale, so the down projection's six different
  scales cannot ride the batch axis; the diagonal form is how the group was batched instead;
* `npu_quant_grouped_matmul_dequant`: `support for Ascend910B is not implemented`;
* `npu_quant_matmul(group_sizes=...)`: `group_sizes only support input with three elements, but got 6`;
* `npu_grouped_matmul` (the list form) needs `group_type`, and the int8 path was not pursued further once
  the diagonal form was bit-identical and cheaper.

## What is left, and why it is not this

The fill is now the largest thing in the step again (TensorMove + ConcatD = 10.8 ms/token), but it is
8.5 GB read + 8.5 GB written of expert bytes that two Cube calls immediately read again: at 1.1 TB/s the
Cube alone is 7.7 ms/token for those bytes.  A cheaper copy is not available (the table above); what is
left is to *overlap* it - the fill is issued on the current stream between `g_pre` and `g_post`, and the
slab is already double-buffered by parity, so it could run on a second stream against the previous layer's
`g_post` Cube work.
