# 2 000-token serve on four 910B3 cards - 8 concurrent, and the Triton cross-card fix

Date 2026-11-26 23:21 - 2026-11-27 00:2x Asia/Shanghai.  Host `blue-910b3` (rebooted 23:04:57),
container `maya-ascend-dev`.  **Physical cards 0-3** (`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`), all four
health OK and no other process on 0-3 before every run (`npu-smi info` checked; cards 4-7 idle
throughout).  Benchmark engine: `rt.build(..., graphs=False)`, resident **15.09 GiB** on npu:0, tier
**2 391 slots** at `hbm_share 0.5` / reserve 13 GiB (**4 979 slots** at 0.85), build ~40 s.  These
numbers are **eager**: `bench_serve` builds its 2 000-token engine with `graphs=False`, so no graph was
captured in these arms.  The spread-tier capture path is proven separately by the parity suite (section 4).

## 0. The result in one line

**8 x 2 000 tokens does not fit the 30.5 GiB half-card share: all eight prefills complete, then six of
the eight decoders OOM.  At `hbm_share 0.85` (51.8 GiB) all eight run: TTFT 339.61 s (spread 0.05 s),
aggregate decode 1.62 tok/s, wall 497.98 s, peak 39.61 / 24.41 / 21.70 / 24.09 GiB on npu:0..3.**
The largest concurrency that ran inside the half-card share on four cards is **4** (section 3).

## 1. 8 x 2 000 tokens, four cards - the headline run (share 0.85, eager)

`ascend/results/serve-4card-par8-2k.json`; cards 0-3; one 2 000-token README prompt shared by all eight
conversations (so the token streams are an isolation check); 32 greedy tokens each.  Started 23:57:46;
wall 497.98 s, process 539.7 s including the 40 s build.

| quantity | value |
|---|---|
| **TTFT per request (s)** | 339.63, 339.63, 339.59, 339.63, 339.59, 339.59, 339.63, 339.58 |
| TTFT median / spread | **339.61 s** / **0.05 s** (min 339.58, max 339.63) |
| **aggregate decode** | **1.62 tok/s** (256 steps total, 32 per request) |
| **per-request decode** | 0.21, 0.21, 0.21, 0.21, 0.20, 0.20, 0.21, 0.21 tok/s (median step **4.65-5.13 s**) |
| **wall clock** | **497.98 s** (prefill/TTFT phase 339.61 s) |
| **peak HBM per card** | **39.61 / 24.41 / 21.70 / 24.09 GiB** (npu:0..3, torch `max_memory_allocated`) |
| **tier hit rate** | **36.85%** (53 594 hits / 91 862 misses; 2 788 live slots of 4 979; 0 OOM skips) |
| **GB read per request** | **406.37 GB** (3 250.94 GB total / 8) |
| **GB read per decode step** | **1.21 GB** (io split: ~2 941 GB in the 339.6 s prefill, ~309 GB in the 158 s decode / 256 steps) |
| page-cache regime | **hot: 99.91% from page cache** (`rchar` 3 251.18 GB vs `read_bytes` 3.03 GB) |
| succeeded | **8 / 8**, no errors; graphs **off / eager** |

This run **exceeds the shipped 0.5 share**: its compute-card peak was 39.61 GiB against a 30.48 GiB cap,
so it is reported as the 0.85 configuration, not as an operation inside the 32 GiB premise.

Request isolation: 6 of the 8 token streams are identical; 2 diverge at token 10.  That is the
direct-drive harness, not the Triton kernels: `bench_serve.concurrent_prefill_decode` calls
`model.forward` from 8 threads against one shared `ExpertStreamer`, whose pinned staging ring is not
reentrant (the `serve_engine` path serialises the fetch with a lock; this harness does not).  The
four-card single-conversation engine output (section 4) is coherent.

## 2. 8 x 2 000 tokens at the 0.5 share - precisely what fails

`ascend/results/serve-4card-par8-2k-share05.json`.  Same eight conversations and tier shape as section 1,
`hbm_share 0.5` (30.48 GiB cap):

* **all eight prefills completed** (first token at ~312.5 s) - the 2 000-token prefill itself fits;
* **six of the eight decoders then died** with `torch.OutOfMemoryError` on the expert cards, e.g.
  `Tried to allocate 68.00 MiB (NPU 1; 25.90 GiB allocated; 30.04 GiB free; 30.48 GiB allowed;
  30.47 GiB reserved)` in `ExpertStreamer._read_many`, and an 18.00 MiB allocation in `expert_mm_raw`;
  NPU 3 failed the same way;
* `succeeded: 2`, wall 479.6 s, peak 25.06 / 25.93 / 23.76 / 26.01 GiB, tier hit 31.7%, 3 475.1 GB read,
  page cache 99.69%.

So 8 x 2 000 does **not** fit the half-card share: resident 15.09 GiB plus the 8 decode working sets
exhaust the 30.48 GiB cap during decode (the allocator had reserved 30.47 of 30.48 GiB).  The 0.85 peak
of 39.61 GiB is 1.30x the half-card share, i.e. this is a memory-cap failure, not a tier-size or a
Triton-path failure (all these arms are eager).

## 3. Concurrency 4 at the 0.5 share (largest N that ran inside the share)

`ascend/results/serve-4card-par4-2k.json` - four conversations, the same 2k prompt, 32 tokens, share
0.5, reserve 13 GiB (measured through the same function with `n=4`; see the harness note in section 7).

| quantity | value |
|---|---|
| TTFT per request / median / spread | 318.38, 318.37, 318.37, 318.37 s / **318.37 s** / **0.00 s** |
| aggregate decode | **1.37 tok/s** |
| per-request decode | 0.34, 0.34, 0.34, 0.34 tok/s (median step 2.97-2.99 s) |
| wall | **411.9 s** |
| peak HBM per card | **21.65 / 20.71 / 17.97 / 20.57 GiB** |
| tier hit rate | **29.92%** (21 748 hits / 50 936 misses; 1 818 live slots) |
| GB read | **1 802.6 GB total = 450.65 GB/request**; **2.55 GB/decode step** |
| page cache | **83.97%** cached (`rchar` 1 803.69 GB vs `read_bytes` 287.65 GB) |
| succeeded / tokens identical | **4 / 4**, all four streams identical |

N=1 was started at 0.5 but had not finished its 2 000-token prefill within the report window; it was
stopped to free the cards, so no N=1 number is reported and none is extrapolated.  The short-prompt
8-way control was not run (time).

## 4. Why the four Triton kernels broke on more than one card, and the fix

**Finding.**  `run_dsv41 --expert-devices npu:0..3` with all kernels on died at `chipId:3`, operator
`_quant_row`, `MTE accesses an invalid GM address or the cross-device memory access times out`.  With
`ASCEND_LAUNCH_BLOCKING=1` the first fault surfaced in `_sinkhorn_kernel` instead, and the trace printed
`mixes=npu:0 scale=npu:0 base=npu:0 cur=3` at the warm call; a quant-only bisect reproduced
`x/q/s=npu:0, cur=3, _quant_row, chipId:3`.  The model's operands agree with each other - the fault is
the **launch card**.  `runtime.build` calls `resolve_device(d)` for every expert-device entry
(`ascend/dsv41/runtime.py:114-115`), and that calls `torch.npu.set_device(index)` (`runtime.py:64`),
leaving card 3 current.  Triton's Ascend launcher takes the launch card from
`torch.npu.current_device()` (`triton/backends/ascend/driver.py:get_current_device`), not from the
pointers.  A wrapper that launched without selecting the operands' card ran card 3's kernel against
card 0's pointers - an async MTE trap that names whichever operator is in flight (`_quant_row` in the
normal run, `_sinkhorn_kernel` under launch blocking).  On one card current == operand device, which
is why every single-card run and the whole single-card suite were clean.

**Fix (launch on the operand's device and assert it).**  New `ascend/dsv41/triton_device.py`:

* `operand_device(op, **tensors)` returns the one device all operands are on, or raises
  `AssertionError: <op>: operands on different cards: x=npu:0, scale=npu:3.  A Triton kernel
  dereferences raw pointers and cannot span cards; place the operands on one card before the call`;
* `select(device)` makes that card current for the launch and restores the caller's card afterwards
  (a no-op when they already agree).

All four wrappers now allocate their outputs on the operands' device, check every operand they will
pass, and launch inside `select`: `quant_triton.py:167-173`, `sinkhorn_triton.py:152-157`,
`expert_triton.py:207-212` (`gate_up`), `expert_triton.py:231-233` (`down_epilogue`),
`expert_triton.py:251-253` (`down_sum`), `attn_triton.py:152-155`.  The level is the wrapper, not the
caller, so every caller (model, capture path, tests, probes) is device-correct; a caller that passes
co-located operands is unchanged (the device selection is a no-op and the outputs were already
allocated on the input device), and a caller that passes mixed operands gets the named assertion
instead of an MTE trap.

**Four-card run with the fix.**  `run_dsv41 --expert-devices npu:0,npu:1,npu:2,npu:3 --chat "The capital
of France is" --gen 8`, all kernels on, tier **2 603 slots over 4 cards**, hbm-share 0.5:

```
prefill: 9 tokens in 20.18s
gen: 1: 'The'  2803 ms ... gen: 8: '**.'  3087 ms
--- text ---
The capital of France is **Paris**.
experts: 1371 hits, 1725 misses, 61.0 GB read, 1480 resident
EXIT=0
```

No trap, 8/8 tokens, 1.8-3.1 s/token eager (log `ascend/results/serve-4card-triton-fixed.log`).

**Numerics are unchanged** (four visible cards, npu:0):

* `probe_quant`: **exact on 19/19 cases** - 0 int8 differences and 0 fp32 scale-bit differences,
  including 1 315 840 values at `[257, 5120]`, the all-zero row, and 40 958 exact-`.5` ties;
* `probe_expert`: rows=1 and rows=5, all four magnitude cases: `hq` differs **0**, `hs` differs **0**,
  down-epilogue `y` differs **0** - the expert block is **bit-identical** to the torch composition;
* `probe_attn_ab`: **80 real attention calls**, per-call `|kernel-torch|/max|torch|` **max 2.422e-6**,
  median 1.177e-6 - unchanged.

**Tests.**  `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 python3 -m unittest discover -s ascend -p "test_*.py" -t .`
-> **Ran 225 tests in 65.84 s, OK (skipped=8)**.  (One card was 225 / 10 skipped; the two extra tests
here are the real two-card ones, including `test_the_captured_decode_runs_on_a_spread_expert_tier`,
which now captures `g_post` on card 1 and passes.)

## 5. Page-cache regime and method

The arms ran back to back, so the cache warmed across them - this is a **warm-cache** study, not a
cold-boot one:

| arm | `rchar` | `read_bytes` | cached |
|---|---|---|---|
| 4 x 2k, share 0.5 | 1 803.69 GB | 287.65 GB | 83.97% |
| 8 x 2k, share 0.5 (OOM) | 3 475.59 GB | 10.96 GB | 99.69% |
| **8 x 2k, share 0.85** | **3 251.18 GB** | **3.03 GB** | **99.91% (hot)** |

`read_bytes`/`rchar` come from `/proc/<pid>/io` sampled every 2 s for the benchmark process (host side;
the process is inside `maya-ascend-dev`).  GB read per request and per step come from the engine's own
expert-byte counter (`tier.gb_read`), split at the measured TTFT instant using the same io series; the
io and tier counters agree to 0.03% on the headline run (3 251.18 vs 3 250.94 GB).  The full split is in
`ascend/results/serve-4card-io.json`.  The headline 8 x 2k is therefore a **hot-page-cache** number; a
cold run would move far more from disk and is not reported here.

## 6. Files

| file | content |
|---|---|
| `serve-4card-par8-2k.json` / `.jsonl` | **8 x 2k, share 0.85 (headline)** |
| `serve-4card-par8-2k-share05.json` / `.jsonl` | 8 x 2k at 0.5: all prefills, 6/8 decoders OOM (the exact failure) |
| `serve-4card-par4-2k.json` / `.jsonl` | 4 x 2k at 0.5: 4/4, 318.37 s TTFT, 1.37 tok/s |
| `serve-4card-io.json` | expert bytes and page-cache fractions, phase split per arm |
| `serve-4card-triton-fixed.log` | the four-card `run_dsv41` all-kernels-on run with the fix (`**Paris**`) |
| `serve-2k8-REPORT.md` | this report |
| `serve-2k8-REPORT-1card.md` | the previous one-card report, kept for history |

## 7. Harness note (why `bench_4card_par.py` exists)

`bench_serve.py`'s default `concurrent`/`control` scenarios call `concurrent_prefill_decode(...)`
without `n`, so the function's default `n=4` is used and `--parallel` only affects the serve-engine
arms: `--only concurrent --parallel 8` really ran four conversations (the JSON said `"requests": 4`).
`ascend/bench_4card_par.py` (benchmark-only, new) builds the same eager engine and calls the **same**
`bench_serve.concurrent_prefill_decode` with `n` set, adding only the io counters; no measurement
code is changed.  The one-line fix in `bench_serve.py` would be to pass `n=a.parallel` at those two
call sites.
