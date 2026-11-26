# 2 000-token serve measurement - one card (physical 4), four cards still faulted

Date 2026-11-25 18:31-20:35 Asia/Shanghai.  Host `blue-910b3`, 8 x 910B3 (64 GB each).
Container `maya-ascend-dev`, `ASCEND_RT_VISIBLE_DEVICES=4` -> logical `npu:0` is **physical card 4**.
**Every number below is one card.**  Four-card numbers do not exist (section 5).

Engine for every run: `serve_engine.Engine`, `--graphs --hbm-share 0.85 --reserve 26
--max-context 2048 --prefill-chunk 2048`; build 37-38 s, resident 15.09 GiB, tier **294 slots /
9.69 GiB**, KV 0.0146 GiB per conversation.  2 000-token prompts are 2 000-token slices of README.md
(distinct per request), 32 greedy tokens.  `--parallel N` gives N conversation slots; all requests are
submitted at once, so none queued in any arm (`queued_requests = 0`).

## 1. 8 concurrent x 2 000 tokens (single run, one card)

| quantity | value |
|---|---|
| **TTFT per request (s)** | rid1 **313.21**, rid2 302.62, rid3 302.08, rid4 301.44, rid5 301.39, rid6 301.30, rid7 267.68, rid8 267.79 |
| TTFT median / spread | **301.42 s** / **45.54 s** (min 267.68, max 313.21) |
| **aggregate decode** | **1.10 tok/s** (248 steps total, 31 per request) |
| **per-request decode** | 0.166, 0.161, 0.162, 0.161, 0.162, 0.162, 0.155, 0.156 tok/s (median step **6.36 s**) |
| **wall clock** | **501.27 s** (prefill phase 313.23 s) |
| **peak HBM** | **33.72 GiB** (card 4) |
| **tier hit rate** | **44.7%** overall (prefill 61.0%, decode 20.7%); 294 slots, 0 OOM skips |
| **GB read / request** | **339.93 GB** (142.89 GB in the prefill, 197.04 GB in decode) |
| **GB read / decode step** | **6.36 GB** |
| page-cache regime | **warm**: rchar 2 719.67 GB vs read_bytes 121.59 GB -> **95.5% from page cache** |
| graph lease | capture **failed**: `GraphRunnerError: capture 'g_post.0' failed ... npuSynchronizeDevice ... error code 107027`; engine logged "graphs: disabled after a capture/replay failure" and ran all 8 conversations **eager** |

Command and duration (11:52:17Z -> 12:01:27Z, 9 m 10 s incl. build):

    ASCEND_RT_VISIBLE_DEVICES=4 python3 -m ascend.bench_par \
      --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
      --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
      --cards npu:0 --parallel 8 --prompt-tokens 2000 --gen 32 \
      --reserve 26 --hbm-share 0.85 --arm 2k \
      --json ascend/results/serve-1card-par8-2k.json

Result: `ascend/results/serve-1card-par8-2k.json`.  Single run - no median of runs.

## 2. Scaling: concurrency 1, 4, 8 (one card, 2 000 tokens, 32 generated)

All four arms below use the same engine build shape (tier 294 slots) and the same prompts; only
`--parallel` changes.  N=1 was run twice: **cold** (first 2k after the box's earlier cold period) and
**warm**; the warm one is the one to compare.

| N | TTFT mean (s) | TTFT list (s) | spread | aggregate decode | per-request decode | wall (s) | peak HBM | hit rate | GB/request | GB/step |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 cold | **181.66** | [181.66] | - | 0.87 tok/s | 0.87 | 218.6 | 26.97 GiB | 13.6% | 555.65 | 5.95 |
| 1 warm | **77.24** | [77.24] | - | 1.12 tok/s | 1.12 | 105.9 | 26.97 GiB | 13.6% | 555.65 | 5.95 |
| 4 warm | **113.93** | [113.98, 113.95, 113.94, 113.86] | 0.12 s | 1.30 tok/s | 0.33, 0.33, 0.32, 0.33 | 212.5 | 30.03 GiB | 43.3% (prefill 64.9%) | 361.50 | 7.52 |
| 8 warm | **294.69** | [313.21, 302.62, 302.08, 301.44, 301.39, 301.30, 267.68, 267.79] | 45.54 s | 1.10 tok/s | 0.15-0.17 | 501.3 | 33.72 GiB | 44.7% (prefill 61.0%) | 339.93 | 6.36 |

Reading the curve:
* **Aggregate decode saturates at ~1.1-1.3 tok/s on one card** (1.12 -> 1.30 -> 1.10).  It does not scale
  with concurrency: the expert fetch is serialised by the engine's one-at-a-time fetch lock, and graphs
  are unavailable once a second conversation is live (capture times out, see section 1).  Per-request
  decode falls 1.12 -> 0.33 -> 0.16 tok/s.
* **TTFT is not monotone in N** because the 2k prefill is dominated by expert reads and the arms were run
  in a warming page cache: N=1 cold 181.66 s vs N=1 warm 77.24 s.  Against the warm N=1, 4 concurrent
  prefills cost 1.5x (113.9 s) and 8 cost 3.8x (294.7 s) - the eight prefills do not overlap their expert
  fetches, so the cost is close to the sum of eight single prefills divided by the shared tier's hits
  (8 x 77 s = 616 s of fetch work against 313 s wall, made up by the 61% prefill hit rate).
* N=2 was skipped for time (the two ends and N=4 bound the shape); no value is extrapolated.

## 3. Short-prompt control at 8 concurrent (one card)

8 x 8-token prompts, 32 generated tokens, same engine shape:

| quantity | value |
|---|---|
| TTFT list (s) | 27.46, 27.53, 28.51, 28.73, 28.29, 29.26, 28.18, 27.84; median **28.24**, spread **1.79** |
| aggregate decode | **1.13 tok/s** |
| per-request decode | 0.18, 0.18, 0.18, 0.18, 0.18, 0.15, 0.15, 0.18 tok/s |
| wall | 202.76 s (prefill phase 29.26 s) |
| peak HBM | 27.63 GiB |
| hit rate | 24.8% (prefill 56.4%, decode 19.2%) |
| GB/request, GB/step | 183.33 GB, 7.04 GB/step |
| page cache | 98.8% (rchar 1 466.86 GB, read 17.17 GB) |

The control separates the two costs: at 8 concurrent the **2k prefill adds 294.7 - 28.2 = 266.5 s of
TTFT** and ~2.7 TB of expert reads, while the **decode rate is the same 1.1 tok/s** with or without the
long prefill.  The concurrency cost is therefore in the decode path (serialised expert fetch + no
graphs), not in the prefill alone.

Result: `ascend/results/serve-1card-par8-short.json`.

## 4. Memory: does 8 x 2k fit the 32 GiB share?

Measured peak at 8 concurrent 2k is **33.72 GiB on card 4**, so it was run at `--hbm-share 0.85`
(54.4 GiB cap).  At the engine's default 0.5 share the cap is 32 GiB, **below the measured peak**, so
8 x 2k would not fit there; the 0.5 run was not executed (time), so the 32-GiB failure is inferred from
the measured peak, not observed.  The same card ran 1 and 4 concurrent at peaks 26.97 and 30.03 GiB.
No OOM occurred in any arm reported here (`oom_skips = 0`, no ERR lines).

## 5. Four cards (physical 4-7): still unusable

Every four-card attempt on 2026-11-25 faulted or wedged, before and after the claimed reboot:

| run | outcome |
|---|---|
| 2k, 4 cards, reserve 13/hbm 0.5 | prefill read 100.3 GB then wedged (`dms_event_wait_interrupt`, AICore 0%, no I/O >10 min); killed |
| 256 tok, 4 cards, 13/0.5 | `ACL stream synchronize failed 507035`; **chipId 5**, `aivec error`, "timeout or trap error" in an expert `weight_scale` `.to(device)` |
| 256 tok, 4 cards, 22/0.85 | vector-core exception **chipId 5** at `0x12c1d5000000`/+0x574, `aclnnNonzero` |
| 8 tok, 4 cards, 13/0.5 | vector-core exception **chipId 7** at `0x124e00000000`/+0x4ac, `rtEventRecord` |
| 8 tok, 2 cards (4,5) | same trap **chipId 5**, `rtEventRecord` |
| 8 tok, 2 cards, tier disabled | same trap **chipId 5** |
| **8 x 2k, 4 cards, 26/0.85** (after the lead's "fixed" check) | wall 119.5 s then `aclnnNonzero` **chipId 5**, PCs +0x478/+0x574; peak 23.9/0.6/0.0/0.0 GiB; process killed |
| **8 tok, 3 cards (4,6,7)** | first forward read 24.8 GB then **wedged** (CPU frozen, AICore 0%) |
| 1 card (4) or 1 card (5), same engine | **works** (all numbers above; card 5: TTFT 8.0 s, 0.83 tok/s, 101.67 GB, peak 20.3 GiB) |

A small copy+peer-event probe between cards passes, and chips report health OK with 0 ECC errors, but the
engine's multi-card expert path does not: it faults on whichever non-compute card is reached second, and
excluding card 5 (3-card 4,6,7) wedges instead.  `npu-smi set -t reset -i 4..7 -c 0 -m 1` is refused
("User aborts reset server by inband").  Note `uptime` shows the host is still on the **11:17 boot** (no
reboot happened on this host), so the fault has not been cleared.  Logs:
`ascend/results/diag-20261125/` (`p8.log`, `d3c.log`, `card5.log`, plus the earlier set).

## 6. Read-ahead and page-cache regime

* `/mnt/weights` on `/dev/mapper/openeuler-root`; `read_ahead_kb = 128` on dm-0 via the lead's
  `/etc/udev/rules.d/60-maya-readahead.rules`.  128 does not survive a reboot; the udev rule does.
* Raw device: `dd bs=8M count=256 iflag=direct` = **2.4 GB/s**; buffered = 680 MB/s.
* The one-card arms read 5.5-2.7 TB of expert bytes at 5-6 GB/s once the page cache was warm.  Regime by
  arm: **N=1 cold 62% cached** (rchar 555.87 GB / read 209.42 GB), N=1 warm 99.6%, N=4 99.9%, N=8 95.5%,
  short-8 98.8%.  The first 2k after the cold period therefore cost 181.7 s of TTFT against 77.2 s warm -
  a 2.35x regime effect that must not be read as a concurrency effect.
* A 2 000-token prefill moves ~185-555 GB of expert bytes depending on the tier's hit rate (13.6% at
  N=1, ~44% at N=4/8), not a fixed amount: the tier (294 slots of 15 360 experts) is what decides it.

## 7. Tests

`ASCEND_RT_VISIBLE_DEVICES=4 python3 -m unittest discover -s ascend -p "test_*.py" -t .`
-> **Ran 225 tests in 64.98 s, OK (skipped=10)**.  No `aclrtMallocHost` failure (host RAM free).

## 8. Files

| file | content |
|---|---|
| `serve-1card-par1-2k.json` | 1 x 2k, cold page cache |
| `serve-1card-par1-2k-warm.json` | 1 x 2k, warm page cache |
| `serve-1card-par4-2k.json` | 4 x 2k |
| `serve-1card-par8-2k.json` | **8 x 2k (headline)** |
| `serve-1card-par8-short.json` | 8 x 8-token control |
| `serve-diag-1card.json`, `serve-diag-card5.json` | single-card validation on cards 4 and 5 |
| `diag-20261125/` | four-card failure logs and card snapshots |

`ascend/bench_par.py` (new, benchmark-only; no engine change) drives `serve_engine.Engine` and reads the
protocol's own `T <rid>` lines for each request's first token; it splits the expert-tier bytes into the
prefill part and the decode part so "GB per request" and "GB per decode step" are separate measurements.
