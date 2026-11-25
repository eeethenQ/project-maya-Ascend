# 2 000-token serve measurement - one card (physical 4), four-card blocked

Date 2026-11-25 18:31-19:20 Asia/Shanghai.  Host `blue-910b3`, 8 x 910B3 (64 GB each).
Container `maya-ascend-dev` (was `Exited (255)` after the 11:17 reboot; started for this run).
`ASCEND_RT_VISIBLE_DEVICES=4` -> logical `npu:0` is **physical card 4**.  All numbers below are
**one card**; none are extrapolated to four.

## Verdict

* **One card runs and was measured**: 2 000-token prompt, sequential (N=1), 32 greedy tokens -
  **TTFT 181.66 s**, 0.87 tok/s aggregate, 13.6% tier hit, 555.65 GB read, peak 26.97 GiB.
* **Four cards (4-7) cannot run on this host right now.**  Every multi-card configuration faults in a
  CANN kernel on the second and later cards (evidence below); the box owner has authorised a reboot.
* The 8-concurrent 2k sweep and the short-prompt controls are **not measured yet**.

## 1. One card, 2 000-token prompt, sequential (single run)

`serve_engine.Engine`, `--cards npu:0 --parallel 1 --graphs --hbm-share 0.85 --reserve 26
--max-context 2048 --prefill-chunk 2048`; one prompt of 2 000 tokens from README.md, 32 greedy tokens.

| quantity | value |
|---|---|
| engine build | 36.9 s; resident 15.09 GiB; tier **294 slots / 9.69 GiB**; KV 0.0146 GiB/conversation |
| **TTFT** | **181.66 s** (= the whole prefill; one 2 000-token chunk) |
| decode, aggregate | **0.87 tok/s** (31 steps in 36.9 s) |
| decode, per step | median **1.077 s**, min 0.865, max 2.913 (the max is the first gap: 80 graphs captured in 2.86 s) |
| wall clock | **218.6 s** (incl. 32-token decode) |
| **peak HBM** | **26.97 GiB** (allocated 25.8 GiB) |
| **tier hit rate** | **13.6%** over the arm (2 470 hits / 15 701 misses); prefill 0.0%, decode **32.2%** |
| **GB read / request** | **555.65 GB** (371.27 GB in the prefill + 184.38 GB in decode) |
| **GB read / decode step** | **5.95 GB** |
| graph lease | 80 graphs captured 181.7-184.6 s, no revoke, no re-arm; 31 replayed steps |
| queueing | 0 requests queued |
| page-cache regime | rchar 555.87 GB vs read_bytes 209.42 GB -> **62% of the bytes came from host page cache**; prefill 201.5 GB read from the device of 371.5 GB requested (46%), decode only 7.89 GB of 184.4 GB (96% cached). This is a partly-warm regime, not the 77-103 s fully-warm pair. |

Command (inside `maya-ascend-dev`, from `/workspace/project-maya-Ascend`):

    ASCEND_RT_VISIBLE_DEVICES=4 python3 -m ascend.bench_par \
      --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
      --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
      --cards npu:0 --parallel 1 --prompt-tokens 2000 --gen 32 \
      --reserve 26 --hbm-share 0.85 --arm 2k \
      --json ascend/results/serve-1card-par1-2k.json        # 11:11:08Z -> 11:15:34Z, 4 m 26 s

Result: `ascend/results/serve-1card-par1-2k.json`.

## 2. One card, 8-token prompt (harness validation, single run)

Same engine shape with `--reserve 10 --hbm-share 0.5` (tier 132 slots / 4.35 GiB), 8 tokens, 8 greedy
tokens: TTFT **12.45 s**, wall 29.86 s, **0.46 tok/s**, tier hit **0.0%**, **101.67 GB** read
(**9.71 GB/step**), peak **20.3 GiB**, 80 graphs in 0.90 s.  Result:
`ascend/results/serve-diag-1card.json`.  This is the run that proves the driver/collector path end to end.

## 3. Why there are no four-card numbers

All four-card runs had cards 4-7 idle (3.4 GB baseline, AICore 0%, no processes) and no other benchmark
process running.  `serve_engine.Engine`, `--parallel 1`, one 2 000-token chunk unless stated:

| run | outcome |
|---|---|
| 2 000 tok, 4 cards, reserve 13/hbm 0.5 | prefill read **100.3 GB** then wedged: `dms_event_wait_interrupt`, NPU AICore 0%, no I/O, no progress >10 min; killed. |
| 256 tok, 4 cards, reserve 13/hbm 0.5 | `RuntimeError: ACL stream synchronize failed, error code:507035`; **chipId 5**, `aivec error`, "timeout or trap error", in the `.to(device)` of an expert `weight_scale` in `_read_many`. |
| 256 tok, 4 cards, reserve 22/hbm 0.85 | same vector-core exception **chipId 5**, same PC (`0x12c1d5000000`/+0x574), in `aclnnNonzero`. |
| 8 tok, 4 cards, reserve 13/hbm 0.5 | vector-core exception **chipId 7**, `0x124e00000000`/+0x4ac, `rtEventRecord`; "MTE accesses an invalid GM address or the cross-device memory access times out". |
| 8 tok, 2 cards (4,5), reserve 10/hbm 0.5 | same trap **chipId 5**, `rtEventRecord`. |
| 8 tok, 2 cards (4,5), reserve 30/hbm 0.5, **no graphs**, tier effectively disabled (pure streaming) | same trap **chipId 5**, `rtEventRecord`. |
| 64 tok, 4 cards, plain eager runtime (`bench_serve --only single`, no serve path) | process abort: **chipId 7**, `_quant_row`, `MTE accesses an invalid GM address`, core dumped. |

Two independent code paths, graphs on and off, tier large and off - all trap on the second+ card in a
cross-card operation.  One card has no cross-card traffic and works.  Chips report health OK, ECC
single/double = 0 on 4-7, per-chip 1024x1024 bf16 matmul and 2048x2048 host->device copy pass;
`npu-smi set -t reset -i 4..7 -c 0 -m 1` is refused ("User aborts reset server by inband").
Container image (2026-10-29), container (2026-11-17) and CANN 9.2.1 are unchanged since the 2026-11-21
results; the host rebooted at 11:17 today, so the regression is host/driver state, not the repository.

Raw logs: `ascend/results/diag-20261125/` (`par1.log`, `tiny.log`, `diag256.log`, `diag256hi.log`,
`diag8.log`, `diag2c.log`, `diag2nt.log`, `short8.log`, `diag1c.log`, `cards-before-par1.txt`).

## 4. Board state

* Cards 0-3: tenant `wyd_opd_fresh_1125c_loop` (`/dev/davinci0-3`, `ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`),
  four `VLLMWorker_TP` at 13-15 GB HBM each, 86-96% AICore, 200-sample collect loop that restarts vLLM
  between rounds.  Not free.
* Cards 4-7: free (3.4 GB driver baseline) but faulty under multi-card load, as above.

## 5. Storage / read-ahead

* `/mnt/weights` is on `/dev/mapper/openeuler-root`; `read_ahead_kb = 128` on dm-0, set by the lead's
  `/etc/udev/rules.d/60-maya-readahead.rules`.  128 does not survive a reboot; the udev rule does.
* `dd bs=8M count=256 iflag=direct` = **2.4 GB/s**; the same read buffered = **680 MB/s**.
* The single-card prefill moved 371.5 GB of expert bytes in 181.7 s = **2.04 GB/s** requested, of which
  201.5 GB came from the device (1.1 GB/s disk) and the rest from the page cache.  A 2 000-token prefill
  is therefore **3-5 min on one card** in this regime, not 13-17 min (that earlier estimate came from the
  faulty four-card run, whose reads ran at 450-630 MB/s while it wedged).
* Page cache is **partly warm**: 62% of the 2k arm's bytes, 96% of the decode bytes, 54% of prefill bytes.

## 6. Tests

`ASCEND_RT_VISIBLE_DEVICES=4 python3 -m unittest discover -s ascend -p "test_*.py" -t .`
-> **Ran 225 tests in 64.98 s, OK (skipped=10)**.  No `aclrtMallocHost` failure (host RAM was free:
~1.4-1.8 TB available).

## 7. Not measured (next, after the reboot if the cards come back)

* 8 x 2 000-token: TTFT per request and distribution, aggregate/per-request tok/s, wall, peak HBM, hit
  rate, GB/request, GB/step.
* Scaling 1/2/4/8 at 2k on one card (the driver supports `--parallel N`; N=1 is done, N=8 is next).
* Short-prompt control at 8 concurrent.
* Four-card numbers at all - blocked by the multi-card fault.

`ascend/bench_par.py` (new, benchmark-only, no engine change) is the driver used here: it runs
`serve_engine.Engine`, reads the protocol's own `T <rid>` lines for each request's first token, and splits
the expert-tier bytes into the prefill part and the decode part so "GB per request" and "GB per decode
step" are separate measurements.
