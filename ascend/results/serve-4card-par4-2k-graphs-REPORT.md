# Four 2 000-token conversations with the captured decode enabled - four cards, hbm-share 0.5

Date 2026-11-27 04:05-05:10 Asia/Shanghai. Host `blue-910b3`, container `maya-ascend-dev`, physical cards
0-3 (`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`).  All eight cards were free before every run (`npu-smi info`; the
smoke run and every arm ended with the cards released).  Engine: `ascend/bench_4card_graphs.py` (new,
benchmark-only, no engine edit), which builds with `rt.build(..., graphs=True|False)` and calls the same
`bench_serve.concurrent_prefill_decode` the 4 x 2k record used.  `graphs=True` sets `host_mixes=False`
(the branch a captured layer contains) and takes the build's extra 1 GiB/card graph reserve.

Commands (each run is one process):

    python3 -m ascend.bench_4card_graphs --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
      --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
      --cards npu:0,npu:1,npu:2,npu:3 --graphs on|off --parallel 4 \
      --prompt-tokens 2000 --gen 32 --max-context 2048 --hbm-share 0.5 --expert-reserve-gib 13 \
      --out ascend/results/serve-4card-par4-2k-graphs-{on,off}.json

    # the same driver at one conversation, and the domain-fit probe
    ... --graphs on|off --parallel 1 ... --out ascend/results/serve-4card-par1-2k-graphs-{on,off}.json
    ... --mode fit --fit-domains 4 --fit-prefill 2000 ... --out ascend/results/serve-4card-domain-fit-4-2k.json

## 0. The result in one line

**At four concurrent 2 000-token conversations the captured decode never runs.**  All four States build a
domain, the first simultaneous capture fails on the shared card stream (`Stream ... is being captured`),
`_graph_decode` disables graphs for the whole process, and 1-3 of the 4 requests die in their post-forward
device sync.  Graphs ON is strictly worse than OFF at N=4.  Four captured domains *do* fit 30.5 GiB
(1.585 GiB each, 0.396 GiB per card; four captured and replayed), but `Dsv41` holds one domain
(`model._decode_graphs`) and releases it for the next State, so the largest concurrency with a working
captured decode is **1**.  At N=1 capture works: 80 graphs in 1.70 s, **0.734 s/step vs 1.677 s eager
(2.28x)**, 1.28 vs 0.60 tok/s.

## 1. 4 x 2 000 tokens, 32 generated tokens, graphs on vs off (interleaved, three runs each)

| run (start) | ok | TTFT per request (s) | spread | decode tok/s | per-req tok/s | median step | wall | peak HBM npu:0..3 (GiB) | tier hit | GB/req | GB/step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ON #1 04:12 | **3/4** | 181.47, 181.49, 181.43 | 0.06 | 1.33 | 0.44 | 2.238 s | 253.6 | 20.57 / 20.97 / 18.07 / 20.50 | 29.78% | 403.86 | n/a |
| ON #2 04:29 | **3/4** | 199.80, 199.78, 199.80 | 0.03 | 1.29 | 0.43 | 2.323-2.333 s | 274.4 | 20.41 / 21.77 / 18.63 / 21.13 | 29.74% | 405.60 | n/a |
| ON #3 05:00 | **1/4** | 179.95 | - | 0.65 | 0.65 | 1.509 s | 229.1 | 20.28 / 21.04 / 18.04 / 20.47 | 9.40% | 403.74 | 3.44* |
| OFF #1 04:18 | 4/4 | 178.21, 178.15, 178.62, 178.68 | 0.53 | 1.41 | 0.35 | 2.797 s | 268.7 | 19.99 / 20.68 / 17.91 / 20.61 | 38.08% | 398.18 | 0.775 |
| OFF #2 04:24 | 4/4 | 189.78, 189.76, 189.74, 189.81 | 0.07 | 1.42 | 0.35 | 2.829 s | 280.2 | 20.47 / 20.71 / 17.93 / 20.64 | 38.08% | 398.18 | - |
| OFF #3 05:05 | 4/4 | 180.55, 180.52, 180.51, 180.50 | 0.05 | 1.43 | 0.36 | 2.802 s | 270.2 | 20.33 / 20.69 / 17.91 / 20.61 | 38.08% | 398.18 | 0.775 |

`*` ON #3's decode split is over 35 one-token steps (1 survivor x 32 + 3 aborted one-step conversations);
it is not a clean per-step figure.  The ON arm's higher *per-request* tok/s is an artefact of fewer live
conversations (3, then 1), not a graphs win - do not read it as one.  ON vs OFF cannot be paired on decode
because there is no captured decode at N=4 to compare; the paired comparison exists only at N=1 (section 3).

Honesty of the pair: ON/OFF/OFF/ON/ON/OFF, each a fresh process, same prompt, same 32 tokens, same tier
shape.  Identical OFF arms drifted 178.4 -> 189.8 -> 180.5 s TTFT (11.4 s) and 268.7 -> 280.2 -> 270.2 s
wall, so the table reports every run and the range, not one number.  Tier GB/request is bit-reproducible
(398.18 GB three times) and every token stream of every successful request (19 in all) is identical to the eager stream.  Today is much faster
than yesterday's record (318 s TTFT / 411.9 s wall for the same arm): the host page cache is fully warm
(see section 4), so yesterday's and today's absolute times are not cross-comparable.

## 2. Did capture happen per conversation?  (probe: every domain created/closed, every disable, every step)

ON #1: 3 domains created (`conv 1` never got one), 2 closed, 1 leaked, **0 graphs captured** (0 of 80 in
every domain), 2 `_disable_graphs`, 0 of 96 decode steps on its own domain, `model.graphs=False` at end.
ON #2: 4 domains created, 3 closed, 1 leaked, **1 graph captured** in total, 3 disables, 0 of 97 steps.
ON #3: 4 domains created, 3 closed, 1 leaked, **1 graph captured**, 3 disables, 0 of 32 steps.
The failure is always the same:

    GraphRunnerError: capture 'g_pre.0.0' failed on npu:0: npuSynchronizeDevice ... error code 107027
    Not_Supported(EE1016): Synchronizing the device failed. Reason: Stream (stream_id=45) on device_id=0
    is being captured. End the stream capture before synchronizing the device.

and the request-level failure is the harness's post-forward `_sync(devices)` hitting the same open capture
stream (`RuntimeError: ... stream is captured`), which kills that conversation's thread: 1 death in #1/#2,
3 in #3.  A leaked 1.585 GiB domain is left alive (unreferenced) after each ON run.

The mechanism is structural: `_graph_decode` (`model.py:2289`) tears a domain down whenever
`g.state is not state`, and `_graph_decode` builds a new `_DecodeGraphs` for the new State - so four
threads each want their own domain, they all construct one at the same instant, and two captures cannot
run on one card's stream.  `serve_engine`'s lease avoids the crash by revoking: with `--graphs --parallel 4`
a second admission sets `model.graphs=False` and releases before the first decode, so a 4-way burst
captures **zero** conversations as well (code path §19.2; the existing `serve-lease-4card` run is
grant-then-revoke).  Either way, N=4 is eager.

## 3. The largest concurrency with a working captured decode: 1 (four cards, 2 000 tokens)

| | graphs ON | graphs OFF | ratio ON/OFF |
|---|---|---|---|
| capture | **80 graphs in 1.70 s**, 32/32 steps on its own domain | no domain (by design) | - |
| TTFT | 85.65 s | 93.87 s | 0.91 (tier: 2267 vs 2391 slots) |
| decode | **1.283 tok/s** | 0.601 tok/s | **2.13x** |
| median decode step | **734.7 ms** | 1 676.7 ms | **2.28x** |
| wall | **110.6 s** | 147.1 s | 0.752 |
| tier hit / GB read | 22.14% / 500.69 GB | 25.52% / 478.93 GB | - |
| tokens | identical to eager (32/32) | | |

## 4. Do four captured domains fit in 30.5 GiB?  HBM yes; the engine cannot use them.

* One domain is exactly **1 701 517 824 B = 1.585 GiB = 405.7 MiB per card** (measured identically by the
  smoke run, the N=1 run and the fit probe; `_DecodeGraphs.bytes()` = registers + the 2-parity landing
  slab, and the peak delta per domain is the same 0.396 GiB/card, so the graph pool itself adds nothing
  measurable).  Four domains = 6.34 GiB total, 1.585 GiB/card.
* `--mode fit --fit-domains 4 --fit-prefill 2000`: four 2 000-token States, then four domains captured
  **sequentially**; all four captured 80 graphs (1.3-1.9 s each) and all four replayed their own State's
  next token (0.34 s).  Peak after the 4th domain: **18.86 / 4.45 / 4.44 / 4.08 GiB** on npu:0..3 against
  the 30.48 GiB share; final allocated 18.62 / 4.39 / 4.38 / 4.02.  No OOM.
* Cross-check against the real tier: the 4 x 2k OFF peaks (20.6-20.7 GiB on the expert cards) plus four
  domains (1.585 GiB/card) is ~22.1 GiB < 30.48, and the ON peaks already include one leaked domain.
  So **four captured conversations fit the 30.5 GiB premise with ~8 GiB to spare** - the blocker is the
  single `model._decode_graphs` slot and the capture-on-one-stream collision, not HBM.

## 5. Tier and page cache

* Graphs ON builds a 2 267-slot tier vs 2 391 for OFF (the build's 1 GiB/card graph reserve, -124 slots):
  the capture is not free even when it never captures.
* OFF #3 split at the first decode step: prefill **1 493.58 GB** (42 204 misses), decode **99.16 GB over
  128 steps = 0.775 GB/step** (26 958 hits / 2 802 misses, 90.6% hit during decode).  Per request:
  398.18 GB (373.4 prefill + 24.8 decode); the ON rows' GB/req is the scenario total / 4 and includes
  the failed request's reads.  The pre-264ffb9 record had 2.55 GB/step; today's serialised
  fetch makes the four identical streams share their decode fetches, hence the lower figure - it is a
  measurement today, not a comparison to yesterday's engine.
* Page cache: `read_bytes` 0.00-0.07 GB against `rchar` 1 592.7-1 622.6 GB, i.e. **100.0% of the 1.6 TB
  came from the host page cache** (yesterday's par4 was 84%).  That is why today's prefill is 1.8x faster.

## 6. Tests

`python3 -m unittest discover -s ascend -p "test_*.py" -t .` -> **Ran 225 tests ... OK (skipped=8)** in
68.1 s, after the benchmarks, with cards 0-3 free.  No engine file was edited; `git status` shows only
the new `ascend/bench_4card_graphs.py` and the new result files (not committed).
