# Four 2 000-token conversations, captured decode serialised - four cards, hbm-share 0.5

Date 2026-11-27 05:14-06:00 Asia/Shanghai.  Host `blue-910b3`, container `maya-ascend-dev`, physical
cards 0-3 (`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`).  All eight cards were free before the first arm
(`npu-smi info`), and every arm ended with its cards released (the next build got its share back).
Engine: `ascend/bench_4card_graphs.py` driving the same `bench_serve.concurrent_prefill_decode` the 4 x 2k
record used.  This round fixes the capture collision in `ascend/dsv41/model.py` and re-measures the arm.

## 0. The result in one line

**All four conversations now capture and keep their own 80-graph domain, and the captured decode is
2.6-3.0x the eager decode at N=4**: 128/128 decode steps ran on their own domain in every ON arm, four
domains alive at the end (`model.graphs=True`, 320 graphs total), no capture failure and no leak.  Wall
209.8 / 209.9 / 208.1 s ON against 271.2 / 268.6 / 267.1 s OFF (paired ratio **0.773 / 0.782 / 0.779**);
per-request decode **1.05 / 1.04 / 0.98 tok/s** against **0.35 / 0.36 / 0.35** (paired **3.00 / 2.88 /
2.76x**); median decode step 774-865 ms against 2 777-2 826 ms.  TTFT is unchanged (~175-180 s).  Peak HBM
22.12 GiB on the tightest card against the 30.48 GiB share.  Every token stream is identical to the eager
arm.

## 1. What the collision actually was, and the fix

The old `_graph_decode` treated "the one `model._decode_graphs` is not my State" as "build a new domain":

    if g is None or stale or g.state is not state or g.state.kv is not state.kv:
        g = self._decode_graphs = _DecodeGraphs(self, state, ...)      # model.py:2289-2299 (d0aeaac)

Four threads did that at the same instant, so four `torch.npu.graph` captures were open on one process.
Nothing in that line is a stream: each `_DecodeGraphs` owns its own `GraphRunner`, and each `GraphRunner`
its own capture stream.  The shared resource is the *device synchronisation* the vendor capture performs:
`torch_npu.npu.graphs.graph.__enter__` calls `torch.npu.synchronize()`, and `GraphRunner._capture_npu`
does it again after the body.  ACL refuses that while the stream it waits on is being captured:

    Not_Supported(EE1016): Synchronizing the device failed. Reason: Stream (stream_id=45) on
    device_id=0 is being captured. End the stream capture before synchronizing the device.

so the first two simultaneous captures killed each other, `_graph_decode` disabled graphs process-wide,
and the threads that then hit the harness's device sync died (`RuntimeError: ... stream is captured`).
The same EE1016 also explains the 1.585 GiB leak: the domains the racing threads built were overwritten
in the single `_decode_graphs` slot and never closed.

**Per-card was measured and is not enough.**  The first fix (a per-card `RLock`) serialised captures on
one card and still failed at layer 9-14: a capture on `npu:0` failed naming a capture open on `npu:1`
(and a `g_post` on `npu:1` naming one on `npu:0`).  A four-card trivial probe reproduced it (`card 0`
failed `synchronize` while cards 1-3 were capturing).  The no-argument synchronise is not reliably
scoped to the calling card when two captures are open.  The shipped lock is therefore **process-wide**.

Where it lives (all `ascend/dsv41/model.py`):

* `_CAPTURE_LOCK` / `_capture_exclusive` (1201-1240): one process-wide `RLock`; it sets the graph's card
  current, holds the lock across capture *and* graph teardown (`_DecodeGraphs._runner`, `_capture`,
  `close`), and restores the caller's current device **and stream** in `finally`.
* `Dsv41._graph_domains` (1293-1311, 2302-2446): one domain per State, keyed by `id(state)` (the domain
  holds the State, so the id cannot be reused while it is a key), under `Dsv41._graph_lock`; the
  `_decode_graphs` attribute is kept as the most-recent alias for the existing readers.
* Capture failure (`_graph_abandon`, 2411): closes that conversation's domain, removes it, marks the
  State eager in a `WeakSet` (no strong State/KV reference), logs the reason and whether the HBM came
  back; the other conversations keep their domains.  Build failure (OOM) is caught the same way.
* `release_graphs` (2302) closes every live domain (failures collected and raised once); `adopt_graph_domain`
  (2330) lets the domain-fit probe register a domain it built itself.

## 2. The leak's status

**No leak in the measured arms**: the ON runs end with four live domains and zero `graph_failures`, and
the process's teardown `release_graphs` closed all four (no "release_graphs failed" line).  A *failed*
capture now closes the half-built domain before falling back - `test_a_failed_capture_frees_its_domain...`
pins that - and the old leak path (a domain overwritten in the single slot) no longer exists.  A domain
whose teardown itself fails is counted against the bound (`_graph_leaked`) for the life of the process and
named in the log, as before.

## 3. Policy for N conversations (loud, bounded)

One captured domain **per conversation**, up to `graph_max_conversations = 4` by default
(`DSV41_GRAPH_MAX_CONVERSATIONS` overrides): 4 is the measured memory fit (one domain is 1.585 GiB,
405.7 MiB a card; four is 1.585 GiB a card, and the OFF peaks plus four domains land at ~22.1 GiB against
the 30.48 GiB share).  At the bound a new conversation runs eagerly and logs once, naming the bound, the
per-domain cost and its own id; it is not evicted silently, and a live conversation's domain is never torn
down to make room.  A capture/replay failure retires only that conversation (eager from then on), not the
process.  `serve_engine.py` was left unchanged: its `--parallel > 1` lease still grants one captured
conversation at a time, so the *server* path is unchanged by this round and still eager at N=4; the model
is what now supports N domains, and the direct-drive arm below is what proves it.

## 4. 4 x 2 000 tokens, 32 generated, four cards, `--hbm-share 0.5`, graphs ON against OFF

Command, run once per arm, per process (`on`/`off`; output `serve-4card-par4-2k-locked-{on,off}-{1..3}.json`):

    python3 -m ascend.bench_4card_graphs --model /mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8 \
      --tokenizer /mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json \
      --cards npu:0,npu:1,npu:2,npu:3 --graphs on|off --parallel 4 \
      --prompt-tokens 2000 --gen 32 --max-context 2048 --hbm-share 0.5 --expert-reserve-gib 13 \
      --out ascend/results/serve-4card-par4-2k-locked-{on|off}-{n}.json

Interleaved ON / OFF / ON / OFF / ON / OFF (starts 05:29, 05:34, 05:39, 05:44, 05:49, 05:54).

| run | ok | TTFT per request (s) | spread | decode tok/s | per-req tok/s | median step (s) | wall (s) | peak HBM npu:0..3 (GiB) | tier hit | GB read | captured |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ON #1 | 4/4 | 179.23, 179.22, 179.20, 179.18 | 0.05 | 4.18 | 1.05 | 0.774-0.775 | 209.8 | 20.34 / 22.11 / 19.21 / 21.64 | 37.21% | 1615.2 | **4/4** |
| ON #2 | 4/4 | 179.02, 179.00, 178.97, 179.01 | 0.05 | 4.14 | 1.04 | 0.785-0.788 | 209.9 | 20.24 / 22.11 / 19.21 / 21.64 | 37.19% | 1615.5 | **4/4** |
| ON #3 | 4/4 | 175.87, 175.79, 173.05, 175.92 | 2.87 | 3.66 | 0.98 | 0.847-0.901 | 208.1 | 20.10 / 22.12 / 19.22 / 21.65 | 34.85% | 1675.7 | **4/4** |
| OFF #1 | 4/4 | 180.11, 180.18, 180.20, 180.16 | 0.09 | 1.40 | 0.35 | 2.813-2.817 | 271.2 | 20.94 / 20.67 / 17.91 / 20.61 | 38.08% | 1592.7 | no domain (by design) |
| OFF #2 | 4/4 | 179.29, 179.26, 179.26, 179.29 | 0.03 | 1.43 | 0.36 | 2.777-2.780 | 268.6 | 20.35 / 20.70 / 17.92 / 20.62 | 38.08% | 1592.7 | - |
| OFF #3 | 4/4 | 177.90, 176.92, 177.91, 176.85 | 1.06 | 1.42 | 0.35 | 2.818-2.826 | 267.1 | 20.12 / 20.68 / 17.91 / 20.61 | 38.08% | 1592.7 | - |

Paired ratios (each pair is the two runs under the same host/cache state, same prompt, same 32 tokens):

| pair | wall ON/OFF | decode OFF/ON | per-request OFF/ON | median step OFF/ON |
|---|---|---|---|---|
| 1 | 0.773 | 2.99x | 3.00x | 3.64x |
| 2 | 0.782 | 2.90x | 2.88x | 3.53x |
| 3 | 0.779 | 2.58x | 2.76x | 3.26x |

TTFT is unaffected (ON 173.1-179.2 s, OFF 176.9-180.2 s; the capture happens on the first decode step,
after the first token has already been read from the prompt's logits).  The four serialised captures sit
inside the ON decode window: 32 steps x ~0.774 s = 24.8 s against a 30.5 s decode window, i.e. ~5.7 s of
the window is the four captures (~1.4 s each; the single-conversation probe measured 1.3-1.9 s a domain).

**Did all four actually capture?**  Yes, in every ON arm: `domains_created=4`, `domains_alive_at_end=4`,
`alive_domain_captured=320` (80 each: one g_pre per layer at its own width + one g_post),
`decode_steps_on_own_domain=128/128`, `all_conversations_captured=true`, `model_graphs_at_end=true`,
`graph_failures=[]`.  The four `graphs: conversation 0x.... captured its own domain (1/4 ... 4/4 ...)`
lines are in each ON log.

Peaks and tier: ON costs the four domains (1.585 GiB total, 405.7 MiB a card) on top of the same load -
tightest card 22.12 GiB against 30.48 GiB - and the build's 1 GiB/card graph reserve lowers the tier from
2 391 to 2 267 slots (-124).  The three ON arms' `GB read` is 1615.2 / 1615.5 / 1675.7 GB; the third's
higher read and lower hit (34.85% vs 37.2%) is the same run whose median step is 0.85-0.90 s instead of
0.77 s, i.e. host/cache noise, not a capture failure - it is reported rather than dropped.

## 5. Numerics

Every token stream of every request in all six arms is identical: the 32 generated ids of all four
conversations match the eager arm token for token (checked across all runs, not only within one).  The
eager arm is `host_mixes=True`; the captured arm forces the device mHC branch (`host_mixes=False`), so the
like-for-like bar is the existing N=1 ON/OFF comparison (tokens identical, 19/19 streams last round) plus
`test_decode_graphs.py`'s captured-vs-eager gate, which still passes bit-for-bit on CPU and covers the
bucket-crossing re-captures.

## 6. Tests

`python3 -m unittest discover -s ascend -p "test_*.py" -t .` -> **Ran 228 tests ... OK (skipped=8)** in
71.2 s, cards 0-3 free.  New tests (CPU, real lifecycle, `ascend/tests/test_decode_graphs.py`):

* `test_two_conversations_capture_concurrently_and_both_keep_their_domain` - both threads start from a
  barrier, every `GraphRunner.capture` is slowed so an overlap would show, and the test asserts **peak
  concurrent capture == 1**, two domains, 8/8 graphs each, and the eager tokens.
* `test_a_failed_capture_frees_its_domain_and_only_that_conversation_goes_eager` - an injected capture
  failure leaves no domain, no HBM claim, the State marked eager, no recapture on the next step, and the
  eager token stream.
* `test_the_bound_sends_the_next_conversation_eager_and_says_so` - with the bound at 2, the third
  conversation is refused loudly (`_graph_cap_logged`), no live domain is evicted, and its token is the
  eager one.

## 7. Caveats

* The host's page cache is warm: prefill reads 1592.7-1675.7 GB of which `read_bytes` is 0.00 GB, i.e.
  100% of `rchar` came from the page cache.  Absolute TTFTs are therefore not comparable to a cold run,
  and the three OFF arms drifted by only 2.9 s this round (previous rounds drifted 11.4 s).
* The serve engine's lease is out of this round's scope and still caps a warmed multi-conversation server
  at one captured conversation; the measurement above bypasses it with `bench_4card_graphs`, which uses
  the model's own concurrency.  Relaxing the lease is the next step if the server arm is wanted.
* `ascend/bench_serve.py` gained an optional `sync=` argument (default unchanged) so a harness
  synchronise that lands inside another thread's ~1.5 s capture waits for it instead of failing with
  EE1016; the wait is charged to that thread's step and is not hidden.  Eager arms never retry.
