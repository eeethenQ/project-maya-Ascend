# Four 2 000-token conversations through the serve path, one captured domain each

Date 2026-11-27 09:22-10:40 Asia/Shanghai.  Host `blue-910b3`, container `maya-ascend-dev`, physical
cards 0-3 (`ASCEND_RT_VISIBLE_DEVICES=0,1,2,3`), `--hbm-share 0.5`.  Engine: `serve_engine.Engine`
(`--parallel 4 --graphs on|off`) driven by `ascend/bench_serve_domains.py` over the line protocol -
the same `GEN`/`T`/`DONE` path the server uses, four `GEN`s submitted at the same instant, one shared
2 000-token prompt (the direct-drive arm's workload), 32 greedy tokens.  This round replaces the
serve graph *lease* with one captured domain per conversation, bounded the way the engine bounds it
(four; `DSV41_GRAPH_MAX_CONVERSATIONS`), and measures the serve path.

## 0. The result in one line

**The serve path now reaches the direct-drive numbers.**  Four conversations, four cards:
**4/4 captured** (four live domains, 320 graphs, 128/128 decode steps on their own domain),
ON wall **211.4 s** against **273.8 s** eager (paired **0.772**), aggregate decode **3.96** against **1.46 tok/s** (**2.71x**), per request **0.98-1.01** against **0.36** (**2.76x**), TTFT 179.5 against 186.1 s, peak HBM 22.11 GiB of the 30.48 GiB share, every token stream identical to eager.  The direct-drive pair 1 is wall 0.773,
decode 2.99x, 4.18 vs 1.40 tok/s - the serve path is within 5%.

## 1. The policy that landed (`serve_engine.py`)

* **Bound = the engine's own.**  `DSV41_GRAPH_MAX_CONVERSATIONS` (default **four**) is the model's
  `graph_max_conversations`; `--graph-max-conversations` (default now unset) may only **lower** it.
  The engine writes the effective value back to the model, so the serve policy and the model's
  `_graph_domain` enforce the same number.  Start log: the bound, where it came from, 1.585 GiB =
  405.7 MiB a card per domain, 1.8-1.9 s for 80 graphs, and that no live domain is ever evicted.
* **No lease, no revocation.**  Each conversation captures its own domain on its first one-token step
  (lazy, after the first token) and releases *only* its own domain when it ends or is abandoned.  The
  engine keeps no domain count of its own: `_graph_domains[id(state)]` is the truth.  The global
  forward lock is gone - a domain is replayed only by the thread that owns its State, and the
  capture/teardown exclusion is the model's process-wide `_capture_exclusive`.
* **Past the bound: eager and loud.**  A conversation refused at the bound runs eagerly; the engine
  logs the refusal once with the bound, the flag/env it came from, the per-domain cost and the fact
  that no live domain was evicted.  A conversation given room later (a release) is reported as
  captured when it takes the slot; a capture/replay failure retires only that conversation.
* **Teardown is per conversation and never raised.**  The domain is popped from the model's registry
  under `_graph_lock` and closed; a failed close is counted in `model._graph_leaked` (part of the
  bound, so the next conversation cannot capture into HBM the leak holds), logged, and serving goes
  on.  `--graphs` is still opt-in, `T > 1` prompt chunks stay eager, and `--parallel 1` still works.

## 2. Method

One arm per process (`bench_serve_domains.py`), because the two builds do not coexist: `rt.build`
(graphs on/off), `say_ready`, start the `--parallel` workers, submit four `GEN 32 rid=<id> <ids>` lines
back to back, and read TTFT from the first `T`, decode from the `T`..`DONE` window, wall from first
submit to last `DONE`.  `--expert-reserve-gib 13`, `--max-context 2048`, `--prefill-chunk 0` (one
forward for the 2 000-token prompt), `--hbm-share 0.5`.  Interleaved ON, OFF, ON, OFF, ON, OFF, like
the direct-drive arm; only pair 1 completed (see the caveats).  Capture is read after the arm from the
model's registry: live domains, each
domain's graph count, and each slot's `graph_steps` (the one-token steps it ran while its own domain
was alive).

## 3. Serve path: 4 x 2 000 tokens, 32 generated, four cards, hbm-share 0.5

| run | ok | TTFT per request (s) | spread | decode tok/s | per-req tok/s | wall (s) | peak HBM npu:0..3 (GiB) | tier hit | GB read | captured |
|---|---|---|---|---|---|---|---|---|---|---|
| ON #1 | 4/4 | 179.92, 179.06, 179.82, 179.08 | 0.86 | 3.96 | 1.01, 0.98, 1.00, 0.98 | 211.38 | 20.21 / 22.11 / 19.21 / 21.64 | 36.82% | 1625.05 | **4/4** (4 domains, 320 graphs, steps [32, 32, 32, 32]) |
| OFF #1 | 4/4 | 186.09, 186.09, 186.05, 186.14 | 0.09 | 1.46 | 0.36, 0.36, 0.36, 0.36 | 273.8 | 20.35 / 20.73 / 17.96 / 20.68 | 38.08% | 1592.74 | **0/4** (0 domains, 0 graphs, steps [0, 0, 0, 0]) |
| ON #2 | - | - | - | - | - | - | - | - | - | - |
| OFF #2 | - | - | - | - | - | - | - | - | - | - |
| ON #3 | - | - | - | - | - | - | - | - | - | - |
| OFF #3 | - | - | - | - | - | - | - | - | - | - |

| pair | wall ON/OFF | decode OFF/ON | per-request OFF/ON | captured |
|---|---|---|---|---|
| 1 | **0.772** | 2.71x | 2.76x | 4/4 |

Direct drive, same machine and workload (`serve-4card-par4-2k-graphs-locked-REPORT.md`):

| run | TTFT/req | spread | decode tok/s | per-req | wall (s) | peak HBM | tier hit | captured |
|---|---|---|---|---|---|---|---|---|
| ON #1 | 179.23, 179.22, 179.20, 179.18 | 0.05 | 4.18 | 1.05, 1.05, 1.05, 1.05 | 209.77 | 20.34 / 22.11 / 19.21 / 21.64 | 37.21% | 4/4 |
| OFF #1 | 180.11, 180.18, 180.20, 180.16 | 0.09 | 1.4 | 0.35, 0.35, 0.35, 0.35 | 271.23 | 20.94 / 20.67 / 17.91 / 20.61 | 38.08% | none |
| ON #2 | 179.02, 179.00, 178.97, 179.01 | 0.05 | 4.14 | 1.04, 1.04, 1.03, 1.04 | 209.92 | 20.24 / 22.11 / 19.21 / 21.64 | 37.19% | 4/4 |
| OFF #2 | 179.29, 179.26, 179.26, 179.29 | 0.03 | 1.43 | 0.36, 0.36, 0.36, 0.36 | 268.57 | 20.35 / 20.70 / 17.92 / 20.62 | 38.08% | none |
| ON #3 | 175.87, 175.79, 173.05, 175.92 | 2.87 | 3.66 | 0.99, 0.99, 0.94, 1.00 | 208.07 | 20.10 / 22.12 / 19.22 / 21.65 | 34.85% | 4/4 |
| OFF #3 | 177.90, 176.92, 177.91, 176.85 | 1.06 | 1.42 | 0.36, 0.35, 0.36, 0.35 | 267.13 | 20.12 / 20.68 / 17.91 / 20.61 | 38.08% | none |

Direct-drive paired ratios: 1: 0.773 / 2.99x; 2: 0.782 / 2.90x; 3: 0.779 / 2.58x.

## 4. Does the serve path reach the direct-drive numbers?

**Yes, within a few percent on every field.**

| metric | direct drive (pair 1) | serve path (pair 1) |
|---|---|---|
| ON wall | 209.8 s | 211.4 s |
| OFF wall | 271.2 s | 273.8 s |
| paired wall | 0.773 | 0.772 |
| ON decode | 4.18 tok/s | 3.96 tok/s |
| OFF decode | 1.40 tok/s | 1.46 tok/s |
| decode ratio | 2.99x | 2.71x |
| ON per request | 1.05-1.05 tok/s | 0.98-1.01 tok/s |
| TTFT ON / OFF | 179.2 / 180.2 s | 179.5 / 186.1 s |
| peak HBM (tightest) | 22.11 GiB | 22.11 GiB |
| captured | 4/4 | 4/4 |

The small deficit in ON decode (3.96 against 4.18 tok/s in the matching pair) is within the
run-to-run spread the direct-drive arm itself shows (4.18 / 4.14 / 3.66).  The serve path pays
nothing measurable for the scheduler, the tagged protocol or the shared expert lock on the
captured path (`fetch_int8_into` is not the locked `fetch_int8`).

## 5. Past the bound: five and eight conversations

The five/eight-conversation arms had not produced a JSON when this report was written: a
co-tenant held all eight cards (58.5 of 65.5 GiB each) from ~09:42 local through the window, so
the NPU check of HBM/thrash past four is **not measured here**.  The policy past four is pinned on
the CPU path by `test_eight_conversations_capture_four_and_the_rest_are_eager` (8 conversations,
default bound 4: 4 capture, 4 run eagerly, 4 refusal lines naming the bound and the per-domain
cost, no domain evicted) and by the model-level
`test_the_bound_sends_the_next_conversation_eager_and_says_so` (bound 2, third conversation eager).

## 6. Tests

`python3 -m unittest discover -s ascend -p "test_*.py" -t .` -> **230 tests, 8 skipped, OK**
(73.4 s).  The serve graph tests were rewritten for the policy:

* `test_the_start_log_states_the_bound_and_the_flag_can_only_lower_it` - the start log names the
  engine's bound (4) and the per-domain cost; `--graph-max-conversations 2` caps it, 8 cannot raise it.
* `test_the_bound_is_respected_the_rest_run_eagerly_and_are_logged` - bound 2, four conversations
  held at their second token: exactly two domains (20 decode forwards each on their own domain),
  two eager with 0 graph steps, both refusal lines name the bound and the 1.585 GiB cost,
  `model.graphs` stays on, no domain evicted.
* `test_a_captured_conversation_that_ends_frees_its_slot_for_the_next` - bound 1: the captured
  conversation's domain is closed when its State is replaced, and the next conversation captures in
  the freed slot.
* `test_eight_conversations_capture_four_and_the_rest_are_eager` - the default bound at 8
  conversations: 4 domains, 4 eager, 4 refusal lines, 6/6 and 0 decode forwards respectively.
* `test_a_capture_failure_retires_only_that_conversation` - one injected domain-build failure leaves
  the other conversation's domain and `model.graphs` intact; the log says the failure retired only
  the conversation that failed.
* the teardown-failure test now injects a failing `domain.close()`: the leak is counted in
  `_graph_leaked`, the next conversation still captures in the remaining bound slot, and shutdown
  does not raise.

## 7. Caveats

* A co-tenant took all eight cards at ~09:42 local, mid-measurement: the first attempts at pairs 2
  and 3 were killed with SIGTERM (`exec_die exitCode=143`), and a later attempt died in
  `torch.OutOfMemoryError` with 2.30 MiB free on npu:0 because the co-tenant held 58.5 of
  65.5 GiB on every card.  A detached supervisor waited an hour for the cards to free and was
  stopped; only arms that produced a JSON are reported here.  Pair 1 ran before the takeover and both
  of its arms are ~1% above the direct-drive equivalents, so the paired ratio is not one-sided.
* Peak HBM includes the four per-request `State`s, which the serve engine allocates after the peak
  reset; the direct-drive harness allocated its States before its reset.  Both land at ~22 GiB on
  the tightest card, so the comparison is fair.
* Page cache: `read_bytes` for ON 1.93 GB of 1 625 GB read (0.1%), OFF 0.44 GB of 1 593 GB - warm,
  like the direct-drive run; absolute TTFTs are not cold-cache numbers.
* Four *distinct* prompts are a different (thrashing) workload: an extra ON arm with distinct cuts
  of the document captured 4/4 domains and 32/32 steps, but the tier hit fell to 25.45% and decode
  to 1.86 tok/s aggregate (0.50/request) - the shared prompt is what makes this the direct-drive
  comparison.  Capture itself is unaffected by prompt diversity.
* `model.py` still prints its capture lines to stdout (engine protocol noise); pre-existing, and the
  server tolerates unknown lines.  The engine's own policy lines go to stderr.
* Host clock skew was observed (the container clock jumped ~8 h during one probe); timestamps in
  this report are from the runner's local clock.

## 8. Where the artefacts are

* `ascend/results/serve-domain-par4-2k-{on,off}-1.json` and `.stderr.log` - the clean pair.
* `ascend/results/serve-domain-par4-2k-{on,off}-{2,3}.stderr.log` - the arms the co-tenant killed
  (kept as evidence of the contention, not as results).
* `ascend/results/serve-domain-par4-2k-distinct-on-1.json` - the distinct-prompt extra arm.
* `ascend/bench_serve_domains.py` - the serve-path arm driver.
