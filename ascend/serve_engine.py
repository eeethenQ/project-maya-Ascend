"""ascend/serve_engine.py - the Ascend DeepSeek-V4.1 engine behind this project's serve line protocol.

`serve/server.py` starts an engine as `[exe, "--serve", *args]` and talks to it over stdin/stdout in lines
(`serve/server.py:956-1030`, and `serve/test_engine_contract.py` is the contract): the engine holds one
conversation, the server sends the whole prompt each time, and the engine reports how much of it it
already had.  `ascend/strata-ascend` is the `exe` that starts this module, so a config like

    {"exe": "<repo>/ascend/strata-ascend",
     "args": ["--model", "/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8",
              "--tokenizer", "/mnt/weights/DeepSeek-V4-Flash-0731-w8a8/tokenizer.json"],
     "tokenizer": "...", "model_name": "DeepSeek-V4.1-Flash"}

gives the OpenAI/Anthropic API, SSE, tools and the web UI over this engine with **no change to
`serve/`** - which is the seam the project was built with, and the cheapest way for a new backend to
become usable.

Lines this engine prints:

    INFO context=... kv=... expert_slots=... experts=... vram_gb=... engine_kind=dsv41-ascend
    <engine log on stderr>, including the "CUDA0 VRAM before the expert pool: ... state/KV (<ctx> ctx) ..."
        line the server reads for its context cost
    READY <ctx> stop pause
    PP <read> <total> <ms> <tok_s>          while reading a prompt
    T <token id>                             one per generated token
    STAT <the counters>                      every 8 tokens (the server's dashboard reads them)
    DONE <n> <prompt> <prompt_ms> <decode_ms> <finish> <da> <do> <reused> <hits> <lookups>
    ERR <reason>                             a request this engine cannot serve

Lines it reads: `GEN <max_new> [k=v ...] <ids>`, `STOP`, `PAUSE [keep]`, `WRAP`, `QUIT`.

With `--graphs` the one-token decode runs as captured NPU graphs, exactly as `run_dsv41 --graphs` does:
a `g_pre`/`g_post` pair per layer, one `g_pre` per indexer selection bucket, and prompt chunks (`T > 1`)
stay eager.  A graph bakes the buffers of the `State` it was captured against, so the graphs belong to a
conversation: this engine captures them for the held conversation, releases them when that conversation
ends (a diverging prompt) or is abandoned (a request cut during its prompt), and reports what the capture
cost (1.8-1.9 s for 80 graphs, charged to the decode after the first token - the first token is read from
the prompt's own logits, so a request never waits on it to answer).  At most
`--graph-max-conversations` captured conversations may be alive at once (default 1, which is what
`runtime.build` reserves HBM for); past it a conversation runs eagerly and says so.

Several conversations at once (`--parallel N`)
------------------------------------------------
The default is one conversation, exactly as before.  With `--parallel N` the engine admits up to N
conversations at once, each with its own `State` (and, with `--parallel-streams`, its own NPU stream), and
advertises `multi=N` in READY
so `serve/server.py` gives its FIFO N places and tags every request (`GEN ... rid=<id>`, `T <rid> <token>`,
`DONE <rid> ...`, `STOP <rid>`).  A request is placed on the idle conversation sharing the longest prefix
with its prompt (so a chat's next turn still reuses the state, and a divergent prompt starts a new one); a
request past N waits for a conversation to free, and the engine logs that it queued.

What this does **not** do is batch: `Dsv41.forward` takes one `State` and one token list, and `KvPlanes`
refuses `batch != 1`, so N conversations cannot share a forward.  Each conversation therefore still pays
its own 32 242 operator calls per token; the overlap is host dispatch across the request threads (and, only
with the opt-in `--parallel-streams`, NPU execution across streams), not one batched call.  The measured consequence is in `docs/ASCEND-DSV41.md` §6 and §16.

Two consequences of sharing one model, both handled here:

* **The expert streamer is not reentrant.**  `ExpertStreamer._read_many` rotates one pinned staging ring
  and, with a spread tier, moves the process's current device around each copy; two forwards fetching at
  once could hand each other's expert bytes to the card.  With `--parallel > 1` the engine wraps
  `fetch_int8` in one lock (host read + H2D only), so the shared ring is used one fetch at a time.  The
  rest of a forward - attention, the expert matmuls - still overlaps across the request threads.
* **Graphs are one conversation's buffers.**  A captured decode is `T == 1` for one `State` and the model
  holds one graph domain, so `--parallel > 1` runs eagerly and says so; `--graphs` is only meaningful with
  `--parallel 1`.  The two are never silently mixed: the conflict is logged at start.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import queue
import sys
import threading
import time
import traceback

import torch

from .dsv41 import runtime as rt
from .dsv41.graph_runner import GraphTeardownError

EOS = 1

# one line is one write: several request threads emit concurrently in --parallel mode
_OUT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _OUT_LOCK:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()


def _out(msg: str) -> None:
    with _OUT_LOCK:
        sys.stdout.write(msg + "\n")
        sys.stdout.flush()


class Conversation:
    """One conversation's State and the tokens it covers.

    The single-conversation engine *is* one of these (`Engine` has `state` and `held` too, and passes
    itself where a conversation is expected), so `generate` has one code path.  In `--parallel` mode each
    scheduler slot is a `Conversation`; the scheduler only reads `held` while the slot is idle, and the
    serving thread owns the State.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.state = None
        self.held: list[int] = []
        self.busy = False
        self.rid: int | None = None
        self.queue: "queue.Queue" = queue.Queue()

    def common_prefix(self, ids: list[int]) -> int:
        n = 0
        for a, b in zip(ids, self.held):
            if a != b:
                break
            n += 1
        return n


def prompt_chunks(n_tokens: int, chunk: int, max_context: int) -> list[int]:
    """The sizes of the eager `Dsv41.forward` calls that read `n_tokens` fresh prompt tokens.

    A prompt is read in pieces of at most `chunk` (`--prompt-chunk`) tokens, and **the chunk size is a
    multiplier on the prefill, not a detail of it.**  Each piece is its own eager forward, a piece of any
    length routes ~377 of the 384 experts in every layer (a 512-token piece already draws 3 072 pairs),
    and the expert tier holds only ~62 of them per layer.  So every extra piece re-reads almost the whole
    expert set from the storage array: four 512-token pieces of a 2 000-token prompt cost about four
    prefills, one piece costs one.  `docs/ascend-notes/prefill-analysis.md` §5 (O1) has the accounting;
    measured TTFT and peak HBM at each size are in `docs/ASCEND-DSV41.md` §15.

    Two bounds are applied, in order:

    * `chunk <= 0` means the whole prompt in one forward (the one-chunk case, without having to know the
      prompt's length first).
    * a forward longer than `max_context` would write more rows into the KV planes' ring than the ring
      has, overwriting rows of its own chunk before they are read, so the chunk is clamped to
      `--max-context`.  That is a correctness bound, not a performance one; it is also why raising
      `--max-context` is a prerequisite for reading a longer prompt in one forward.
    """
    n = max(0, int(n_tokens))
    if n == 0:
        return []
    size = int(chunk)
    size = n if size <= 0 else size
    if int(max_context) > 0:
        size = min(size, int(max_context))
    size = max(1, size)
    return [min(size, n - done) for done in range(0, n, size)]


class Engine:
    """One conversation, held across requests: the state a reply left is where the next prompt starts."""

    def __init__(self, args):
        self.args = args
        self.parallel = max(1, int(getattr(args, "parallel", 1) or 1))
        # A captured decode is T == 1 for one State and the model holds one graph domain: it cannot be
        # shared by N conversations, and switching domains on every step is the thrash the graph work
        # exists to avoid.  The policy is explicit - --parallel > 1 runs eagerly - never a silent mix.
        self._graphs_conflict = bool(getattr(args, "graphs", False)) and self.parallel > 1
        self._graphs_wanted = bool(getattr(args, "graphs", False)) and self.parallel == 1
        t0 = time.time()
        self.rt = rt.build(model=args.model, device=args.device, dtype=args.dtype,
                           hbm_share=args.hbm_share, expert_slots=args.expert_slots,
                           io_threads=args.io_threads, expert_batch=args.expert_batch,
                           expert_reserve_gib=args.expert_reserve_gib,
                           expert_devices=args.expert_devices.split(",") if args.expert_devices else None,
                           engram_tokenizer=args.engram_tokenizer or None,
                           graphs=self._graphs_wanted,
                           graph_verify=getattr(args, "graph_verify", False),
                           verbose=args.verbose)
        self.cfg = self.rt.cfg
        self.load_s = time.time() - t0
        self.state = None
        self.held: list[int] = []                 # the tokens the state covers
        self.control: queue.Queue[str] = queue.Queue()
        self.input: queue.Queue[str | None] = queue.Queue()
        self.quit = False
        self.stats = {"tokens": 0, "steps": 0, "decode_ms": 0.0}
        # -- several conversations at once (--parallel) -----------------------------------------
        # `self` is the single conversation of the default engine and the first slot's owner is never
        # it: with --parallel the scheduler hands requests to `self.slots`, each a `Conversation` with
        # its own State, and `generate` treats `self` and a slot alike.  `_sched` guards the slot table
        # and the control state; a worker owns its slot's State while it serves.
        self.slots: list[Conversation] = ([Conversation(f"slot{i}") for i in range(self.parallel)]
                                          if self.parallel > 1 else [])
        self._sched = threading.Lock()
        self._cut = 0                            # incremented by a global STOP/PAUSE/QUIT
        self._wrap = 0                           # incremented by WRAP (a thinking request's soft cut)
        self._rid_cut: set[int] = set()          # request ids the server has stopped
        self._waiting: "collections.deque[tuple[str, int]]" = collections.deque()
        self._workers: list[threading.Thread] = []
        self._multi_running = False
        self._queued_seen = 0
        # Per-conversation NPU streams are opt-in: measured on one 910B3 they were ~10-15% slower than the
        # device's default stream at the same tier (four conversations then contend not only for the SM but
        # for the shared staging ring's per-slot events, whose cross-stream waits add to the work).  The
        # default keeps the one stream and still overlaps host dispatch across the request threads.
        self._parallel_streams = bool(getattr(args, "parallel_streams", False)) and self.parallel > 1
        self._expert_lock = threading.Lock()
        self._install_expert_lock()
        self._closed = False
        # -- the captured decode, per conversation ----------------------------------------------
        # The graphs bake the held State's buffers, so there is nothing to reuse once the conversation
        # is over.  `_graphs_held` is the conversation being served; `_graphs_leaked` counts domains a
        # failed teardown left alive (their HBM never comes back in this process), and the two together
        # are what the bound counts.  A leaked domain must never turn into a second landing zone on
        # every expert card by accident, which is what the bound is for.
        self._graph_max = max(1, int(getattr(args, "graph_max_conversations", 1) or 1))
        self._graphs_held = 0
        self._graphs_leaked = 0
        self._graphs_blocked = False             # this engine forced eager at the bound
        self._graphs_captured_seen = 0
        self._graphs_capture_s_seen = 0.0
        # The engine only needs the end-of-sequence id; serve/ does every tokenization.  `--tokenizer` is
        # therefore optional, and a path to something other than a Hugging Face tokenizer.json (the pack
        # layout, say) is a note in the log rather than a failed start.
        self.tok = None
        if args.tokenizer:
            try:
                from tokenizers import Tokenizer
                self.tok = Tokenizer.from_file(args.tokenizer)
            except Exception as e:                        # noqa: BLE001
                _log(f"dsv41: --tokenizer {args.tokenizer} is not a Hugging Face tokenizer.json ({e}); "
                     f"using the checkpoint's eos_token_id")

    # -- the start lines -----------------------------------------------------------------------
    def say_ready(self) -> None:
        cfg = self.cfg
        slots = getattr(self.rt.experts, "cache_slots", 0)
        per_expert = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        resident = self.rt.weights.resident_bytes() / rt.GIB
        experts_total = cfg.num_hidden_layers * cfg.n_routed_experts
        ctx = self.args.max_context
        # the server's context-cost reader wants this exact shape, on stderr
        _log(f"dsv41 fast: CUDA0 VRAM before the expert pool: {resident + slots * per_expert / rt.GIB:.2f} "
             f"of 32.00 GB in use - dense weights {resident:.2f}, state/KV ({ctx} ctx) "
             f"{self.kv_gib(ctx):.2f}, activations 0.50, other (context, scratch, prompt path) 0.50")
        _log(f"maya generate: session is up (ascend dsv41, {self.rt.device}, weights in {self.load_s:.1f}s, "
             f"graphs {'on' if self._graphs_wanted else 'off'}, conversations {self.parallel})")
        if self._graphs_conflict:
            _log(f"dsv41 graphs: OFF - --graphs was asked for with --parallel {self.parallel}.  A captured "
                 f"decode is one State at T == 1 and the model holds one graph domain, so N conversations "
                 f"cannot replay it (a second conversation would recapture on every step, the thrash "
                 f"bench_serve measures); this engine runs eagerly.  Use --parallel 1 with --graphs, or "
                 f"--parallel N without it")
        if self._graphs_wanted:
            _log(f"dsv41 graphs: capturing the one-token decode per conversation (g_pre/g_post per layer, "
                 f"one g_pre per indexer bucket; measured 1.8-1.9 s for 80 graphs).  At most "
                 f"{self._graph_max} captured conversation(s) alive at once; past that bound a "
                 f"conversation runs eagerly and says so.  Prompt chunks stay eager.")
        if self.parallel > 1:
            _log(f"dsv41 parallel: {self.parallel} conversations at once, one State each"
                 f"{' and one NPU stream each' if self._parallel_streams else ''}; "
                 f"a request past {self.parallel} waits for one to free (the server's FIFO has {self.parallel} "
                 f"places when it reads multi={self.parallel}).  The shared expert streamer is held one fetch "
                 f"at a time (its pinned ring is not reentrant); each conversation's KV costs "
                 f"{self.kv_gib(ctx):.3f} GiB at {ctx} context, so size --expert-reserve-gib for N "
                 f"concurrent activations")
        _out(f"INFO context={ctx} kv=bf16 kv_resident=0 expert_slots={slots} "
             f"expert_cache_mib={slots * per_expert // (1 << 20)} engine_kind=dsv41-ascend "
             f"experts={experts_total} vram_slots={slots} vram_gb={resident + slots * per_expert / rt.GIB:.2f} "
             f"ram_slots=0 ram_gb=0.0 mtp=0 dflash=0 vision_lend=0 ling=0 cpu_threads=0 cpu_threads_max=0 "
             f"pcie_share={self.args.hbm_share:.2f} logit_bias=1")
        if self.parallel > 1:
            # serve/server.py reads multi=<n>: n places in its FIFO, and every GEN carries rid=<id>
            _out(f"READY {ctx} stop pause multi={self.parallel}")
        else:
            _out(f"READY {ctx} stop pause")
        _log("maya generate: dsv41 ascend serve: GEN <max_new> [keys] <ids>, STOP, PAUSE, WRAP, QUIT")

    def kv_gib(self, ctx: int) -> float:
        """What one conversation's KV costs: 40 window planes plus the four compressed ones."""
        cfg = self.cfg
        D = cfg.head_dim
        entry = D * 2 / (1 << 30)                       # bf16 per cached row
        swa = cfg.num_hidden_layers * min(ctx, cfg.sliding_window) * entry
        long_kv = ctx * entry * len(cfg.kv_source_layer_ids)
        index_k = ctx * cfg.index_head_dim * 2 / (1 << 30) * len(cfg.kv_source_layer_ids)
        return swa + long_kv + index_k

    # -- the captured decode, one conversation at a time ----------------------------------------
    def graphs_live(self) -> int:
        """Captured conversations whose graphs are still alive: one held plus any leaked domains."""
        return self._graphs_held + self._graphs_leaked

    def _graphs_can_capture(self) -> bool:
        """Whether a new conversation may capture, logging the bound once when it refuses.

        The landing zone is 425 MB on every card the experts live on, plus that card's graph pool and
        registers; `runtime.build(graphs=True)` takes 1 GiB per card out of the expert tier for exactly
        one conversation's worth, so the default bound is 1.  A domain a failed teardown left alive is
        counted here too: it still holds its HBM, and the model has forgotten it (so it cannot be
        released later - only the process exit returns it).
        """
        if not self._graphs_wanted:
            return False
        if not bool(getattr(self.rt.model, "graphs", False)):
            # --graphs off, or the model disabled the captured path after a capture/replay failure
            return False
        if self.graphs_live() >= self._graph_max:
            if not self._graphs_blocked:
                self._graphs_blocked = True
                self.rt.model.graphs = False
                _log(f"dsv41 graphs: {self.graphs_live()} captured conversation(s) already hold their "
                     f"graphs (--graph-max-conversations {self._graph_max}); this conversation runs "
                     f"eagerly - the tier reserves one landing zone per card, not another")
            return False
        return True

    def _plan_graphs(self) -> None:
        """Count the conversation that is about to capture and report its layout and the capture cost."""
        if self._graphs_held or not self._graphs_can_capture():
            return
        self._graphs_held = 1
        _log("dsv41 graphs: this conversation captures its decode on the first one-token step "
             "(measured 1.8-1.9 s for 80 graphs); the first token is read from the prompt's own logits "
             "before that, so the request never waits on the capture to answer")
        try:
            _log(self.rt.model.graph_summary(self.state))
        except Exception as e:                           # noqa: BLE001 - a log line is not worth a request
            _log(f"dsv41 graphs: (could not describe the capture table: {type(e).__name__}: {e})")

    def _report_graph_capture(self) -> None:
        """Say what the capture cost, in the engine log the server keeps (the protocol has no field for it)."""
        g = getattr(self.rt.model, "_decode_graphs", None)
        if g is None:
            return
        captured, seconds = int(getattr(g, "captured", 0)), float(getattr(g, "capture_s", 0.0))
        graphs = captured - self._graphs_captured_seen
        spent = seconds - self._graphs_capture_s_seen
        self._graphs_captured_seen, self._graphs_capture_s_seen = captured, seconds
        if graphs > 0:
            _log(f"dsv41 stat: graphs captured {graphs} in {spent * 1e3:.0f} ms "
                 f"({seconds * 1e3:.0f} ms, {captured} graphs for this conversation); it is charged to the "
                 f"decode step, after the token that preceded it was already written")

    def _release_graphs(self, reason: str) -> None:
        """Release the graphs of the conversation being dropped; a failed teardown is logged, not raised.

        ``Dsv41.release_graphs`` raises :class:`GraphTeardownError` after dropping the domain when a
        graph could not be destroyed - its HBM is not returned in this process.  A leaked model must not
        take the server down with it: the failure is named in the log, counted against
        ``--graph-max-conversations`` (so the bound then sends the next conversation to the eager path
        instead of capturing into HBM that is not there), and serving continues.
        """
        model = self.rt.model
        domain = getattr(model, "_decode_graphs", None)
        had = domain is not None
        try:
            model.release_graphs()
        except GraphTeardownError as exc:
            if had:
                self._graphs_leaked += 1
            self._graphs_held = 0
            _log(f"dsv41 graphs: could not release the graphs of the conversation that ended "
                 f"({reason}) - {exc}.  The engine keeps serving; the leaked graph models stay alive "
                 f"until this process exits")
            return
        finally:
            self._graphs_captured_seen = 0
            self._graphs_capture_s_seen = 0.0
        if had:
            _log(f"dsv41 graphs: released {int(getattr(domain, 'captured', 0))} graphs of the "
                 f"conversation that ended ({reason})")
        # nothing is held for the dropped conversation: whether it captured or fell back to eager, the
        # domain is gone now (a capture that never happened leaves nothing behind either)
        self._graphs_held = 0
        if self._graphs_blocked and self.graphs_live() < self._graph_max:
            # only a block this engine set is undone here; a model that disabled graphs itself (a
            # capture/replay failure) must stay on the eager path
            self._graphs_blocked = False
            model.graphs = True
            _log(f"dsv41 graphs: the bound has room again ({self.graphs_live()}/"
                 f"{self._graph_max}); new conversations capture")

    def _drop_conversation(self, reason: str, conv=None) -> None:
        """Forget a conversation, releasing its graphs (its State is garbage after this).

        Only the engine's own (the single-conversation default) holds a graph domain; a `--parallel` slot
        runs eagerly, and releasing the model's graphs from a slot thread would tear down somebody else's.
        """
        conv = self if conv is None else conv
        if conv is self:
            self._release_graphs(reason)
        conv.state = None
        conv.held = []

    # -- reading the input ---------------------------------------------------------------------
    def _reader(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            # `STOP <rid>` / `PAUSE <rid> [keep]` are one request's control in --parallel mode
            if line in ("STOP", "WRAP", "PAUSE", "PAUSE keep") or line.startswith(("STOP ", "PAUSE ")):
                self.control.put(line)
            elif line == "QUIT":
                self.control.put("QUIT")
                self.input.put("QUIT")
                break
            else:
                self.input.put(line)
        self.input.put(None)

    def controls_held(self) -> str | None:
        try:
            return self.control.get_nowait()
        except queue.Empty:
            return None

    def _control_for(self, rid: int | None, cut0: int, wrap0: int) -> str | None:
        """The control this request should see now: `STOP`/`PAUSE`/`WRAP`/`QUIT`, or None.

        In single-conversation mode this is the old control queue (`--parallel` not in use).  In
        `--parallel` mode a request started at global cut generation ``cut0``/wrap ``wrap0`` is cut by a
        later global STOP/PAUSE or, if the server named it, by its own ``STOP <rid>``.  WRAP is not a cut:
        a thinking request closes its reasoning at the next token (the caller's loop does that).
        """
        if rid is None:
            return self.controls_held()
        with self._sched:
            if self.quit:
                return "QUIT"
            if rid in self._rid_cut:
                self._rid_cut.discard(rid)
                return "STOP"
            if self._cut != cut0:
                return "STOP"
            if self._wrap != wrap0:
                return "WRAP"
        return None

    # -- several conversations at once ---------------------------------------------------------
    def _install_expert_lock(self) -> None:
        """Make the one shared ExpertStreamer safe for N concurrent forwards.

        `ExpertStreamer._read_many` rotates **one** pinned staging ring (`self._slot`, `self._staging`)
        and, with a spread tier, moves the process's current device around each copy; that is written for
        one forward at a time.  Two conversations fetching at once can pick the same staging slot and hand
        each other's expert bytes to the card - a silently wrong answer, not a crash.  The wrapper takes
        `self._expert_lock` for the host read + H2D of a fetch only, and only while the multi scheduler is
        running (`--parallel 1`, and the bench's own before-arm, are untouched).  The rest of a forward -
        attention, the router, the expert matmuls - still overlaps across the request threads.

        `Dsv41.forward` reaching the streamer through `self.weights.experts.fetch_int8` means an instance
        attribute shadows the bound method, which is why this lives here rather than in `model.py` (owned
        elsewhere this round).
        """
        if self.parallel <= 1:
            return
        experts = getattr(self.rt, "experts", None)
        fetch = getattr(experts, "fetch_int8", None)
        if fetch is None or getattr(fetch, "_serve_locked", False):
            return
        lock, engine = self._expert_lock, self

        def fetch_int8(*a, **kw):
            if engine._multi_running:
                with lock:
                    return fetch(*a, **kw)
            return fetch(*a, **kw)

        fetch_int8._serve_locked = True
        experts.fetch_int8 = fetch_int8

    def _emit(self, msg: str, rid: int | None = None) -> None:
        """Write one protocol line, tagged with its request id in --parallel mode.

        `serve/server.py` routes `T`/`PP`/`DONE`/`ERR` by the id after the verb; `STAT` stays engine-wide.
        """
        if rid is not None:
            verb, sep, rest = msg.partition(" ")
            if sep and verb in ("T", "PP", "DONE", "ERR"):
                msg = f"{verb} {rid} {rest}"
        self.emit(msg)

    def emit(self, msg: str) -> None:
        """Where protocol lines go; a test or a benchmark may replace this with a collector."""
        _out(msg)

    def start_workers(self) -> None:
        """Start the `--parallel` worker threads (one per conversation slot); idempotent."""
        if self.parallel <= 1 or self._workers:
            return
        self.quit = False
        self._multi_running = True
        for slot in self.slots:
            t = threading.Thread(target=self._worker, args=(slot,), name=f"serve-{slot.name}", daemon=True)
            t.start()
            self._workers.append(t)

    def stop_workers(self, timeout: float = 120.0) -> None:
        """Stop the workers, ending any request in flight (QUIT semantics: no DONE is promised).

        The expert lock stays armed until the last worker has left: a worker can still be inside a
        forward when this is called, and disarming it then would let two fetches race the staging ring.
        """
        self.quit = True
        workers, self._workers = self._workers, []
        for slot in self.slots:
            slot.queue.put(None)
        for t in workers:
            t.join(timeout)
        self._multi_running = False

    def wait_idle(self, timeout: float = 3600.0) -> bool:
        """Wait until no conversation is running and no request is waiting; False on timeout."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            with self._sched:
                if not self._waiting and not any(s.busy for s in self.slots):
                    return True
            time.sleep(0.02)
        return False

    def submit(self, line: str) -> bool:
        """Hand one line to the `--parallel` scheduler; False when this engine is single-conversation."""
        if self.parallel <= 1:
            return False
        self._dispatch(line)
        return True

    def _dispatch(self, line: str) -> None:
        """A line from the server, in `--parallel` mode: a GEN to admit, or a control to apply."""
        if line.startswith("GEN "):
            rid = self._rid_of(line)
            if rid is None:
                self._emit("ERR multi mode needs rid=<id> on GEN (READY multi=N)")
                return
            self._admit(line, rid)
        elif line.startswith(("STOP", "PAUSE", "WRAP", "QUIT")):
            self._apply_control(line)
        elif line.startswith("GENI "):
            self._emit("ERR this engine serves text only: no images")
        elif line in ("VLEND", "VRECLAIM"):
            self._emit(f"ERR {line} text-only model: no vision encoder to lend memory to")
        else:
            self._emit(f"ERR unsupported command '{line.split()[0]}'")

    @staticmethod
    def _rid_of(line: str) -> int | None:
        fields = line.split()
        for f in fields[2:-1]:
            if f.startswith("rid="):
                try:
                    return int(f.split("=", 1)[1])
                except ValueError:
                    return None
        return None

    def _apply_control(self, line: str) -> None:
        """Apply STOP / PAUSE / WRAP / QUIT to the conversations they are meant for.

        A bare STOP/PAUSE cuts every running request (and cancels the queued ones, each with its own
        `DONE ... cancel`, so the server's drain does not wait for a line that will not come); `STOP <rid>`
        cuts one.  WRAP is not a cut.  QUIT ends the engine.
        """
        parts = line.split()
        verb = parts[0]
        rid = parts[1] if len(parts) > 1 and parts[1].isdigit() else None
        cancelled: list[tuple[int, int]] = []
        with self._sched:
            if verb == "QUIT":
                self._cut += 1
                self.quit = True
                cancelled = [(r, self._prompt_len(line_)) for line_, r in self._waiting]
                self._waiting.clear()
            elif verb == "WRAP":
                self._wrap += 1
            elif verb == "STOP" or verb == "PAUSE":
                if rid is not None:
                    self._rid_cut.add(int(rid))
                else:
                    self._cut += 1
                    cancelled = [(r, self._prompt_len(line_)) for line_, r in self._waiting]
                    self._waiting.clear()
        for r, n in cancelled:
            self._emit(f"DONE 0 {n} 0.0 0.0 cancel 0 0 0 0 0", r)

    @staticmethod
    def _prompt_len(line: str) -> int:
        fields = line.split()
        if not fields:
            return 0
        return len([x for x in fields[-1].split(",") if x])

    def _admit(self, line: str, rid: int) -> None:
        """Place a request on the idle conversation sharing the longest prefix, else queue it."""
        ids = [int(x) for x in line.split()[-1].split(",") if x]
        with self._sched:
            if rid in self._rid_cut:
                # stopped while it was on its way in: answer its DONE so the server's drain completes
                self._rid_cut.discard(rid)
                cancelled = True
                slot = None
            else:
                cancelled = False
                slot, best = None, -1
                for s in self.slots:
                    if s.busy:
                        continue
                    n = s.common_prefix(ids)
                    if n > best:
                        slot, best = s, n
                if slot is not None:
                    self._assign(slot, line, rid, self._cut, self._wrap)
                else:
                    self._waiting.append((line, rid))
                    self._queued_seen += 1
        if cancelled:
            self._emit(f"DONE 0 {len(ids)} 0.0 0.0 cancel 0 0 0 0 0", rid)
        elif slot is None:
            _log(f"dsv41 parallel: request {rid} queued behind {self.parallel} running conversation(s); "
                 f"it starts when one frees (the server should hold at most multi={self.parallel})")

    def _assign(self, slot: Conversation, line: str, rid: int, cut0: int, wrap0: int) -> None:
        """Give a request to an idle slot.  Caller holds `_sched`."""
        slot.busy = True
        slot.rid = rid
        slot.queue.put((line, rid, cut0, wrap0))

    def _worker(self, slot: Conversation) -> None:
        """Serve this slot's requests, in this thread's own NPU stream if enabled, until stopped."""
        stream = self._new_stream()
        while True:
            item = slot.queue.get()
            if item is None:
                return
            line, rid, cut0, wrap0 = item
            try:
                with self._sched:
                    cancelled = rid in self._rid_cut
                    self._rid_cut.discard(rid)
                if self.quit or cancelled:
                    if not self.quit:
                        self._emit(f"DONE 0 {self._prompt_len(line)} 0.0 0.0 cancel 0 0 0 0 0", rid)
                elif stream is None:
                    self._serve_one(line, slot, rid, cut0, wrap0)
                else:
                    with torch.npu.stream(stream):
                        self._serve_one(line, slot, rid, cut0, wrap0)
            finally:
                self._finish_slot(slot)

    def _serve_one(self, line: str, conv: Conversation, rid: int, cut0: int, wrap0: int) -> None:
        """One request: an exception is this request's error, never the engine's (the others go on)."""
        try:
            self.generate(line, conv=conv, rid=rid, cut0=cut0, wrap0=wrap0)
        except Exception as exc:                          # noqa: BLE001 - one request's failure
            _log("dsv41 parallel: request %s failed - %s: %s\n%s"
                 % (rid, type(exc).__name__, exc, traceback.format_exc()))
            conv.state = None
            conv.held = []
            self._emit(f"ERR {type(exc).__name__}: {exc}", rid)

    def _finish_slot(self, slot: Conversation) -> None:
        """Free the slot and start a queued request on it, if any (FIFO; the oldest waiter goes first)."""
        cancelled: list[tuple[int, int]] = []
        with self._sched:
            if slot.rid is not None:
                self._rid_cut.discard(slot.rid)       # a STOP that lost the race with the request's end
            slot.busy = False
            slot.rid = None
            while self._waiting and not self.quit:
                line, rid = self._waiting.popleft()
                if rid in self._rid_cut:
                    self._rid_cut.discard(rid)
                    cancelled.append((rid, self._prompt_len(line)))
                    continue
                self._assign(slot, line, rid, self._cut, self._wrap)
                break
        for rid, n in cancelled:
            self._emit(f"DONE 0 {n} 0.0 0.0 cancel 0 0 0 0 0", rid)

    def _new_stream(self):
        """This worker's own NPU stream (opt-in), or None: CPU tests, the default path, `--parallel` without
        `--parallel-streams`."""
        if not self._parallel_streams or self.parallel <= 1 or \
                not str(getattr(self.rt, "device", "cpu")).startswith("npu"):
            return None
        if not hasattr(torch, "npu"):
            return None
        try:
            return torch.npu.Stream()
        except Exception:                                 # noqa: BLE001 - streams are an optimisation
            _log("dsv41 parallel: could not create a per-conversation NPU stream; using the default one")
            return None

    # -- one request ---------------------------------------------------------------------------
    def generate(self, line: str, conv=None, rid: int | None = None,
                 cut0: int = 0, wrap0: int = 0) -> None:
        conv = self if conv is None else conv
        fields = line.split()
        max_new = int(fields[1])
        keys = {}
        for f in fields[2:-1]:
            if "=" in f:
                k, v = f.split("=", 1)
                keys[k] = v
        keys.pop("rid", None)
        ids = [int(x) for x in fields[-1].split(",") if x]
        if not ids:
            self._emit("ERR the request carries no tokens", rid)
            return
        prompt_tokens = len(ids)
        temperature = float(keys.get("temperature", 0.0) or 0.0)
        top_p = float(keys.get("top_p", 0.0) or 0.0)
        top_k = int(keys.get("top_k", 0) or 0)
        seed = keys.get("seed")
        think_budget = int(keys.get("think_budget", 0) or 0)
        think_end = int(keys.get("think_end", -1) or -1)
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        # how much of this prompt the held conversation already covers
        reused = 0
        for a, b in zip(ids, conv.held):
            if a != b:
                break
            reused += 1
        fresh = reused < len(conv.held) or conv.state is None
        if fresh:
            # the prompt diverged from what the state holds: start the conversation over.  The graphs
            # of the finished conversation bake its State's buffers and are of no use to the new one,
            # so they are released *before* a new State (and its KV) is built.
            reused = 0
            self._drop_conversation("the prompt diverged from the held conversation", conv)
            conv.state = self.rt.state(self.args.max_context)
            conv.held = []
        to_read = ids[reused:]

        t_prompt = time.perf_counter()
        if reused == len(ids):
            # the state already covers the whole prompt, but decoding needs the *last* token's logits:
            # step back one and read it again
            conv.state.rewind(1)
            conv.held.pop()
            to_read = [ids[-1]]
        logits = None
        if to_read:
            plan = prompt_chunks(len(to_read), self.args.prefill_chunk, self.args.max_context)
            if len(plan) > 1:
                # One PP line per forward, so a one-forward prompt reports progress only when it is done.
                # The plan is in the log at the start, which is what a long silent prompt needs.
                _log(f"dsv41 prompt: {len(to_read)} fresh tokens in {len(plan)} forward(s) of at most "
                     f"{max(plan)} (--prompt-chunk {self.args.prefill_chunk}, --max-context "
                     f"{self.args.max_context}); each forward re-reads the expert set, so fewer is faster")
            done = 0
            for size in plan:
                piece = to_read[done:done + size]
                done += len(piece)
                last = done >= len(to_read)
                got = self.rt.model.forward(piece, conv.state, logits_last_only=True, compute_logits=last)
                if last:
                    logits = got
                ms = (time.perf_counter() - t_prompt) * 1e3
                self._emit(f"PP {reused + done} {prompt_tokens} {ms:.1f} "
                           f"{done / max(ms / 1e3, 1e-9):.1f} {reused}", rid)
                control = self._control_for(rid, cut0, wrap0)
                if control == "QUIT":
                    self.quit = True
                    return
                if control in ("STOP", "PAUSE", "PAUSE keep"):
                    self._emit(f"DONE 0 {prompt_tokens} {ms:.1f} 0.0 cancel 0 0 {reused} 0 0", rid)
                    # The state is part-way through the prompt and `held` was never updated, so this
                    # conversation cannot be continued: it is abandoned, and its graphs (if it had
                    # any from an earlier turn) go with it.
                    _log("dsv41: the request was abandoned during its prompt - dropping the "
                         "conversation's state" + (" and its graphs" if self._graphs_held else ""))
                    self._drop_conversation("the request was abandoned during its prompt", conv)
                    return
        prompt_ms = (time.perf_counter() - t_prompt) * 1e3
        conv.held = list(ids)
        if max_new > 0 and conv is self:
            # Count the conversations that hold captured graphs.  The capture itself is lazy (the first
            # one-token step below); this is the bound's accounting and the report of what it will cost.
            self._plan_graphs()

        # --- decode
        produced: list[int] = []
        finish = "length"
        t0 = time.perf_counter()
        in_think = think_budget > 0 and think_end >= 0
        while len(produced) < max_new:
            control = self._control_for(rid, cut0, wrap0)
            if control == "QUIT":
                self.quit = True
                finish = "cancel"
                break
            if control in ("STOP", "PAUSE", "PAUSE keep"):
                finish = "cancel"
                break
            if in_think and (control == "WRAP" or len(produced) >= think_budget):
                tok = think_end
                in_think = False
            else:
                tok = self.sample(logits, temperature, top_p, top_k, gen)
                if in_think and tok == think_end:
                    in_think = False
            self._emit(f"T {tok}", rid)
            produced.append(tok)
            conv.held.append(tok)
            if tok == self.eos_id():
                finish = "stop"
                break
            logits = self.rt.model.forward([tok], conv.state, logits_last_only=True)
            n = len(produced)
            if n % 8 == 0:
                self.emit(self.stat_line(t0, n))
        decode_ms = (time.perf_counter() - t0) * 1e3
        if conv is self:
            self._report_graph_capture()
        with self._sched:
            self.stats["tokens"] += len(produced)
            self.stats["steps"] += 1
            self.stats["decode_ms"] += decode_ms
        self._emit(f"DONE {len(produced)} {prompt_tokens} {prompt_ms:.1f} {decode_ms:.1f} {finish} 0 0 "
                   f"{reused} {len(produced)} {len(produced)}", rid)
        _log(f"dsv41 stat: decode {decode_ms / max(1, len(produced)):.2f} ms/tok "
             f"({1000.0 * len(produced) / max(decode_ms, 1e-3):.2f} tok/s) | reused {reused}/{prompt_tokens} | "
             f"finish {finish}")

    def stat_line(self, t0: float, n: int) -> str:
        st = getattr(self.rt.experts, "stats", lambda: {})()
        ms = (time.perf_counter() - t0) * 1e3 / max(1, n)
        total = max(1, st.get("expert_hits", 0) + st.get("expert_misses", 0))
        return (f"STAT tok_s={1000.0 / max(ms, 1e-9):.2f} ms_tok={ms:.2f} "
                f"vram_hit={st.get('expert_hits', 0) / total:.4f} ram_fetch=0.0000 disk=0.0000 promo=0.000 "
                f"vram_used={st.get('expert_slots', 0)} vram_slots={getattr(self.rt.experts, 'cache_slots', 0)} "
                f"ram_used=0 ram_slots=0 ram_gb=0.0 "
                f"vram_gb={self.rt.weights.resident_bytes() / rt.GIB:.2f} disk_gb=0.0 draft_acc=0 draft_off=0")

    def eos_id(self) -> int:
        if self.tok is not None:
            got = self.tok.token_to_id("<｜end▁of▁sentence｜>")
            if got is not None:
                return int(got)
        return int(self.cfg.raw.get("eos_token_id", EOS))

    @staticmethod
    def sample(logits: torch.Tensor, temperature: float, top_p: float, top_k: int,
               gen: torch.Generator) -> int:
        """Greedy unless a temperature was asked for - the protocol's default (`temperature=0` is greedy)."""
        row = logits[0].float().cpu()
        if temperature <= 0:
            return int(torch.argmax(row))
        row = row / temperature
        if top_k > 0:
            kth = torch.topk(row, min(top_k, row.numel())).values[-1]
            row = torch.where(row < kth, torch.full_like(row, float("-inf")), row)
        probs = torch.softmax(row, -1)
        if 0 < top_p < 1:
            order = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(order.values, -1)
            cut = int((cumulative > top_p).nonzero()[0]) + 1 if bool((cumulative > top_p).any()) else order.values.numel()
            keep = order.indices[:cut]
            masked = torch.zeros_like(probs)
            masked[keep] = probs[keep]
            probs = masked / masked.sum()
        return int(torch.multinomial(probs, 1, generator=gen))

    # -- the loop ------------------------------------------------------------------------------
    def run(self) -> int:
        if self.parallel > 1:
            return self._run_multi()
        self.say_ready()
        threading.Thread(target=self._reader, daemon=True).start()
        while not self.quit:
            try:
                line = self.input.get(timeout=0.2)
            except queue.Empty:
                continue
            if line is None:
                break
            if line == "QUIT":
                break
            if line.startswith("GEN "):
                while True:                       # a stray control between requests
                    held = self.controls_held()
                    if held is None:
                        break
                    if held == "QUIT":
                        self.shutdown()
                        return 0
                self.generate(line)
            elif line.startswith("GENI "):
                self._emit("ERR this engine serves text only: no images")
            elif line in ("VLEND", "VRECLAIM"):
                self._emit(f"ERR {line} text-only model: no vision encoder to lend memory to")
            else:
                self._emit(f"ERR unsupported command '{line.split()[0]}'")
        self.shutdown()
        return 0

    def _run_multi(self) -> int:
        """The `--parallel` loop: admit GEN lines onto conversation slots, apply controls, never block a
        request on another's device sync."""
        self.say_ready()
        self.start_workers()
        threading.Thread(target=self._reader, daemon=True).start()
        while not self.quit:
            while True:                           # controls first: STOP/PAUSE/WRAP/QUIT are not queued behind GENs
                held = self.controls_held()
                if held is None:
                    break
                self._apply_control(held)
            try:
                line = self.input.get(timeout=0.05)
            except queue.Empty:
                continue
            if line is None:
                break
            if line == "QUIT":
                self.quit = True
                break
            self._dispatch(line)
        self.stop_workers()
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        """Close the runtime; a graph that could not be destroyed is named, never raised.

        `rt.close()` tears the graphs down first and `Dsv41.release_graphs` raises after dropping the
        domain if the destroy failed.  A leak is a real loss of HBM, but it must not turn a clean
        shutdown into a failed one: the log names it and the process exits, which is when the leaked
        models finally go.  Idempotent: a benchmark may stop the workers and then close the engine.
        """
        if self._closed:
            return
        self._closed = True
        close = getattr(self.rt, "close", None)
        if close is None:                     # a test double: nothing to close
            return
        try:
            close()
        except GraphTeardownError as exc:
            _log(f"dsv41 graphs: teardown at shutdown failed ({exc}); the leaked graph models end with "
                 f"this process")


def build_parser() -> argparse.ArgumentParser:
    """The engine's arguments; separate from `main` so tests can check the flags without loading a model."""
    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash W8A8 engine on Ascend, serve protocol")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16", choices=list(rt.DTYPES))
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--engram-tokenizer", default="",
                    help="the model's Hugging Face tokenizer.json: enables Engram (it hashes n-grams "
                         "through the tokenizer's compressed map).  Off when not given.")
    ap.add_argument("--expert-devices", default="",
                    help="comma-separated cards to spread the routed experts' tier over, e.g. "
                         "npu:0,npu:1,npu:2,npu:3; the layers stay on --device")
    ap.add_argument("--expert-reserve-gib", type=float, default=6.0,
                    help="HBM kept free for activations and temporaries; the tier gets the rest")
    ap.add_argument("--io-threads", type=int, default=6,
                    help="threads the expert fetch reads the checkpoint with (one preadv per thread per "
                         "expert projection).  The array knee on this machine is 12-24: 1.4 GB/s at 1 "
                         "thread, 2.3 at 6, 2.5 at 12 (docs/ASCEND-DSV41.md §1); a prefill is "
                         "bandwidth-bound, so 12 is the measured-better value for a long prompt")
    ap.add_argument("--expert-batch", type=int, default=8,
                    help="experts fetched per MoE batch; sizes the pinned staging ring "
                         "(2 x 3 x batch x 11.8 MiB) and the per-layer batching.  Also the model's "
                         "read batch: a prefill reads ~318 experts a layer, so this is how many large "
                         "preads each fetch issues (see docs/ASCEND-DSV41.md §15 for the measured effect)")
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--parallel-streams", action="store_true",
                    help="run each --parallel conversation on its own NPU stream.  Off by default: measured "
                         "on one 910B3, four streams were ~10-15%% slower than the device's default stream "
                         "at the same tier (the shared staging ring's per-slot events then couple the "
                         "streams).  Host dispatch still overlaps across the request threads either way")
    ap.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="conversations served at once (default 1, the pre-existing behaviour).  With N > 1 "
                         "READY says multi=N, serve/server.py gives its FIFO N places and tags every request "
                         "with rid=<id>; each conversation gets its own State (and, with --parallel-streams, "
                         "its own NPU stream).  This is NOT batching: forward takes one State (KvPlanes "
                         "refuses batch > 1), so the gain is host dispatch overlap across the request "
                         "threads.  Runs eagerly - a captured decode is one State - "
                         "so --graphs needs --parallel 1.  Size --expert-reserve-gib for N concurrent "
                         "activations; a request past N waits in the engine's queue")
    ap.add_argument("--prompt-chunk", "--prefill-chunk", dest="prefill_chunk", type=int, default=512,
                    metavar="N",
                    help="the most tokens one prompt forward reads (default 512; 0 = the whole prompt).  "
                         "Each extra forward re-reads almost the whole expert set from the storage "
                         "array, so this is the largest prefill lever: measured on a 2 000-token prompt, "
                         "4 x 512 costs ~4x one forward in TTFT.  Clamped to --max-context (a longer "
                         "forward would overwrite its own KV rows); docs/ASCEND-DSV41.md §15")
    ap.add_argument("--graphs", action="store_true",
                    help="capture the one-token decode path as per-layer NPU graphs, as --graphs in "
                         "ascend/run_dsv41.py: a g_pre/g_post pair per layer, one g_pre per indexer "
                         "selection bucket, the expert fetch between them on the host.  Off by default; "
                         "prompt chunks (T > 1) stay eager.  The graphs bake the conversation State's "
                         "buffers, so they are captured per conversation (1.8-1.9 s for 80 graphs) and "
                         "released when it ends")
    ap.add_argument("--graph-verify", action="store_true",
                    help="run the did-it-capture guard on every graph (slow: 4 eager runs per graph at "
                         "start-up, then bit-equality at two input values); only meaningful with "
                         "--graphs")
    ap.add_argument("--graph-max-conversations", type=int, default=1,
                    help="the most conversations whose captured graphs may be alive at once.  Each "
                         "conversation costs a 425 MB landing zone on every card that holds experts plus "
                         "that card's graph pool and registers, and --graphs reserves 1 GiB per card for "
                         "one; past this bound a conversation runs eagerly and says so in the log")
    ap.add_argument("--verbose", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    serve = "--serve" in argv
    ap = build_parser()
    args = ap.parse_args(argv)
    if not serve:
        ap.error("this module is an engine: start it with --serve (see serve/server.py --config)")
    engine = Engine(args)
    try:
        return engine.run()
    except BrokenPipeError:
        engine.shutdown()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
