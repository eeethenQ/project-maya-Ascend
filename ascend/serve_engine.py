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
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time

import torch

from .dsv41 import runtime as rt
from .dsv41.graph_runner import GraphTeardownError

EOS = 1


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


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
        t0 = time.time()
        self.rt = rt.build(model=args.model, device=args.device, dtype=args.dtype,
                           hbm_share=args.hbm_share, expert_slots=args.expert_slots,
                           io_threads=args.io_threads, expert_batch=args.expert_batch,
                           expert_reserve_gib=args.expert_reserve_gib,
                           expert_devices=args.expert_devices.split(",") if args.expert_devices else None,
                           engram_tokenizer=args.engram_tokenizer or None,
                           graphs=getattr(args, "graphs", False),
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
        # -- the captured decode, per conversation ----------------------------------------------
        # The graphs bake the held State's buffers, so there is nothing to reuse once the conversation
        # is over.  `_graphs_held` is the conversation being served; `_graphs_leaked` counts domains a
        # failed teardown left alive (their HBM never comes back in this process), and the two together
        # are what the bound counts.  A leaked domain must never turn into a second landing zone on
        # every expert card by accident, which is what the bound is for.
        self._graphs_wanted = bool(getattr(args, "graphs", False))
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
             f"graphs {'on' if self._graphs_wanted else 'off'})")
        if self._graphs_wanted:
            _log(f"dsv41 graphs: capturing the one-token decode per conversation (g_pre/g_post per layer, "
                 f"one g_pre per indexer bucket; measured 1.8-1.9 s for 80 graphs).  At most "
                 f"{self._graph_max} captured conversation(s) alive at once; past that bound a "
                 f"conversation runs eagerly and says so.  Prompt chunks stay eager.")
        _out(f"INFO context={ctx} kv=bf16 kv_resident=0 expert_slots={slots} "
             f"expert_cache_mib={slots * per_expert // (1 << 20)} engine_kind=dsv41-ascend "
             f"experts={experts_total} vram_slots={slots} vram_gb={resident + slots * per_expert / rt.GIB:.2f} "
             f"ram_slots=0 ram_gb=0.0 mtp=0 dflash=0 vision_lend=0 ling=0 cpu_threads=0 cpu_threads_max=0 "
             f"pcie_share={self.args.hbm_share:.2f} logit_bias=1")
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

    def _drop_conversation(self, reason: str) -> None:
        """Forget the held conversation, releasing its graphs (its State is garbage after this)."""
        self._release_graphs(reason)
        self.state = None
        self.held = []

    # -- reading the input ---------------------------------------------------------------------
    def _reader(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if line in ("STOP", "WRAP", "PAUSE", "PAUSE keep"):
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

    # -- one request ---------------------------------------------------------------------------
    def generate(self, line: str) -> None:
        fields = line.split()
        max_new = int(fields[1])
        keys = {}
        for f in fields[2:-1]:
            if "=" in f:
                k, v = f.split("=", 1)
                keys[k] = v
        ids = [int(x) for x in fields[-1].split(",") if x]
        if not ids:
            _out("ERR the request carries no tokens")
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
        for a, b in zip(ids, self.held):
            if a != b:
                break
            reused += 1
        fresh = reused < len(self.held) or self.state is None
        if fresh:
            # the prompt diverged from what the state holds: start the conversation over.  The graphs
            # of the finished conversation bake its State's buffers and are of no use to the new one,
            # so they are released *before* a new State (and its KV) is built.
            reused = 0
            self._drop_conversation("the prompt diverged from the held conversation")
            self.state = self.rt.state(self.args.max_context)
            self.held = []
        to_read = ids[reused:]

        t_prompt = time.perf_counter()
        if reused == len(ids):
            # the state already covers the whole prompt, but decoding needs the *last* token's logits:
            # step back one and read it again
            self.state.rewind(1)
            self.held.pop()
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
                got = self.rt.model.forward(piece, self.state, logits_last_only=True, compute_logits=last)
                if last:
                    logits = got
                ms = (time.perf_counter() - t_prompt) * 1e3
                _out(f"PP {reused + done} {prompt_tokens} {ms:.1f} {done / max(ms / 1e3, 1e-9):.1f}")
                control = self.controls_held()
                if control == "QUIT":
                    self.quit = True
                    return
                if control in ("STOP", "PAUSE", "PAUSE keep"):
                    _out(f"DONE 0 {prompt_tokens} {ms:.1f} 0.0 cancel 0 0 {reused} 0 0")
                    # The state is part-way through the prompt and `held` was never updated, so this
                    # conversation cannot be continued: it is abandoned, and its graphs (if it had
                    # any from an earlier turn) go with it.
                    _log("dsv41: the request was abandoned during its prompt - dropping the "
                         "conversation's state" + (" and its graphs" if self._graphs_held else ""))
                    self._drop_conversation("the request was abandoned during its prompt")
                    return
        prompt_ms = (time.perf_counter() - t_prompt) * 1e3
        self.held = list(ids)
        if max_new > 0:
            # Count the conversations that hold captured graphs.  The capture itself is lazy (the first
            # one-token step below); this is the bound's accounting and the report of what it will cost.
            self._plan_graphs()

        # --- decode
        produced: list[int] = []
        finish = "length"
        t0 = time.perf_counter()
        in_think = think_budget > 0 and think_end >= 0
        while len(produced) < max_new:
            control = self.controls_held()
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
            _out(f"T {tok}")
            produced.append(tok)
            self.held.append(tok)
            if tok == self.eos_id():
                finish = "stop"
                break
            logits = self.rt.model.forward([tok], self.state, logits_last_only=True)
            n = len(produced)
            if n % 8 == 0:
                _out(self.stat_line(t0, n))
        decode_ms = (time.perf_counter() - t0) * 1e3
        self._report_graph_capture()
        self.stats["tokens"] += len(produced)
        self.stats["steps"] += 1
        self.stats["decode_ms"] += decode_ms
        _out(f"DONE {len(produced)} {prompt_tokens} {prompt_ms:.1f} {decode_ms:.1f} {finish} 0 0 {reused} "
             f"{len(produced)} {len(produced)}")
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
                _out("ERR this engine serves text only: no images")
            elif line in ("VLEND", "VRECLAIM"):
                _out(f"ERR {line} text-only model: no vision encoder to lend memory to")
            else:
                _out(f"ERR unsupported command '{line.split()[0]}'")
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        """Close the runtime; a graph that could not be destroyed is named, never raised.

        `rt.close()` tears the graphs down first and `Dsv41.release_graphs` raises after dropping the
        domain if the destroy failed.  A leak is a real loss of HBM, but it must not turn a clean
        shutdown into a failed one: the log names it and the process exits, which is when the leaked
        models finally go.
        """
        try:
            self.rt.close()
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
