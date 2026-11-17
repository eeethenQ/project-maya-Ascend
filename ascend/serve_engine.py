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

EOS = 1


def _log(msg: str) -> None:
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


def _out(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


class Engine:
    """One conversation, held across requests: the state a reply left is where the next prompt starts."""

    def __init__(self, args):
        self.args = args
        t0 = time.time()
        self.rt = rt.build(model=args.model, device=args.device, dtype=args.dtype,
                           hbm_share=args.hbm_share, expert_slots=args.expert_slots,
                           io_threads=args.io_threads, verbose=args.verbose)
        self.cfg = self.rt.cfg
        self.load_s = time.time() - t0
        self.state = None
        self.held: list[int] = []                 # the tokens the state covers
        self.control: queue.Queue[str] = queue.Queue()
        self.input: queue.Queue[str | None] = queue.Queue()
        self.quit = False
        self.stats = {"tokens": 0, "steps": 0, "decode_ms": 0.0}
        if args.tokenizer:
            from tokenizers import Tokenizer
            self.tok = Tokenizer.from_file(args.tokenizer)
        else:
            self.tok = None

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
        _log(f"maya generate: session is up (ascend dsv41, {self.rt.device}, weights in {self.load_s:.1f}s)")
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
            # the prompt diverged from what the state holds: start the conversation over
            reused = 0
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
            chunk = max(1, self.args.prefill_chunk)
            done = 0
            while done < len(to_read):
                piece = to_read[done:done + chunk]
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
                    return
        prompt_ms = (time.perf_counter() - t_prompt) * 1e3
        self.held = list(ids)

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
        return EOS

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
                        return 0
                self.generate(line)
            elif line.startswith("GENI "):
                _out("ERR this engine serves text only: no images")
            elif line in ("VLEND", "VRECLAIM"):
                _out(f"ERR {line} text-only model: no vision encoder to lend memory to")
            else:
                _out(f"ERR unsupported command '{line.split()[0]}'")
        self.rt.close()
        return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    serve = "--serve" in argv
    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash W8A8 engine on Ascend, serve protocol")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", default="")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--dtype", default="bfloat16", choices=list(rt.DTYPES))
    ap.add_argument("--hbm-share", type=float, default=0.5)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--io-threads", type=int, default=6)
    ap.add_argument("--max-context", type=int, default=8192)
    ap.add_argument("--prefill-chunk", type=int, default=512)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    if not serve:
        ap.error("this module is an engine: start it with --serve (see serve/server.py --config)")
    engine = Engine(args)
    try:
        return engine.run()
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
