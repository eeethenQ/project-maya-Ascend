"""ascend/tests/test_serve_engine.py - the Ascend engine against the protocol `serve/server.py` drives.

`serve/test_engine_contract.py` is that contract, but it needs the real model and a card.  This drives the
same protocol with the engine started the way the server starts it - `[exe, "--serve", ...]`, lines over
stdin/stdout - on a four-layer model on the host, so the parts that are easy to get wrong are tested in CI:
READY and the INFO keys the dashboard reads, one PP per prompt chunk, one T per token, DONE's eleven
fields, the conversation the engine keeps between requests (`reused`), STOP cutting a decode, and QUIT
ending with exit code 0.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from ascend.tests.synthetic import TinyModel

try:                                                    # the engine loads a model, which needs torch
    import torch  # noqa: F401
    HAVE_TORCH = True
except Exception:                                       # noqa: BLE001
    HAVE_TORCH = False

NEEDS_TORCH = unittest.skipUnless(HAVE_TORCH, "needs torch")

ROOT = Path(__file__).resolve().parents[2]
ENGINE = ROOT / "ascend" / "strata-ascend"
GIB = 1 << 30


def engine_args(model_dir: Path, **overrides) -> argparse.Namespace:
    """The arguments `main` would hand `Engine`, for tests that drive the engine in this process."""
    args = argparse.Namespace(
        model=str(model_dir), tokenizer="", device="cpu", dtype="float32", hbm_share=0.5,
        expert_slots=None, io_threads=1, expert_batch=8, expert_reserve_gib=6.0, expert_devices="",
        engram_tokenizer="", verbose=False, max_context=64, prefill_chunk=512, serve=True,
        graphs=True, graph_verify=False, graph_max_conversations=None, parallel=1)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def quiet_generate(engine, line: str) -> tuple[str, str]:
    """Run one request, returning (stdout, stderr); the engine writes both directly."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        engine.generate(line)
    return out.getvalue(), err.getvalue()


class Engine:
    """The engine as the server sees it: a subprocess and a line queue."""

    def __init__(self, model_dir: Path, *extra: str):
        self.proc = subprocess.Popen(
            [str(ENGINE), "--serve", "--model", str(model_dir), "--device", "cpu", "--dtype", "float32",
             "--max-context", "64", *extra],
            cwd=str(ROOT), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.log: list[str] = []
        threading.Thread(target=self._pump, daemon=True).start()
        threading.Thread(target=self._pump_log, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def _pump_log(self):
        for line in self.proc.stderr:
            self.log.append(line.rstrip("\n"))

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read_until(self, prefix: str, timeout: float = 120.0) -> list[str]:
        got, end = [], time.time() + timeout
        while time.time() < end:
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                raise AssertionError(f"the engine ended while waiting for {prefix!r}; log:\n"
                                     + "\n".join(self.log[-20:]))
            got.append(line)
            if line.startswith(prefix):
                return got
        raise AssertionError(f"no {prefix!r} within {timeout}s; saw {got[-8:]}")

    def stop(self) -> int:
        try:
            self.send("QUIT")
        except (BrokenPipeError, ValueError):
            pass
        try:
            return self.proc.wait(timeout=60)
        finally:
            for s in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
                try:
                    s.close()
                except Exception:                       # noqa: BLE001
                    pass


def kv(line: str) -> dict:
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)


@NEEDS_TORCH
class TheProtocol(unittest.TestCase):
    """One engine for the class: the weights load once, as in the server's resident engine."""

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()
        cls.engine = Engine(cls.model.root)
        cls.start = cls.engine.read_until("READY")

    @classmethod
    def tearDownClass(cls):
        cls.engine.stop()
        cls.model.close()

    def test_ready_after_info(self):
        info = [l for l in self.start if l.startswith("INFO ")]
        ready = [l for l in self.start if l.startswith("READY ")]
        self.assertEqual(len(ready), 1)
        self.assertTrue(info, "the server reads INFO for the dashboard")
        self.assertEqual(ready[0].split()[1], "64", "READY carries the context")
        self.assertIn("stop", ready[0].split()[2:], "the engine honours STOP mid-request")

    def test_the_info_keys_the_dashboard_reads(self):
        info = kv([l for l in self.start if l.startswith("INFO ")][0])
        for key in ("context", "kv", "expert_slots", "expert_cache_mib", "engine_kind", "experts",
                    "vram_slots", "vram_gb", "ram_slots", "ram_gb", "mtp", "vision_lend", "cpu_threads",
                    "pcie_share"):
            self.assertIn(key, info, f"INFO is missing {key}")
        self.assertEqual(info["engine_kind"], "dsv41-ascend")

    def test_the_start_log_carries_the_context_cost_line(self):
        """serve/server.py's _context_cost matches this exact shape on stderr."""
        import re
        dead = time.time() + 20
        while time.time() < dead:
            hit = [l for l in self.engine.log
                   if re.search(r"CUDA0 VRAM before the expert pool:.*?state/KV \((\d+) ctx\) ([\d.]+)", l)]
            if hit:
                return
            time.sleep(0.05)
        self.fail("no context-cost line on stderr; log:\n" + "\n".join(self.engine.log[-10:]))

    def test_a_generation_streams_pp_t_then_done(self):
        self.engine.send("GEN 3 3,5,7")
        lines = self.engine.read_until("DONE ")
        pp = [l for l in lines if l.startswith("PP ")]
        ts = [l for l in lines if l.startswith("T ")]
        self.assertTrue(pp, "a prompt reports progress")
        self.assertEqual(len([l for l in ts if l.split()[1].isdigit()]), 3, "three tokens asked, three sent")
        done = [l for l in lines if l.startswith("DONE ")][0].split()
        self.assertEqual(len(done), 11, f"DONE has eleven fields: {done}")
        self.assertEqual(int(done[1]), 3, "generated")
        self.assertEqual(int(done[2]), 3, "prompt tokens")
        self.assertIn(done[5], ("length", "stop", "cancel"))
        self.assertEqual(int(done[8]), 0, "nothing of this prompt was held before")
        for i, t in enumerate([l for l in ts if l.split()[1].isdigit()]):
            self.assertLess(int(t.split()[1]), 32, "a token id inside the tiny vocabulary")

    def test_the_conversation_is_kept_between_requests(self):
        """The next GEN extends the held conversation, and DONE says how much of it the engine had."""
        self.engine.send("GEN 1 3,5,7")
        first = self.engine.read_until("DONE ")
        produced = [int(l.split()[1]) for l in first if l.startswith("T ") and l.split()[1].isdigit()]
        longer = [3, 5, 7] + produced + [11]
        self.engine.send("GEN 1 " + ",".join(str(i) for i in longer))
        second = self.engine.read_until("DONE ")
        reused = int([l for l in second if l.startswith("DONE ")][0].split()[8])
        self.assertEqual(reused, len(longer) - 1, "everything but the token whose logits are needed again")

    def test_stop_cuts_a_decode(self):
        self.engine.send("GEN 500 3,5,7")
        self.engine.read_until("T ")
        self.engine.send("STOP")
        lines = self.engine.read_until("DONE ")
        done = [l for l in lines if l.startswith("DONE ")][0].split()
        self.assertEqual(done[5], "cancel")
        self.assertLess(int(done[1]), 500)

    def test_a_stat_line_reports_the_expert_tier(self):
        self.engine.send("GEN 9 3,5,7")
        lines = self.engine.read_until("DONE ")
        stats = [l for l in lines if l.startswith("STAT ")]
        self.assertTrue(stats, "the dashboard reads STAT every 8 tokens")
        fields = kv(stats[0])
        for key in ("tok_s", "ms_tok", "vram_hit", "vram_used", "vram_slots", "vram_gb"):
            self.assertIn(key, fields)

    def test_images_and_vision_are_refused_in_words(self):
        self.engine.send("VLEND")
        line = self.engine.read_until("ERR ")[-1]
        self.assertIn("text-only", line)


@NEEDS_TORCH
class Quit(unittest.TestCase):
    def test_quit_ends_the_engine_with_exit_code_zero(self):
        with TinyModel() as model:
            engine = Engine(model.root)
            engine.read_until("READY")
            self.assertEqual(engine.stop(), 0)

    def test_graphs_over_the_line_protocol(self):
        """`--graphs` starts, captures, and quits cleanly over the real executable's protocol."""
        with TinyModel() as model:
            engine = Engine(model.root, "--graphs")
            engine.read_until("READY")
            engine.send("GEN 2 3,5,7")
            lines = engine.read_until("DONE ")
            self.assertEqual(len([l for l in lines if l.startswith("T ")]), 2)
            self.assertEqual(engine.stop(), 0)
            log = "\n".join(engine.log)
            self.assertIn("graphs on", log, "the start log names the captured path")
            self.assertIn("graphs captured", log, "the engine reports what the capture cost")


@NEEDS_TORCH
class TheGraphsArguments(unittest.TestCase):
    """The flag exists, graphs are off unless asked for, and the bound has a default."""

    @staticmethod
    def parser():
        from ascend import serve_engine
        return serve_engine.build_parser()

    def test_graphs_are_off_by_default_and_the_bound_is_the_engine_s(self):
        args = self.parser().parse_args(["--serve", "--model", "/tmp/model"])
        self.assertFalse(args.graphs, "the captured decode is opt-in, as in run_dsv41")
        self.assertFalse(args.graph_verify)
        self.assertIsNone(args.graph_max_conversations,
                          "unset means the engine's own bound (DSV41_GRAPH_MAX_CONVERSATIONS, 4)")

    def test_the_graph_flags_parse(self):
        args = self.parser().parse_args(
            ["--serve", "--model", "/tmp/model", "--graphs", "--graph-verify",
             "--graph-max-conversations", "3"])
        self.assertTrue(args.graphs)
        self.assertTrue(args.graph_verify)
        self.assertEqual(args.graph_max_conversations, 3)


@NEEDS_TORCH
class TheGraphsPlumbing(unittest.TestCase):
    def test_the_flags_reach_runtime_build_and_the_bound_is_capped_not_raised(self):
        from types import SimpleNamespace
        from ascend import serve_engine

        seen: dict = {}
        real = serve_engine.rt.build

        def fake(**kwargs):
            seen.update(kwargs)
            model = SimpleNamespace(cfg=SimpleNamespace(num_hidden_layers=4),
                                    graph_max_conversations=4)
            return SimpleNamespace(cfg=model.cfg, model=model)

        try:
            serve_engine.rt.build = fake
            engine = serve_engine.Engine(engine_args(Path("/tmp/not-loaded"), graphs=True,
                                                     graph_verify=True, graph_max_conversations=2))
        finally:
            serve_engine.rt.build = real
        self.assertIs(seen["graphs"], True)
        self.assertIs(seen["graph_verify"], True)
        self.assertEqual(engine._graph_max, 2)
        self.assertEqual(engine.rt.model.graph_max_conversations, 2,
                         "the model and the serve policy enforce the same number")
        try:
            serve_engine.rt.build = fake
            high = serve_engine.Engine(engine_args(Path("/tmp/not-loaded"), graphs=True,
                                                   graph_max_conversations=8))
        finally:
            serve_engine.rt.build = real
        self.assertEqual(high._graph_max, 4, "--graph-max-conversations can only lower the engine's bound")
        self.assertEqual(high.rt.model.graph_max_conversations, 4)


@NEEDS_TORCH
class TheConversationGraphLifecycle(unittest.TestCase):
    """A graph bakes a State's buffers: capture per conversation, release when it ends, never leak.

    The CPU capture wrapper is the eager fallback, but the *domain* (`_DecodeGraphs`), its lifecycle and
    the engine's accounting are the real ones, which is what this class exercises.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def engine(self, **overrides):
        from ascend import serve_engine
        return serve_engine.Engine(engine_args(self.model.root, **overrides))

    def test_a_rewind_to_a_shared_prefix_keeps_the_captured_graphs(self):
        """Prefix reuse is a rewind of the same State, so the graph domain (and its capture) survives it.

        The old path dropped and reallocated the State on a diverging prompt, which released the graphs and
        made the new conversation pay the 1.8-1.9 s capture again.  A rewind keeps the State, so it keeps
        the graphs.
        """
        engine = self.engine()
        try:
            quiet_generate(engine, "GEN 2 3,5,7,11,13,17")
            graphs = engine.rt.model._decode_graphs
            self.assertIsNotNone(graphs)
            out, err = quiet_generate(engine, "GEN 1 3,5,7,19,23")   # shares 3 tokens, then diverges
            self.assertIn("dsv41 prefix: the prompt shares 3", err)
            self.assertIs(engine.rt.model._decode_graphs, graphs,
                          "the State was rewound, not replaced: the captured domain is still valid")
            self.assertNotIn("released", err, "no teardown on a prefix rewind")
            self.assertEqual(int([l for l in out.splitlines() if l.startswith("DONE ")][0].split()[8]), 3)
        finally:
            engine.shutdown()

    def test_a_conversation_captures_on_its_first_decode_and_keeps_the_graphs(self):
        engine = self.engine()
        try:
            out, err = quiet_generate(engine, "GEN 2 3,5,7")
            self.assertIn("DONE 2 3", out)
            graphs = engine.rt.model._decode_graphs
            self.assertIsNotNone(graphs)
            self.assertEqual(graphs.captured, 2 * engine.cfg.num_hidden_layers,
                             "one g_pre and one g_post per layer")
            self.assertIn("capture its decode", err)
            self.assertIn("graphs captured", err, "the capture cost is reported in the log")
            self.assertEqual(engine.graphs_live(), 1)
            # the next turn of the same conversation reuses the captured domain (that is the point)
            held = list(engine.held)
            quiet_generate(engine, "GEN 1 " + ",".join(str(t) for t in held + [11]))
            self.assertIs(engine.rt.model._decode_graphs, graphs,
                          "a continuing conversation must not recapture")
        finally:
            engine.shutdown()

    def test_a_finished_conversation_releases_its_graphs(self):
        engine = self.engine()
        try:
            quiet_generate(engine, "GEN 2 3,5,7")
            finished = engine.rt.model._decode_graphs
            self.assertIsNotNone(finished)
            out, err = quiet_generate(engine, "GEN 1 11,13")      # a different prompt: new conversation
            self.assertIn("DONE 1 2", out)
            self.assertTrue(finished.closed, "the finished conversation's graphs are torn down")
            self.assertIsNot(engine.rt.model._decode_graphs, finished)
            self.assertIsNotNone(engine.rt.model._decode_graphs, "the new conversation captures its own")
            self.assertEqual(engine.graphs_live(), 1, "old released, new held")
            self.assertIn("released", err)
        finally:
            engine.shutdown()

    def test_an_abandoned_prompt_drops_the_conversation_and_its_graphs(self):
        engine = self.engine(prefill_chunk=1)
        try:
            quiet_generate(engine, "GEN 2 3,5,7")
            finished = engine.rt.model._decode_graphs
            self.assertIsNotNone(finished)
            engine.control.put("STOP")              # the next request is cut part-way through its prompt
            out, err = quiet_generate(engine, "GEN 5 11,13,17")
            self.assertIn("DONE 0 3", out)
            self.assertIn("cancel", out)
            self.assertTrue(finished.closed)
            self.assertIsNone(engine.state, "a half-read prompt cannot be the next turn's state")
            self.assertEqual(engine.held, [])
            self.assertIsNone(engine.rt.model._decode_graphs)
            self.assertIn("abandoned during its prompt", err)
            # and the engine serves the next request, capturing a fresh domain
            out, _ = quiet_generate(engine, "GEN 1 11,13")
            self.assertIn("DONE 1 2", out)
            self.assertIsNotNone(engine.rt.model._decode_graphs)
        finally:
            engine.shutdown()

    def test_graphs_off_by_default_captures_nothing(self):
        engine = self.engine(graphs=False)
        try:
            for line in ("GEN 2 3,5,7", "GEN 1 11,13"):
                out, _ = quiet_generate(engine, line)
                self.assertIn("DONE ", out)
            self.assertFalse(engine.rt.model.graphs)
            self.assertIsNone(engine.rt.model._decode_graphs)
        finally:
            engine.shutdown()

    def test_a_teardown_failure_is_counted_and_logged_without_retiring_the_others(self):
        from ascend.dsv41.graph_runner import GraphTeardownError

        engine = self.engine(graph_max_conversations=2)
        try:
            quiet_generate(engine, "GEN 2 3,5,7")
            model = engine.rt.model
            leaked = model._decode_graphs
            self.assertIsNotNone(leaked)

            def failing():
                # what a destroy failure is: the domain is already out of the registry, its HBM is not
                # returned, and it can never be freed again
                raise GraphTeardownError("npu:0", [("g_pre.0.0", "rtModelDestroy failed")])

            leaked.close = failing
            out, err = quiet_generate(engine, "GEN 1 11,13")
            self.assertIn("DONE 1 2", out, "one conversation's teardown failure must not kill the engine")
            self.assertIn("could not release", err)
            self.assertNotIn("Traceback", err)
            self.assertEqual(int(model._graph_leaked), 1,
                             "the leaked domain is counted against the bound")
            self.assertEqual(engine.graphs_live(), 2, "one live domain plus the leak")
            self.assertTrue(model.graphs,
                            "a leak must not disable the captured path for every other conversation")
            # one slot of the bound is spent on the leak; the next conversation still captures in the other
            fresh = model._decode_graphs
            self.assertIsNotNone(fresh, "the other bound slot still captures")
            self.assertIsNot(fresh, leaked)
            self.assertFalse(leaked.closed, "the leaked domain could not be closed")
        finally:
            # the leaked domain is already out of the registry (and counted); shutdown must not raise
            engine.shutdown()

    def test_shutdown_names_a_failed_teardown_instead_of_raising(self):
        from ascend.dsv41.graph_runner import GraphTeardownError

        engine = self.engine()

        def failing():
            raise GraphTeardownError("npu:0", [("g_post.0", "destroy failed")])

        engine.rt.model.release_graphs = failing
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            engine.shutdown()
        self.assertIn("teardown at shutdown failed", err.getvalue())


@NEEDS_TORCH
class ThePromptChunkArguments(unittest.TestCase):
    """`--prompt-chunk` (with `--prefill-chunk` kept as the old name) and what bounds one forward."""

    @staticmethod
    def parser():
        from ascend import serve_engine
        return serve_engine.build_parser()

    def test_the_default_is_the_measured_one_forward_value_and_both_names_parse(self):
        base = self.parser().parse_args(["--serve", "--model", "/tmp/model"])
        self.assertEqual(base.prefill_chunk, 2048,
                         "2048 is one forward for a typical prompt and -47% TTFT against 4 x 512")
        new = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--prompt-chunk", "4096"])
        old = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--prefill-chunk", "1024"])
        self.assertEqual(new.prefill_chunk, 4096)
        self.assertEqual(old.prefill_chunk, 1024, "--prefill-chunk keeps working")
        zero = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--prompt-chunk", "0"])
        self.assertEqual(zero.prefill_chunk, 0, "0 is the one-forward case")

    def test_the_plan_is_one_forward_per_chunk_and_the_whole_prompt_at_zero(self):
        from ascend.serve_engine import prompt_chunks

        self.assertEqual(prompt_chunks(2000, 512, 8192), [512, 512, 512, 464])
        self.assertEqual(prompt_chunks(2000, 1024, 8192), [1024, 976])
        self.assertEqual(prompt_chunks(2000, 2048, 8192), [2000])
        self.assertEqual(prompt_chunks(2000, 0, 8192), [2000])
        self.assertEqual(prompt_chunks(0, 512, 8192), [])
        self.assertEqual(prompt_chunks(1, 512, 8192), [1])

    def test_a_chunk_is_never_longer_than_the_kv_ring(self):
        """A forward longer than --max-context would overwrite its own rows in the planes' ring."""
        from ascend.serve_engine import prompt_chunks

        self.assertEqual(prompt_chunks(5000, 8192, 2048), [2048, 2048, 904])
        self.assertEqual(prompt_chunks(10, 512, 4), [4, 4, 2])
        self.assertEqual(prompt_chunks(10, 0, 4), [4, 4, 2], "0 is clamped by the ring too")

    def test_a_prompt_reports_one_pp_per_forward(self):
        """The chunk flag really changes the number of eager forwards the serve path runs."""
        from ascend import serve_engine

        with TinyModel() as model:
            for chunk, forwards in ((512, 1), (2, 3), (0, 1)):
                engine = serve_engine.Engine(engine_args(model.root, prefill_chunk=chunk, graphs=False))
                try:
                    out, err = quiet_generate(engine, "GEN 1 3,5,7,11,13,17")
                finally:
                    engine.shutdown()
                pp = [l for l in out.splitlines() if l.startswith("PP ")]
                self.assertEqual(len(pp), forwards,
                                 f"--prompt-chunk {chunk} must run {forwards} forward(s), saw {pp}")
                if forwards > 1:
                    self.assertIn("fresh tokens in 3 forward(s)", err)
                self.assertIn("DONE 1 6", out, "the prompt's last token was still decoded")


@NEEDS_TORCH
class TheStreamingFlags(unittest.TestCase):
    """O7: `--io-threads` and `--expert-batch` reach the streamer that reads the array, and do what they say."""

    @staticmethod
    def parser():
        from ascend import serve_engine
        return serve_engine.build_parser()

    def test_they_are_on_the_serve_cli_with_the_measured_defaults(self):
        args = self.parser().parse_args(["--serve", "--model", "/tmp/model"])
        self.assertEqual(args.io_threads, 6, "the engine's historical default")
        self.assertEqual(args.expert_batch, 8)
        chosen = self.parser().parse_args(
            ["--serve", "--model", "/tmp/model", "--io-threads", "12", "--expert-batch", "32"])
        self.assertEqual(chosen.io_threads, 12)
        self.assertEqual(chosen.expert_batch, 32)

    def test_the_flags_reach_runtime_build(self):
        from types import SimpleNamespace
        from ascend import serve_engine

        seen: dict = {}
        real = serve_engine.rt.build

        def fake(**kwargs):
            seen.update(kwargs)
            return SimpleNamespace(cfg=SimpleNamespace(num_hidden_layers=4))

        try:
            serve_engine.rt.build = fake
            serve_engine.Engine(engine_args(Path("/tmp/not-loaded"), io_threads=12, expert_batch=32))
        finally:
            serve_engine.rt.build = real
        self.assertEqual(seen["io_threads"], 12)
        self.assertEqual(seen["expert_batch"], 32)

    def test_the_streamer_gets_that_many_threads_and_experts_per_fetch(self):
        """`--expert-batch` is the model's read batch, not just the staging ring's capacity."""
        from ascend.dsv41 import runtime as rt

        with TinyModel() as model:
            run = rt.build(model=str(model.root), device="cpu", dtype="float32",
                           io_threads=3, expert_batch=5, graphs=False)
            try:
                self.assertEqual(run.experts._pool._max_workers, 3)
                self.assertEqual(run.experts._max_batch, 5)
                self.assertEqual(run.model.expert_batch, 5, "the MoE batches at the ring's size")
            finally:
                run.close()


@NEEDS_TORCH
class TheParallelArguments(unittest.TestCase):
    """`--parallel N` is opt-in, advertises multi=N, and its one conflict (graphs) is stated, not silent."""

    @staticmethod
    def parser():
        from ascend import serve_engine
        return serve_engine.build_parser()

    def test_one_conversation_is_the_default_and_ready_does_not_say_multi(self):
        args = self.parser().parse_args(["--serve", "--model", "/tmp/model"])
        self.assertEqual(args.parallel, 1, "the pre-existing, single-conversation behaviour")
        with TinyModel() as model:
            from ascend import serve_engine
            engine = serve_engine.Engine(engine_args(model.root, graphs=False))
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    engine.say_ready()
            finally:
                engine.shutdown()
        ready = [l for l in out.getvalue().splitlines() if l.startswith("READY")]
        self.assertEqual(ready, ["READY 64 stop pause"], "a single-conversation engine must not say multi=")
        self.assertNotIn("multi=", err.getvalue())

    def test_multi_is_the_number_the_scheduler_holds_and_the_graph_policy_is_logged(self):
        from ascend import serve_engine
        from types import SimpleNamespace

        seen: dict = {}
        real = serve_engine.rt.build

        def fake(**kwargs):
            seen.update(kwargs)
            model = SimpleNamespace(cfg=SimpleNamespace(num_hidden_layers=4),
                                    graph_max_conversations=4)
            return SimpleNamespace(cfg=model.cfg, model=model)

        try:
            serve_engine.rt.build = fake
            engine = serve_engine.Engine(engine_args(Path("/tmp/not-loaded"), parallel=4, graphs=True))
        finally:
            serve_engine.rt.build = real
        try:
            self.assertIs(seen["graphs"], True,
                          "graphs compose with --parallel as one domain per conversation, so the reserve is taken")
            self.assertTrue(engine._graphs_wanted)
            self.assertEqual(engine._graph_max, 4, "with no flag the bound is the engine's own (4)")
        finally:
            engine.shutdown()

        with TinyModel() as model:
            engine = serve_engine.Engine(engine_args(model.root, parallel=4, graphs=True))
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    engine.say_ready()
            finally:
                engine.shutdown()
        self.assertIn("READY 64 stop pause multi=4", out.getvalue())
        self.assertIn("one captured decode per conversation", err.getvalue(),
                      "the composition is stated, not silently resolved")
        self.assertIn("Up to 4 conversation(s)", err.getvalue())

    def test_parallel_parses(self):
        args = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--parallel", "8"])
        self.assertEqual(args.parallel, 8)

    def test_per_conversation_streams_are_opt_in(self):
        """Measured slower than the default stream on one 910B3, so the flag defaults off."""
        base = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--parallel", "4"])
        self.assertFalse(base.parallel_streams)
        on = self.parser().parse_args(["--serve", "--model", "/tmp/model", "--parallel", "4",
                                       "--parallel-streams"])
        self.assertTrue(on.parallel_streams)
        with TinyModel() as model:
            from ascend import serve_engine
            engine = serve_engine.Engine(engine_args(model.root, parallel=4, graphs=False,
                                                     parallel_streams=False))
            try:
                self.assertFalse(engine._parallel_streams)
                self.assertIsNone(engine._new_stream())
            finally:
                engine.shutdown()
            engine = serve_engine.Engine(engine_args(model.root, parallel=4, graphs=False,
                                                     parallel_streams=True))
            try:
                self.assertTrue(engine._parallel_streams)
                self.assertIsNone(engine._new_stream(), "a CPU test has no NPU stream to make")
            finally:
                engine.shutdown()


@NEEDS_TORCH
class TheParallelScheduler(unittest.TestCase):
    """`--parallel` serving: N requests in flight, tagged lines, per-request control, and isolation."""

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def engine(self, **overrides):
        from ascend import serve_engine
        base = {"parallel": 4, "graphs": False}
        base.update(overrides)
        return serve_engine.Engine(engine_args(self.model.root, **base))

    @staticmethod
    def collector(engine):
        """Record the engine's protocol lines; (`engine.emit` is the seam the bench uses too)."""
        lines: list[str] = []
        engine.emit = lines.append
        return lines

    @staticmethod
    def wait_done(lines, n: int, timeout: float = 60.0) -> list[str]:
        end = time.time() + timeout
        while time.time() < end:
            dones = [l for l in lines if l.startswith("DONE ")]
            if len(dones) >= n:
                return dones
            time.sleep(0.01)
        raise AssertionError(f"{n} DONE line(s) did not arrive; saw {lines[-12:]}")

    @staticmethod
    def tokens(lines, rid: int) -> list[int]:
        return [int(l.split()[2]) for l in lines if l.startswith(f"T {rid} ")]

    def test_four_requests_are_served_at_once_and_each_rides_its_own_tag(self):
        engine = self.engine()
        try:
            engine.say_ready()
            lines = self.collector(engine)
            engine.start_workers()
            prompts = [[3, 5, 7], [11, 13, 17], [19, 23, 29], [31, 2, 4]]
            for rid, ids in enumerate(prompts, 1):
                self.assertTrue(engine.submit(f"GEN 3 rid={rid} " + ",".join(str(i) for i in ids)))
            dones = self.wait_done(lines, 4)
            self.assertTrue(engine.wait_idle(timeout=10))
        finally:
            engine.stop_workers()
            engine.shutdown()
        self.assertEqual(len({int(d.split()[1]) for d in dones}), 4, f"one DONE per request: {dones}")
        for rid in range(1, 5):
            self.assertEqual(len(self.tokens(lines, rid)), 3, f"request {rid}'s three tokens are tagged with it")
        # every T line carries a request id, and only the four requests' ids
        tagged = {int(l.split()[1]) for l in lines if l.startswith("T ")}
        self.assertEqual(tagged, {1, 2, 3, 4})

    def test_a_request_past_parallel_waits_in_the_engine_queue(self):
        engine = self.engine(parallel=2)
        try:
            engine.say_ready()
            lines = self.collector(engine)
            # slow every step just enough that three requests cannot all be in flight at the slot count
            engine.start_workers()
            for rid in (1, 2, 3):
                engine.submit(f"GEN 2 rid={rid} " + ",".join(str(i) for i in (3, 5, 7 + rid)))
            dones = self.wait_done(lines, 3)
            self.assertTrue(engine.wait_idle(timeout=10))
        finally:
            engine.stop_workers()
            engine.shutdown()
        self.assertEqual(len(dones), 3)
        self.assertEqual(engine._queued_seen, 1, "the third request waited; the queue is what admitted it")

    def test_continuation_reuses_the_prefix_on_a_slot(self):
        engine = self.engine()
        try:
            engine.say_ready()
            lines = self.collector(engine)
            engine.start_workers()
            engine.submit("GEN 2 rid=1 3,5,7")
            first = self.wait_done(lines, 1)[0].split()
            produced = self.tokens(lines, 1)
            longer = [3, 5, 7] + produced + [11]
            engine.submit("GEN 1 rid=2 " + ",".join(str(i) for i in longer))
            second = self.wait_done(lines, 2)[1].split()
            self.assertTrue(engine.wait_idle(timeout=10))
        finally:
            engine.stop_workers()
            engine.shutdown()
        # DONE <rid> n prompt prompt_ms decode_ms finish da do reused hits lookups
        self.assertEqual(int(first[1]), 1)
        self.assertEqual(int(second[9]), len(longer) - 1, "the next turn's whole prefix was held")

    def test_stop_rid_cuts_only_that_request(self):
        engine = self.engine()
        try:
            engine.say_ready()
            lines = self.collector(engine)
            real_sample = engine.sample

            def slow_sample(*a, **kw):
                time.sleep(0.05)                     # ~2 s for 40 tokens: room to cut one and not the other
                return real_sample(*a, **kw)

            engine.sample = slow_sample
            engine.start_workers()
            engine.submit("GEN 40 rid=1 3,5,7")
            engine.submit("GEN 40 rid=2 11,13,17")
            end = time.time() + 20
            while len([l for l in lines if l.startswith("T ")]) < 2 and time.time() < end:
                time.sleep(0.01)
            engine.submit("STOP 1")
            dones = self.wait_done(lines, 1, timeout=20)
            cut_at = len(self.tokens(lines, 2))
            engine.submit("STOP 2")
            dones = self.wait_done(lines, 2, timeout=20)
        finally:
            engine.stop_workers()
            engine.shutdown()
        by_rid = {int(d.split()[1]): d.split() for d in dones}
        self.assertEqual(by_rid[1][6], "cancel")
        self.assertLess(int(by_rid[1][2]), 32, "request 1 was cut")
        self.assertGreater(len(self.tokens(lines, 2)), cut_at, "request 2 kept generating after request 1 was cut")
        self.assertEqual(by_rid[2][6], "cancel")

    def test_the_same_requests_produce_the_same_tokens_alone_and_concurrently(self):
        """The isolation bar: four distinct prompts, greedy, same ids with and without the other three."""
        prompts = [[3, 5, 7, 11, 13], [17, 19, 23, 29, 31], [2, 4, 6, 8, 10], [1, 3, 9, 27, 12]]
        engine = self.engine()
        try:
            engine.say_ready()
            alone = []
            for ids in prompts:                       # one at a time, the pre-change serve path
                out, _ = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out):
                    engine.generate("GEN 6 " + ",".join(str(i) for i in ids))
                alone.append([int(l.split()[1]) for l in out.getvalue().splitlines()
                              if l.startswith("T ")])
            lines = self.collector(engine)
            engine.start_workers()
            for rid, ids in enumerate(prompts, 1):
                engine.submit(f"GEN 6 rid={rid} " + ",".join(str(i) for i in ids))
            self.wait_done(lines, 4)
            self.assertTrue(engine.wait_idle(timeout=30))
            concurrent = [self.tokens(lines, rid) for rid in range(1, 5)]
        finally:
            engine.stop_workers()
            engine.shutdown()
        self.assertEqual(concurrent, alone, "a concurrent conversation must not change another's tokens")
        self.assertEqual(len({tuple(s) for s in concurrent}), 4, "the four prompts are actually distinct")

    def test_an_untagged_gen_is_refused_rather_than_guessed(self):
        engine = self.engine()
        try:
            engine.say_ready()
            lines = self.collector(engine)
            engine.start_workers()
            engine.submit("GEN 1 3,5,7")               # no rid=: a legacy line cannot be routed
            end = time.time() + 5
            while not any(l.startswith("ERR ") for l in lines) and time.time() < end:
                time.sleep(0.01)
        finally:
            engine.stop_workers()
            engine.shutdown()
        self.assertTrue(any("needs rid=" in l for l in lines), lines)

    def test_the_expert_streamer_is_only_locked_while_the_scheduler_runs(self):
        """The shared pinned ring needs one fetch at a time under --parallel; the default path is untouched."""
        engine = self.engine(parallel=1, graphs=False)
        try:
            self.assertFalse(getattr(engine.rt.experts.fetch_int8, "_serve_locked", False),
                             "--parallel 1 must not add a lock to the streaming path")
        finally:
            engine.shutdown()
        engine = self.engine()
        try:
            self.assertTrue(getattr(engine.rt.experts.fetch_int8, "_serve_locked", False))
            self.assertFalse(engine._multi_running)
        finally:
            engine.shutdown()


@NEEDS_TORCH
class ThePrefixReuse(unittest.TestCase):
    """A prompt that shares a prefix with a resident conversation resumes from its KV.

    The bar is exact: the tokens a reused conversation produces must be the tokens a conversation that
    prefilled the same prompt from scratch produces (greedy, same sampling).
    """

    SHARED = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 4, 6, 8, 10, 12, 14]     # 16 tokens
    A_TAIL = [16, 18, 20, 22]                                              # request A's own suffix
    B_TAIL = [9, 15, 21, 27, 25, 26]                                       # request B's own suffix
    A = SHARED + A_TAIL
    B = SHARED + B_TAIL

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def engine(self, **overrides):
        from ascend import serve_engine
        base = {"parallel": 1, "graphs": False, "prefill_chunk": 512}
        base.update(overrides)
        return serve_engine.Engine(engine_args(self.model.root, **base))

    @staticmethod
    def tokens(out: str) -> list[int]:
        return [int(l.split()[1]) for l in out.splitlines() if l.startswith("T ")]

    @staticmethod
    def done(out: str) -> list[str]:
        return [l for l in out.splitlines() if l.startswith("DONE ")][0].split()

    def test_a_divergent_prompt_reads_only_its_new_suffix(self):
        engine = self.engine()
        try:
            out_a, _ = quiet_generate(engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            self.assertIn("DONE 4", out_a)
            out_b, err_b = quiet_generate(engine, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            engine.shutdown()
        done = self.done(out_b)
        self.assertEqual(int(done[8]), len(self.SHARED),
                         "DONE's reused field reports the shared prefix, so the server sees the hit")
        self.assertIn("dsv41 prefix: the prompt shares 16", err_b, "the reuse is stated in the log")
        pp = [l for l in out_b.splitlines() if l.startswith("PP ")]
        self.assertEqual(len(pp), 1, "one eager forward: the 6-token suffix, not the 22-token prompt")
        self.assertEqual(int(pp[0].split()[2]), 16 + len(self.B_TAIL))

    def test_resumed_tokens_equal_a_scratch_prefill(self):
        """The correctness bar: same prompt, same greedy tokens, reused or read from token 0."""
        reused_engine = self.engine()
        try:
            quiet_generate(reused_engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            out_reused, _ = quiet_generate(reused_engine, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            reused_engine.shutdown()
        scratch = self.engine(prefill_chunk=512, prefix_reuse=False)     # a fresh State, token 0
        try:
            out_scratch, err = quiet_generate(scratch, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            scratch.shutdown()
        self.assertEqual(int(self.done(out_scratch)[8]), 0, "the control really prefilled from token 0")
        self.assertNotIn("dsv41 prefix:", err)
        self.assertEqual(self.tokens(out_reused), self.tokens(out_scratch),
                         "a resumed-from-prefix conversation must produce the same tokens")

    def test_a_prompt_that_is_a_prefix_of_the_held_conversation_rewinds(self):
        """A re-sent shorter prompt: the tail is discarded and only the last token is read again."""
        engine = self.engine()
        try:
            quiet_generate(engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            shorter = self.A[:12]
            out, _ = quiet_generate(engine, "GEN 2 " + ",".join(str(t) for t in shorter))
        finally:
            engine.shutdown()
        self.assertEqual(int(self.done(out)[8]), len(shorter) - 1,
                         "everything but the token whose logits are needed was reused")
        self.assertEqual(len([l for l in out.splitlines() if l.startswith("PP ")]), 1)

    def test_prefix_reuse_can_be_turned_off(self):
        engine = self.engine(prefix_reuse=False)
        try:
            quiet_generate(engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            out, err = quiet_generate(engine, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            engine.shutdown()
        self.assertEqual(int(self.done(out)[8]), 0, "the flag restores the re-prefill-everything path")
        self.assertNotIn("dsv41 prefix:", err)

    def test_safe_mode_reuses_only_the_chunk_aligned_prefix(self):
        """`--prefix-reuse=safe` stops on the chunk grid, not at the whole common prefix.

        The conversation's first 16 tokens are four full 4-token prompt chunks; the held state also has
        four decode-written rows.  A prompt that shares 18 tokens may resume at 16 (the last aligned chunk
        boundary) but not at 18: the from-scratch path writes tokens 16..19 in one 4-token forward, so a
        suffix starting at 18 would compute rows 18..19 with a different shape.
        """
        engine = self.engine(prefill_chunk=4, prefix_reuse="safe")
        longer = self.A[:18] + [9, 9]
        try:
            quiet_generate(engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            out, err = quiet_generate(engine, "GEN 6 " + ",".join(str(t) for t in longer))
        finally:
            engine.shutdown()
        self.assertEqual(int(self.done(out)[8]), 16,
                         "safe reuse must stop at the last full aligned 4-token chunk")
        self.assertIn("not on the 4-token chunk grid", err)
        scratch = self.engine(prefill_chunk=4, prefix_reuse=False)
        try:
            out_scratch, _ = quiet_generate(scratch, "GEN 6 " + ",".join(str(t) for t in longer))
        finally:
            scratch.shutdown()
        self.assertEqual(self.tokens(out), self.tokens(out_scratch),
                         "the aligned reuse must equal a from-scratch prefill, token for token")

    def test_safe_mode_is_a_no_op_when_the_chunk_exceeds_the_prompt(self):
        """No chunk grid to align to: safe mode must not reuse, so it cannot be inexact either."""
        engine = self.engine(prefill_chunk=512, prefix_reuse="safe")
        try:
            quiet_generate(engine, "GEN 4 " + ",".join(str(t) for t in self.A))
            out, err = quiet_generate(engine, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            engine.shutdown()
        self.assertEqual(int(self.done(out)[8]), 0,
                         "a 20-token prompt on a 512-token grid has no aligned chunk to reuse")
        self.assertIn("safe reuse stops at 0", err)

    def test_the_eos_is_not_claimed_by_the_state_it_never_entered(self):
        """The emitted eos is not fed (the loop breaks first), so it must not be in `held`.

        Otherwise the next turn's prompt - which usually does contain the eos - starts one position early,
        and a rewind to a shared prefix is off by one.  The resumed turn must equal a fresh prefill.
        """
        engine = self.engine()
        try:
            engine.sample = lambda *a, **k: 1                 # the first sampled token is the tiny eos
            out_eos, _ = quiet_generate(engine, "GEN 2 3,5,7")
            self.assertEqual(self.done(out_eos)[1], "1")
            self.assertIn("stop", out_eos)
            self.assertEqual(engine.held, [3, 5, 7], "the emitted eos is not covered by the State")
            self.assertEqual(engine.state.next_pos, 3, "cursor and held agree")
            del engine.sample
            out, _ = quiet_generate(engine, "GEN 2 3,5,7,1,5")   # the next turn carries the eos
        finally:
            engine.shutdown()
        scratch = self.engine()
        try:
            out_scratch, _ = quiet_generate(scratch, "GEN 2 3,5,7,1,5")
        finally:
            scratch.shutdown()
        self.assertEqual(self.tokens(out), self.tokens(out_scratch),
                         "a turn after an eos must match a fresh prefill of the same prompt")

    def test_a_parallel_slot_reuses_the_prefix_of_another_conversation(self):
        """Cross-conversation reuse: the second request lands on the slot holding its prefix."""
        engine = self.engine(parallel=2)
        try:
            engine.say_ready()
            lines = []
            engine.emit = lines.append
            engine.start_workers()
            engine.submit("GEN 4 rid=1 " + ",".join(str(t) for t in self.A))
            self.assertTrue(self.wait_done(lines, 1))
            engine.submit("GEN 6 rid=2 " + ",".join(str(t) for t in self.B))
            self.assertTrue(self.wait_done(lines, 2))
            self.assertTrue(engine.wait_idle(timeout=30))
        finally:
            engine.stop_workers()
            engine.shutdown()
        done = [l for l in lines if l.startswith("DONE 2 ")][0].split()
        self.assertEqual(int(done[9]), len(self.SHARED),
                         "the second conversation resumed from the resident one's KV")
        self.assertEqual(engine._queued_seen, 0)
        scratch = self.engine(parallel=1, prefix_reuse=False)
        try:
            out_scratch, _ = quiet_generate(scratch, "GEN 6 " + ",".join(str(t) for t in self.B))
        finally:
            scratch.shutdown()
        self.assertEqual([int(l.split()[2]) for l in lines if l.startswith("T 2 ")],
                         self.tokens(out_scratch), "the parallel reuse is exact too")

    def test_done_releases_the_slot_before_the_server_can_send_the_next_turn(self):
        """The next turn arrives on DONE; the slot must be free by then, not one line later.

        `engine.emit` is the seam a server reads the protocol through, and `DONE` is its end-of-request
        signal: submitting the next turn from inside `emit` is that contract in its tightest form, with
        no scheduling window for the engine's slot bookkeeping to hide in.  If the slot were still
        marked busy when `DONE` was written, the turn would land on the other, empty slot and `reused`
        would be 0: the prompt prefill and its KV would be paid twice across a conversation.
        """
        engine = self.engine(parallel=2)
        lines: list[str] = []
        sent = {"next": False}

        def emit(line: str) -> None:
            lines.append(line)
            if line.startswith("DONE 1 ") and not sent["next"]:
                sent["next"] = True
                engine.submit("GEN 6 rid=2 " + ",".join(str(t) for t in self.B))

        engine.emit = emit
        try:
            engine.say_ready()
            engine.start_workers()
            engine.submit("GEN 4 rid=1 " + ",".join(str(t) for t in self.A))
            self.assertTrue(self.wait_done(lines, 2))
            self.assertTrue(engine.wait_idle(timeout=30))
        finally:
            engine.stop_workers()
            engine.shutdown()
        done = [l for l in lines if l.startswith("DONE 2 ")][0].split()
        self.assertEqual(int(done[9]), len(self.SHARED),
                         "the turn after DONE found its conversation idle and resumed from its KV")
        self.assertEqual(engine._queued_seen, 0, "a freed slot is available; nothing had to queue")

    @staticmethod
    def wait_done(lines, n: int, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            if len([l for l in lines if l.startswith("DONE ")]) >= n:
                return True
            time.sleep(0.01)
        raise AssertionError(f"{n} DONE line(s) did not arrive; saw {lines[-12:]}")


@NEEDS_TORCH
class TheGraphsParallelDomains(unittest.TestCase):
    """`--graphs --parallel N`: up to the engine's bound conversations each hold a captured domain.

    The bound is the model's (`DSV41_GRAPH_MAX_CONVERSATIONS`, four by default) and
    `--graph-max-conversations` can only lower it.  A conversation past the bound runs eagerly and says so
    (naming the bound and the 1.585 GiB per-domain cost); no live domain is evicted; a conversation that
    ends releases *its* domain so the next one can take the slot; and a capture failure retires only the
    conversation that failed.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = TinyModel()

    @classmethod
    def tearDownClass(cls):
        cls.model.close()

    def engine(self, **overrides):
        from ascend import serve_engine
        base = {"parallel": 2, "graphs": True}
        base.update(overrides)
        return serve_engine.Engine(engine_args(self.model.root, **base))

    @staticmethod
    def collector(engine):
        lines: list[str] = []
        engine.emit = lines.append
        return lines

    @staticmethod
    def wait_done(lines, n: int, timeout: float = 120.0) -> list[str]:
        end = time.time() + timeout
        while time.time() < end:
            dones = [l for l in lines if l.startswith("DONE ")]
            if len(dones) >= n:
                return dones
            time.sleep(0.01)
        raise AssertionError(f"{n} DONE line(s) did not arrive; saw {lines[-12:]}")

    def test_the_start_log_states_the_bound_and_the_flag_can_only_lower_it(self):
        from ascend import serve_engine

        with TinyModel() as model:
            engine = serve_engine.Engine(engine_args(model.root, parallel=4, graphs=True,
                                                     graph_max_conversations=2))
            out, err = io.StringIO(), io.StringIO()
            try:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    engine.say_ready()
            finally:
                engine.shutdown()
        log = err.getvalue()
        self.assertIn("READY 64 stop pause multi=4", out.getvalue())
        self.assertIn("one captured decode per conversation", log)
        self.assertIn("Up to 2 conversation(s)", log)
        self.assertIn("caps the engine's bound of 4", log)
        self.assertIn("1.585 GiB = 405.7 MiB a card", log,
                      "the per-domain cost is stated, not implied")
        self.assertIn("no live domain is ever evicted", log)

        with TinyModel() as model:
            engine = serve_engine.Engine(engine_args(model.root, parallel=4, graphs=True,
                                                     graph_max_conversations=8))
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                    engine.say_ready()
            finally:
                engine.shutdown()
        self.assertEqual(engine._graph_max, 4, "the flag cannot raise the engine's bound")
        self.assertIn("cannot raise the engine's bound of 4", err.getvalue())

    def test_the_bound_is_respected_the_rest_run_eagerly_and_are_logged(self):
        """With the bound at 2 and four conversations, exactly two capture and two are eager (and say so).

        All four are held at their second token: the first decode step - the lazy capture - has completed
        for every one of them by then, so the domain count is final.
        """
        engine = self.engine(parallel=4, graph_max_conversations=2)
        engine.eos_id = lambda: -1                # 20 tokens of every stream: the step counts are exact
        real_sample, local = engine.sample, threading.local()
        first = threading.Barrier(4, timeout=120)
        second = threading.Semaphore(0)
        gate = threading.Event()

        def held_sample(*a, **kw):
            n = getattr(local, "n", 0) + 1
            local.n = n
            if n == 1:
                first.wait()                  # all four are admitted and about to reach their first step
            elif n == 2:
                second.release()              # its first decode step (and any capture) is done
                if not gate.wait(120):
                    raise AssertionError("the test did not release the decodes")
            return real_sample(*a, **kw)

        engine.sample = held_sample
        err, lines = io.StringIO(), self.collector(engine)
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                engine.say_ready()
                engine.start_workers()
                for rid, ids in enumerate(((3, 5, 7), (11, 13, 17), (19, 23, 29), (2, 4, 31)), 1):
                    engine.submit(f"GEN 20 rid={rid} " + ",".join(map(str, ids)))
                for _ in range(4):
                    self.assertTrue(second.acquire(timeout=120),
                                    f"not every conversation reached its second token; saw {lines[-8:]}")
                gate.set()
                self.assertTrue(engine.wait_idle(timeout=120))
            model = engine.rt.model
            domains = dict(model._graph_domains)
            captured = [s for s in engine.slots if s.state is not None and id(s.state) in domains]
            eager = [s for s in engine.slots if s.state is not None and id(s.state) not in domains]
            self.assertEqual(len(captured), 2, "exactly the bound captures")
            self.assertEqual(len(eager), 2, "the rest still serve, on their own States, eagerly")
            for slot in captured:
                self.assertEqual(slot.graph_steps, 20,
                                 "every decode step after the capture ran on the conversation's own domain")
            for slot in eager:
                self.assertEqual(slot.graph_steps, 0, "a conversation past the bound never got a domain")
            self.assertEqual(engine.graphs_live(), 2)
            self.assertTrue(model.graphs, "the bound must not disable the captured path")
            self.assertNotIn("revoked", err.getvalue())
            self.assertEqual(err.getvalue().count("already hold their domain (bound 2"), 2,
                             "every conversation past the bound says so, naming the bound")
            self.assertIn("1.585 GiB = 405.7 MiB a card", err.getvalue())
            self.assertIn("no live domain is evicted", err.getvalue())
            self.assertEqual(err.getvalue().count("graphs captured"), 2,
                             "only the two captured conversations report a capture")
        finally:
            gate.set()
            engine.stop_workers()
            engine.shutdown()

    def test_eight_conversations_capture_four_and_the_rest_are_eager(self):
        """Eight concurrent conversations at the engine's default bound of four: 4 capture, 4 are refused.

        This is the "past the bound" shape the task asks for, at the policy level: the four extra
        conversations still serve, eagerly, and every one of them logs the bound and the per-domain cost;
        no live domain is touched.  (The NPU 8-conversation arm also checks HBM and thrash; this test
        pins the policy and the log without a card.)
        """
        engine = self.engine(parallel=8, graph_max_conversations=None)
        self.assertEqual(engine._graph_max, 4, "unset means the engine's own bound (4)")
        engine.eos_id = lambda: -1
        real_sample, local = engine.sample, threading.local()
        first = threading.Barrier(8, timeout=180)
        second = threading.Semaphore(0)
        gate = threading.Event()

        def held_sample(*a, **kw):
            n = getattr(local, "n", 0) + 1
            local.n = n
            if n == 1:
                first.wait()
            elif n == 2:
                second.release()
                if not gate.wait(180):
                    raise AssertionError("the test did not release the decodes")
            return real_sample(*a, **kw)

        engine.sample = held_sample
        prompts = ((3, 5, 7), (11, 13, 17), (19, 23, 29), (2, 4, 31),
                   (6, 8, 10), (12, 14, 16), (18, 20, 22), (24, 26, 28))
        err, lines = io.StringIO(), self.collector(engine)
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                engine.say_ready()
                engine.start_workers()
                for rid, ids in enumerate(prompts, 1):
                    engine.submit(f"GEN 6 rid={rid} " + ",".join(map(str, ids)))
                for _ in range(8):
                    self.assertTrue(second.acquire(timeout=180),
                                    f"not every conversation reached its second token; saw {lines[-8:]}")
                gate.set()
                self.assertTrue(engine.wait_idle(timeout=180))
            model = engine.rt.model
            domains = dict(model._graph_domains)
            captured = [s for s in engine.slots if s.state is not None and id(s.state) in domains]
            eager = [s for s in engine.slots if s.state is not None and id(s.state) not in domains]
            self.assertEqual(len(captured), 4, "the engine's default bound captures four")
            self.assertEqual(len(eager), 4, "the other four conversations run eagerly")
            for slot in captured:
                self.assertEqual(slot.graph_steps, 6, "6 decode forwards on its own domain")
            for slot in eager:
                self.assertEqual(slot.graph_steps, 0)
            self.assertEqual(engine.graphs_live(), 4)
            self.assertTrue(model.graphs)
            log = err.getvalue()
            self.assertEqual(log.count("already hold their domain (bound 4"), 4,
                             "each refused conversation names the bound, and none is silent")
            self.assertIn("1.585 GiB = 405.7 MiB a card", log)
            self.assertIn("no live domain is evicted", log)
            self.assertEqual(log.count("graphs captured"), 4)
        finally:
            gate.set()
            engine.stop_workers()
            engine.shutdown()

    def test_a_captured_conversation_that_ends_frees_its_slot_for_the_next(self):
        """With the bound at 1, the conversation that ends releases its domain and its successor captures.

        The second request's first token differs from the held conversation's, so the engine drops that
        conversation (and only its graphs) before building the new State.  The bound is 1, so the new
        conversation can capture only if the old domain really was released - the freed slot is reused,
        not a second domain added.
        """
        engine = self.engine(parallel=2, graph_max_conversations=1)
        err, lines = io.StringIO(), self.collector(engine)
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                engine.say_ready()
                engine.start_workers()
                engine.submit("GEN 3 rid=1 3,5,7")
                self.wait_done(lines, 1)
                self.assertTrue(engine.wait_idle(timeout=120))
                first = engine.rt.model._decode_graphs
                self.assertIsNotNone(first, "the lone conversation captured")
                self.assertEqual(engine.graphs_live(), 1)
                # the first token differs: the held conversation is dropped before the prompt is read
                engine.submit("GEN 3 rid=2 11,13,17")
                self.wait_done(lines, 2)
                self.assertTrue(engine.wait_idle(timeout=120))
            log = err.getvalue()
            self.assertTrue(first.closed, "the ended conversation's graphs were torn down")
            self.assertEqual(engine.graphs_live(), 1,
                             "the released domain freed the slot the next conversation took")
            fresh = engine.rt.model._decode_graphs
            self.assertIsNotNone(fresh, "the conversation started in its place captured")
            self.assertIsNot(fresh, first)
            self.assertGreater(int(getattr(fresh, "captured", 0)), 0)
            self.assertIn("released", log)
            self.assertIn("graphs captured", log)
        finally:
            engine.stop_workers()
            engine.shutdown()

    def test_a_capture_failure_retires_only_that_conversation(self):
        from ascend.dsv41 import model as model_mod

        engine = self.engine(parallel=2, graph_max_conversations=2)
        real_init = model_mod._DecodeGraphs.__init__
        calls, lock = {"n": 0}, threading.Lock()

        def failing_init(self, *a, **kw):
            with lock:
                calls["n"] += 1
                fail = calls["n"] == 1
            if fail:
                raise RuntimeError("injected domain build failure")
            return real_init(self, *a, **kw)

        err, lines = io.StringIO(), self.collector(engine)
        model_mod._DecodeGraphs.__init__ = failing_init
        try:
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
                engine.say_ready()
                engine.start_workers()
                engine.submit("GEN 6 rid=1 3,5,7")
                engine.submit("GEN 6 rid=2 11,13,17")
                self.wait_done(lines, 2)
                self.assertTrue(engine.wait_idle(timeout=120))
        finally:
            model_mod._DecodeGraphs.__init__ = real_init
            engine.stop_workers()
        try:
            model = engine.rt.model
            self.assertTrue(model.graphs,
                            "a failed capture must not take the captured path away from the others")
            self.assertEqual(len(model._graph_domains), 1, "the healthy conversation keeps its domain")
            self.assertEqual(len(model._graph_failures), 1)
            self.assertEqual(engine.graphs_live(), 1)
            domains = dict(model._graph_domains)
            captured = [s for s in engine.slots if s.state is not None and id(s.state) in domains]
            eager = [s for s in engine.slots if s.state is not None and id(s.state) not in domains]
            self.assertEqual((len(captured), len(eager)), (1, 1))
            self.assertEqual(eager[0].graph_steps, 0)
            self.assertIn("capture/replay failed", err.getvalue())
            self.assertIn("retired only this conversation", err.getvalue())
        finally:
            engine.shutdown()

    def test_a_parallel_engine_tags_its_lines_and_quits(self):
        """The real executable: READY multi=N, a rid-tagged GEN, tagged T/DONE, and a clean QUIT."""
        with TinyModel() as model:
            engine = Engine(model.root, "--parallel", "2", "--graphs")
            ready = engine.read_until("READY")[-1]
            self.assertIn("multi=2", ready.split())
            end = time.time() + 20
            while not any("one captured decode per conversation" in l for l in engine.log) and time.time() < end:
                time.sleep(0.02)
            self.assertIn("one captured decode per conversation", "\n".join(engine.log),
                          "the graphs/parallel policy is stated at start")
            engine.send("GEN 2 rid=7 3,5,7")
            lines = engine.read_until("DONE ")
            self.assertTrue([l for l in lines if l.startswith("T 7 ")], lines)
            self.assertTrue([l for l in lines if l.startswith("DONE 7 ")], lines)
            self.assertEqual(engine.stop(), 0)


if __name__ == "__main__":
    unittest.main()