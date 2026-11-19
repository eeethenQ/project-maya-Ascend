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
        graphs=True, graph_verify=False, graph_max_conversations=1)
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

    def test_graphs_are_off_by_default(self):
        args = self.parser().parse_args(["--serve", "--model", "/tmp/model"])
        self.assertFalse(args.graphs, "the captured decode is opt-in, as in run_dsv41")
        self.assertFalse(args.graph_verify)
        self.assertEqual(args.graph_max_conversations, 1)

    def test_the_graph_flags_parse(self):
        args = self.parser().parse_args(
            ["--serve", "--model", "/tmp/model", "--graphs", "--graph-verify",
             "--graph-max-conversations", "3"])
        self.assertTrue(args.graphs)
        self.assertTrue(args.graph_verify)
        self.assertEqual(args.graph_max_conversations, 3)


@NEEDS_TORCH
class TheGraphsPlumbing(unittest.TestCase):
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
            engine = serve_engine.Engine(engine_args(Path("/tmp/not-loaded"), graphs=True,
                                                     graph_verify=True, graph_max_conversations=2))
        finally:
            serve_engine.rt.build = real
        self.assertIs(seen["graphs"], True)
        self.assertIs(seen["graph_verify"], True)
        self.assertEqual(engine._graph_max, 2)


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

    def test_a_conversation_captures_on_its_first_decode_and_keeps_the_graphs(self):
        engine = self.engine()
        try:
            out, err = quiet_generate(engine, "GEN 2 3,5,7")
            self.assertIn("DONE 2 3", out)
            graphs = engine.rt.model._decode_graphs
            self.assertIsNotNone(graphs)
            self.assertEqual(graphs.captured, 2 * engine.cfg.num_hidden_layers,
                             "one g_pre and one g_post per layer")
            self.assertIn("captures its decode", err)
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

    def test_a_teardown_failure_is_logged_and_the_bound_falls_back_to_eager(self):
        from ascend.dsv41.graph_runner import GraphTeardownError

        engine = self.engine()
        try:
            quiet_generate(engine, "GEN 2 3,5,7")
            model = engine.rt.model
            real = model.release_graphs
            calls = {"n": 0}

            def failing():
                calls["n"] += 1
                if calls["n"] == 1:
                    # what `Dsv41.release_graphs` does when a destroy failed: drop the reference, name
                    # the leak.  The HBM is not returned, and the domain can never be freed again.
                    domain, model._decode_graphs = model._decode_graphs, None
                    if domain is not None:
                        domain.close()
                    raise GraphTeardownError(
                        "npu:0", [("g_pre.0.0", "rtModelDestroy failed: model not in current ctx")])
                return real()

            model.release_graphs = failing
            out, err = quiet_generate(engine, "GEN 1 11,13")
            self.assertIn("DONE 1 2", out, "one conversation's teardown failure must not kill the engine")
            self.assertIn("could not release", err)
            self.assertEqual(engine.graphs_live(), 1, "the leaked domain is counted against the bound")
            self.assertFalse(model.graphs, "the bound sends the next conversation to the eager path")
            self.assertIsNone(model._decode_graphs, "nothing captures into HBM the leak already holds")
            self.assertIn("eagerly", err)
            out, _ = quiet_generate(engine, "GEN 1 19,23")
            self.assertIn("DONE 1 2", out)
            self.assertIsNone(model._decode_graphs)
        finally:
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


if __name__ == "__main__":
    unittest.main()
