"""ascend/tests/test_serve_engine.py - the Ascend engine against the protocol `serve/server.py` drives.

`serve/test_engine_contract.py` is that contract, but it needs the real model and a card.  This drives the
same protocol with the engine started the way the server starts it - `[exe, "--serve", ...]`, lines over
stdin/stdout - on a four-layer model on the host, so the parts that are easy to get wrong are tested in CI:
READY and the INFO keys the dashboard reads, one PP per prompt chunk, one T per token, DONE's eleven
fields, the conversation the engine keeps between requests (`reused`), STOP cutting a decode, and QUIT
ending with exit code 0.
"""
from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
