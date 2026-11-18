"""ascend/tests/test_graph_runner.py - the capture wrapper's contract, mostly without a card.

Three of the four properties under test are properties of the *wrapper*, not of the NPU: the CPU
fallback must equal eager, a changed input shape/address must be refused by name, and ``close`` must
be idempotent.  The did-it-capture guard is also checked on the fallback, where it must report
``mode="fallback"`` rather than pretend a graph was captured.  The one NPU test is opt-in with
``DSV41_RUN_NPU_TESTS=1`` because the cards are shared; it is skipped automatically when no device
node exists.
"""
from __future__ import annotations

import os
import unittest

import torch

from ascend.dsv41.graph_runner import (
    CaptureVerificationError,
    GraphInputContractError,
    GraphRunner,
)


def _npu_test_device() -> str | None:
    """The device for the opt-in NPU test, or ``None``.

    This runs at import time and deliberately does not touch ``torch.npu``: importing torch_npu only
    registers the backend.  The environment variable is an explicit second lock because the cards are
    shared with another tenant.
    """
    if os.environ.get("DSV41_RUN_NPU_TESTS") != "1":
        return None
    if not any(os.path.exists(f"/dev/davinci{i}") for i in range(16)):
        return None
    try:
        import torch_npu  # noqa: F401 - registers the backend
    except Exception:                                   # noqa: BLE001 - no torch_npu means no NPU test
        return None
    if not hasattr(torch, "npu"):
        return None
    return os.environ.get("DSV41_NPU_TEST_DEVICE", "npu:0")


NPU_DEVICE = _npu_test_device()


class TheFallback(unittest.TestCase):
    """On CPU the runner must be a straight eager call, not a fake capture."""

    def setUp(self):
        self.runner = GraphRunner("cpu")
        self.x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.out = torch.zeros_like(self.x)

    def tearDown(self):
        self.runner.close()

    def test_capture_and_replay_equal_eager(self):
        def fn():
            self.out.copy_(self.x * 2.0 + 1.0)
            return self.out

        want = fn().clone()
        self.runner.capture("scale", fn, inputs={"x": self.x})

        report = self.runner.report("scale")
        self.assertEqual(report.mode, "fallback")
        self.assertFalse(report.verified)
        self.assertTrue(report.ok)
        self.assertIsNone(self.runner.pool)
        self.assertEqual(self.runner.names, ["scale"])

        self.out.zero_()
        self.runner.replay("scale")
        torch.testing.assert_close(self.runner.outputs("scale"), want, rtol=0, atol=0)

    def test_capture_does_not_run_the_callable_on_cpu(self):
        calls = []

        def fn():
            calls.append(1)
            self.out.copy_(self.x)
            return self.out

        self.runner.capture("stored", fn, warmup=True, inputs={"x": self.x})
        self.assertEqual(calls, [], "cpu capture must store the callable, not warm it up")
        self.runner.replay("stored")
        self.assertEqual(calls, [1])


class TheInputContract(unittest.TestCase):
    def setUp(self):
        self.runner = GraphRunner("cpu")
        self.x = torch.zeros(2, 3)
        self.out = torch.zeros(2, 3)
        self.runner.capture("copy", lambda: self.out.copy_(self.x), inputs={"activation": self.x})

    def tearDown(self):
        self.runner.close()

    def test_a_changed_shape_is_refused_by_name(self):
        self.x.resize_(2, 4)
        with self.assertRaises(GraphInputContractError) as caught:
            self.runner.replay("copy")
        message = str(caught.exception)
        self.assertIn("activation", message)
        self.assertIn("shape", message)
        self.assertIn("(2, 3)", message)
        self.assertIn("(2, 4)", message)

    def test_a_reallocated_tensor_is_refused_by_address(self):
        replacement = torch.ones(2, 3)
        self.x.set_(replacement)
        with self.assertRaises(GraphInputContractError) as caught:
            self.runner.replay("copy")
        message = str(caught.exception)
        self.assertIn("activation", message)
        self.assertIn("data_ptr", message)


class TheGuard(unittest.TestCase):
    def setUp(self):
        self.runner = GraphRunner("cpu")
        self.x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.out = torch.zeros_like(self.x)

    def tearDown(self):
        self.runner.close()

    def test_the_guard_reports_the_fallback_not_a_graph(self):
        def fn():
            self.out.copy_(self.x * 3.0)
            return self.out

        self.runner.capture("guarded", fn, inputs={"x": self.x}, verify=True)
        report = self.runner.report("guarded")
        self.assertEqual(report.mode, "fallback")
        self.assertFalse(report.verified, "a CPU fallback must never be reported as a verified graph")
        self.assertTrue(report.ok)
        self.assertIn("fallback", report.detail.lower())
        self.assertIn("fallback", str(report).lower())
        self.assertTrue(report.checks, "the guard should document the checks it did run")

    def test_the_guard_refuses_a_callable_that_ignores_its_input(self):
        def fn():
            self.out.zero_()
            return self.out

        with self.assertRaises(CaptureVerificationError) as caught:
            self.runner.capture("proves-nothing", fn, inputs={"x": self.x}, verify=True)
        self.assertIn("did not change", str(caught.exception))
        self.assertEqual(self.runner.names, [], "a failed guard must not leave a captured entry behind")


class TheLifecycle(unittest.TestCase):
    def test_close_is_idempotent_and_drops_the_pool(self):
        runner = GraphRunner("cpu")
        x = torch.zeros(2, 3)
        out = torch.zeros(2, 3)
        runner.capture("g", lambda: out.copy_(x), inputs={"x": x})

        self.assertIsNone(runner.pool)
        runner.close()
        self.assertTrue(runner.closed)
        self.assertIsNone(runner.pool)
        self.assertEqual(runner.names, [])
        runner.close()                                  # second close is a no-op
        with self.assertRaises(RuntimeError):
            runner.replay("g")


@unittest.skipUnless(NPU_DEVICE is not None,
                     "set DSV41_RUN_NPU_TESTS=1 with a free card for the NPU test")
class OnAnNpu(unittest.TestCase):
    """The real thing, opt-in only: capture, guard, replay, bit-identical."""

    @classmethod
    def setUpClass(cls):
        if torch.npu.device_count() < 1:
            raise unittest.SkipTest("torch_npu sees no Ascend device")
        cls.runner = GraphRunner(NPU_DEVICE, capture_error_mode="relaxed")
        cls.x = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(NPU_DEVICE)
        cls.out = torch.zeros_like(cls.x)

    @classmethod
    def tearDownClass(cls):
        cls.runner.close()

    def test_capture_and_replay_is_bit_identical(self):
        def fn():
            self.out.copy_(self.x * 2.0 + 1.0)
            return self.out

        self.runner.capture("npu_scale", fn, inputs={"x": self.x}, verify=True)
        report = self.runner.report("npu_scale")
        self.assertEqual(report.mode, "graph")
        self.assertTrue(report.verified)
        self.assertTrue(report.ok)

        self.out.zero_()
        self.runner.replay("npu_scale", sync=True)
        want = self.x * 2.0 + 1.0
        torch.testing.assert_close(self.out, want, rtol=0, atol=0)
