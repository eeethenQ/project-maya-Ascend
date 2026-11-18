"""ascend/dsv41/graph_runner.py - capture one fixed sequence of NPU ops, replay it many times.

The engine makes 32 242 operator calls a decode step and the card is busy for only 164 ms of the
1 661 ms.  A captured sequence removes the per-op dispatcher toll (measured here: 47.4 us eager,
1.88 us in a graph), but this stack fails *silently* in ways that turn a wrong graph into plausible
numbers rather than an exception.  This wrapper is deliberately a little paranoid:

* **The input contract.**  Every tensor the caller declares as an input is recorded at capture time
  with its shape, dtype, device and ``data_ptr``.  Every replay checks those four things *before*
  launching anything and raises :class:`GraphInputContractError` naming the tensor.  This is what turns
  "a shape change replays the old shape" into an immediate error, and what catches a buffer that was
  reallocated under a graph that still reads the capture-time address.  The contract only covers the
  declared tensor objects; the callable must close over those same fixed buffers.  A wrapper cannot see
  a caller that rebinds a closed-over name to a different tensor, so the engine's capture code must keep
  the buffers it declared.
* **The did-it-capture guard.**  The stack was measured to capture nothing while reporting success.
  ``verify=True`` runs the callable eagerly at two different input values (same shapes, same
  addresses), requires the two eager results to differ, then captures and replays at both values and
  requires each replay to match its eager result *bit for bit*.  That last comparison, not merely
  "A differs from B", is what catches a host value silently baked into the capture: a baked value
  gives a replay that is plausible but not equal to eager for the changed input.  The guard costs two
  eager runs and two replays, so it is **off by default** (``verify=False``).  It varies the declared
  *tensor* inputs; a host scalar that is constant in the callable's closure is outside what it can
  change, which is why the engine must address state with device tensors (``index_copy_``) rather than
  with a capture-time Python index.
* **One pool handle.**  Every graph captured by one runner is passed the same
  ``torch.npu.graph_pool_handle()``; the vendor does this too and 40 graphs captured in 0.25 s in the
  probe.  ``close`` resets the graphs and drops the handle.

Capture does **not** execute the callable's device work: the first ``replay`` is the first execution.
A captured-but-never-replayed layer therefore produces nothing, which is why the engine must replay
each graph before its output is read.  ``capture`` returns ``None``; its verdict is available from
``report(name)``, and the tensors the callable returned are available from ``outputs(name)``.

``device="cpu"`` is an honest fallback, not a fake graph: ``capture`` stores the callable and
``replay`` calls it eagerly.  No capture is attempted, no graph is claimed, and the report says
``mode="fallback"``.  That is why the unit tests can run without a card.  ``warmup`` is a no-op on CPU
because there is no capture to warm up; the callable first runs at ``replay``.

Nothing here imports ``torch_npu`` unless the device is an NPU, and the module itself never touches a
card at import time.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

import torch

_CAPTURE_MODES = ("global", "thread_local", "relaxed")


class GraphRunnerError(RuntimeError):
    """A capture, replay, or verification failure that this wrapper can name."""


class GraphInputContractError(GraphRunnerError):
    """A declared input no longer has the shape/dtype/device/address it had at capture time."""


class CaptureVerificationError(GraphRunnerError):
    """The did-it-capture guard ran and did not prove the graph it was asked to prove."""

    def __init__(self, report: "CaptureReport"):
        self.report = report
        failures = [c for c in report.checks if not c.ok]
        detail = "; ".join(f"{c.name}: {c.detail}" for c in failures) or report.detail
        super().__init__(f"capture verification failed for {report.name!r}: {detail}")


@dataclass(frozen=True)
class InputSpec:
    """The capture-time contract for one caller-declared input tensor."""

    label: str
    tensor: torch.Tensor
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: str
    data_ptr: int

    def problems(self) -> list[str]:
        """What changed since capture, in the order a human wants to read it."""
        t = self.tensor
        got_shape = tuple(t.shape)
        got_dtype = t.dtype
        got_device = str(t.device)
        got_ptr = int(t.data_ptr())
        out: list[str] = []
        if got_shape != self.shape:
            out.append(f"shape {self.shape} -> {got_shape}")
        if got_dtype != self.dtype:
            out.append(f"dtype {self.dtype} -> {got_dtype}")
        if got_device != self.device:
            out.append(f"device {self.device} -> {got_device}")
        if got_ptr != self.data_ptr:
            out.append(f"data_ptr 0x{self.data_ptr:x} -> 0x{got_ptr:x}")
        return out


@dataclass(frozen=True)
class Check:
    """One line of the guard's verdict."""

    name: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class CaptureReport:
    """What ``capture`` actually did, and what the guard (if any) proved.

    ``mode`` is ``"graph"`` on an NPU and ``"fallback"`` on CPU.  ``verified`` is true only when an
    NPU graph was captured *and* the guard proved it reads live input values; a CPU fallback is never
    reported as a verified graph.  ``ok`` is the verdict.
    """

    name: str
    mode: str
    verified: bool
    ok: bool
    checks: tuple[Check, ...]
    detail: str

    def __str__(self) -> str:
        lines = [f"{self.name}: mode={self.mode} verified={self.verified} ok={self.ok}"]
        for check in self.checks:
            lines.append(f"  [{'pass' if check.ok else 'FAIL'}] {check.name}: {check.detail}")
        if self.detail:
            lines.append(f"  {self.detail}")
        return "\n".join(lines)


@dataclass
class _Entry:
    name: str
    fn: Callable[[], Any]
    inputs: tuple[InputSpec, ...]
    graph: Any = None
    outputs: Any = None
    report: CaptureReport | None = None


class GraphRunner:
    """Capture/replay fixed-shape op sequences on one NPU device, with a loud input contract.

    Parameters
    ----------
    device:
        ``"cpu"`` for the eager fallback, or ``"npu"`` / ``"npu:N"``.  On NPU the runner sets the
        current device to ``N`` before it asks for the shared pool handle.
    capture_error_mode:
        Passed to ``torch.npu.graph`` (``aclmdlRICaptureMode``): ``"global"``, ``"thread_local"`` or
        ``"relaxed"``.  The default is ``"relaxed"`` because the expert streamer's I/O threads are
        live during capture; ``global`` errors on device work from another thread.
    """

    def __init__(self, device: str, capture_error_mode: str = "relaxed"):
        if not isinstance(device, str) or not device:
            raise ValueError("device must be 'cpu' or 'npu[:index]'")
        if capture_error_mode not in _CAPTURE_MODES:
            raise ValueError(
                f"capture_error_mode must be one of {_CAPTURE_MODES}, got {capture_error_mode!r}")
        self.device = device
        self.capture_error_mode = capture_error_mode
        self._is_npu = device == "npu" or device.startswith("npu:")
        if not self._is_npu and device != "cpu":
            raise ValueError(f"unsupported device {device!r}; use 'cpu' or 'npu[:index]'")
        self._closed = False
        self._graphs: OrderedDict[str, _Entry] = OrderedDict()
        self._torch_npu: Any = None
        self._pool: Any = None
        self._capture_stream: Any = None
        if self._is_npu:
            index = self._npu_index(device)
            self._torch_npu = self._require_npu()
            torch.npu.set_device(index)
            # A capture stream of this card's own.  `torch.npu.graph` otherwise captures on
            # `torch.npu.graph.default_capture_stream`, a *class* attribute created once on whatever
            # device was current at the first capture in the process - so every later capture on another
            # card records an empty graph on the first card's stream: it reports a successful capture and
            # then replays without executing.  A process that decodes on one card is unaffected, which is
            # why the single-card path never saw it.  An explicit stream also has to be the *same* stream
            # for every capture that shares a pool.
            self._capture_stream = torch.npu.Stream()
            # One pool for every graph this runner captures, exactly as the vendor's acl_graph.py does.
            self._pool = torch.npu.graph_pool_handle()

    # -- small public surface ----------------------------------------------------------------
    @property
    def names(self) -> list[str]:
        """The captured names, in capture order."""
        return list(self._graphs)

    @property
    def pool(self) -> Any:
        """The shared opaque pool token, or ``None`` on CPU and after ``close``."""
        return self._pool

    @property
    def closed(self) -> bool:
        return self._closed

    def report(self, name: str) -> CaptureReport | None:
        """The verdict ``capture`` recorded for ``name`` (see :class:`CaptureReport`)."""
        return self._entry(name).report

    def outputs(self, name: str) -> Any:
        """What the callable returned: at capture on NPU, or from the last replay on CPU.

        On NPU the graph writes into the same buffers the callable allocated during capture, so the
        object returned here is updated by every ``replay``.  On CPU each replay is an eager call and
        this is that call's return value.
        """
        return self._entry(name).outputs

    # -- capture / replay --------------------------------------------------------------------
    def capture(self, name: str, fn: Callable[[], Any], warmup: bool = True, *,
                inputs: "Mapping[str, torch.Tensor] | torch.Tensor | list[torch.Tensor] | None" = None,
                verify: bool = False,
                perturb: "Callable[[Mapping[str, torch.Tensor]], None] | None" = None) -> None:
        """Store ``fn`` as a graph called ``name``.

        Parameters
        ----------
        name:
            Unique within this runner.  Re-capturing an existing name is refused rather than leaking
            a graph and silently replacing a live one.
        fn:
            A zero-argument callable closing over fixed, preallocated tensors.  It must not call
            ``.item()``/``.tolist()``/``.cpu()`` or use a host int as a shape or slice index inside
            the captured region; those either raise or bake the capture-time value.
        warmup:
            Run ``fn`` once eagerly before capturing (NPU only), to trigger lazy materialisation and
            allocator growth outside the capture.  Ignored when ``verify=True`` because the guard's
            two eager runs are the warmup, and ignored on CPU because there is no capture.
        inputs:
            The tensors ``fn`` reads.  A mapping is best because its keys name the tensors in error
            messages; a single tensor or a sequence is also accepted.  This is the replay contract.
        verify:
            Run the did-it-capture guard (two eager values, two replays, bitwise equal to eager).
            Off by default because it adds two eager runs and two replays to start-up.  On CPU it
            reports the fallback and never claims a graph.
        perturb:
            Optional in-place value changer used by the guard; it receives the mapping and must keep
            every tensor's shape, dtype, device and address.  The default is a deterministic numeric
            change that is only appropriate for ordinary numeric buffers; index tensors should pass
            their own ``perturb`` (for example, an index tensor needs values that stay in range).
        """
        self._check_open()
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if name in self._graphs:
            raise ValueError(f"a graph named {name!r} is already captured; close it or use another name")
        if not callable(fn):
            raise TypeError("fn must be callable")
        if verify and perturb is not None and not callable(perturb):
            raise TypeError("perturb must be callable or None")

        specs = self._record_inputs(inputs)
        if verify and not specs:
            raise ValueError(
                "verify=True needs at least one declared input: the guard changes an input value in "
                "place and requires the output to follow, and cannot prove anything without one")

        if not self._is_npu:
            report = self._capture_cpu(name, fn, specs, verify, perturb)
            self._graphs[name] = _Entry(name, fn, specs, graph=None, outputs=None, report=report)
            return

        self._capture_npu(name, fn, specs, warmup, verify, perturb)

    def replay(self, name: str, sync: bool = False) -> None:
        """Launch the graph (or call the fallback).  Does not synchronise unless ``sync=True``.

        The input contract is checked first.  A shape/dtype/address change raises before any work is
        launched.  On CPU ``sync`` is a documented no-op: there is no device queue to drain.
        """
        entry = self._entry(name)
        self._check_contract(entry)
        if self._is_npu:
            entry.graph.replay()
            if sync:
                torch.npu.synchronize()
        else:
            entry.outputs = entry.fn()

    def replay_all(self, names: "list[str] | tuple[str, ...]", sync: bool = False) -> None:
        """Replay several graphs in order, validating every contract before launching any.

        Validating first matters: a half-replayed layer is worse than a refused one.  The replays are
        enqueued back to back; ``sync=True`` drains them once at the end, not between them.
        """
        self._check_open()
        if isinstance(names, str):
            raise TypeError("names must be an iterable of graph names, not a single string")
        entries = [self._entry(n) for n in names]
        for entry in entries:
            self._check_contract(entry)
        for entry in entries:
            if self._is_npu:
                entry.graph.replay()
            else:
                entry.outputs = entry.fn()
        if sync and self._is_npu:
            torch.npu.synchronize()

    def close(self) -> None:
        """Reset every graph and drop the shared pool handle.  Idempotent.

        The opaque pool token has no explicit ``free`` in this API; resetting the graphs and dropping
        the last references is what releases it.  ``close`` drains the device first so a graph is not
        reset out from under an in-flight replay.  A second ``close`` does nothing.
        """
        if self._closed:
            return
        self._closed = True
        if self._is_npu:
            try:
                torch.npu.synchronize()
            except Exception:                              # noqa: BLE001 - teardown must not mask the reason
                pass
        for entry in self._graphs.values():
            if entry.graph is not None:
                _safe_reset(entry.graph)
                entry.graph = None
            entry.outputs = None
        self._graphs.clear()
        self._pool = None
        self._torch_npu = None

    def __enter__(self) -> "GraphRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # -- internals ---------------------------------------------------------------------------
    def _check_open(self) -> None:
        if self._closed:
            raise GraphRunnerError("GraphRunner is closed")

    def _entry(self, name: str) -> _Entry:
        self._check_open()
        try:
            return self._graphs[name]
        except KeyError:
            raise KeyError(f"no graph named {name!r}; captured: {self.names}") from None

    def _check_contract(self, entry: _Entry) -> None:
        problems: list[str] = []
        for spec in entry.inputs:
            for problem in spec.problems():
                problems.append(f"input {spec.label!r}: {problem}")
        if problems:
            raise GraphInputContractError(
                f"graph {entry.name!r} input contract broken at replay: " + "; ".join(problems) +
                ". The graph reads the capture-time addresses; replay only through fixed, "
                "preallocated buffers of the same shape, dtype and device")

    def _graph_kwargs(self) -> dict:
        """What every capture of this runner passes to `torch.npu.graph`.

        ``stream`` is always explicit and always this runner's own capture stream: the default one is a
        process-global created on the first capturing device (see ``__init__``), and ``pool`` is the
        runner's one shared pool.
        """
        return {"capture_error_mode": self.capture_error_mode, "stream": self._capture_stream,
                "pool": self._pool}

    def _record_inputs(self, inputs) -> tuple[InputSpec, ...]:
        if inputs is None:
            items: list[tuple[Any, Any]] = []
        elif isinstance(inputs, torch.Tensor):
            items = [("input", inputs)]
        elif isinstance(inputs, Mapping):
            items = list(inputs.items())
        else:
            try:
                items = list(enumerate(inputs))
            except TypeError:
                raise TypeError(
                    "inputs must be a tensor, a mapping of name -> tensor, a sequence of tensors, "
                    "or None") from None
        specs: list[InputSpec] = []
        for label, tensor in items:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"declared input {label!r} is {type(tensor).__name__}, not a torch.Tensor")
            specs.append(InputSpec(
                label=str(label),
                tensor=tensor,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device=str(tensor.device),
                data_ptr=int(tensor.data_ptr()),
            ))
        return tuple(specs)

    def _capture_cpu(self, name: str, fn: Callable[[], Any], specs: tuple[InputSpec, ...],
                     verify: bool, perturb) -> CaptureReport:
        if not verify:
            return CaptureReport(
                name=name, mode="fallback", verified=False, ok=True, checks=(),
                detail="cpu fallback: capture stores the callable and replay calls it eagerly; no NPU "
                       "graph is captured or claimed. warmup is not run on cpu.")

        input_map = OrderedDict((s.label, s.tensor) for s in specs)
        checks: list[Check] = []
        original = _snapshot_inputs(input_map)
        try:
            eager_a = _snapshot_outputs(fn())
            _perturb_inputs(input_map, perturb)
            eager_b = _snapshot_outputs(fn())
        finally:
            _restore_inputs(input_map, original)
        checks.append(_check_differs("eager inputs change output", eager_a, eager_b))

        try:
            replay_a = _snapshot_outputs(fn())
            _perturb_inputs(input_map, perturb)
            replay_b = _snapshot_outputs(fn())
        finally:
            _restore_inputs(input_map, original)
        checks.append(_check_equal("replay A equals eager A", replay_a, eager_a))
        checks.append(_check_equal("replay B equals eager B", replay_b, eager_b))
        checks.append(_check_differs("replay B differs from replay A", replay_b, replay_a))

        ok = all(c.ok for c in checks)
        report = CaptureReport(
            name=name, mode="fallback", verified=False, ok=ok, checks=tuple(checks),
            detail="cpu fallback: replay calls the callable directly, so the guard ran on the "
                   "fallback path only; it cannot prove that a graph was captured because no graph "
                   "exists on this device.")
        if not ok:
            raise CaptureVerificationError(report)
        return report

    def _capture_npu(self, name: str, fn: Callable[[], Any], specs: tuple[InputSpec, ...],
                     warmup: bool, verify: bool, perturb) -> None:
        input_map = OrderedDict((s.label, s.tensor) for s in specs)
        graph = None
        outputs = None
        checks: tuple[Check, ...] = ()
        try:
            graph = torch.npu.NPUGraph()
            if verify:
                outputs, checks = self._verify_npu(graph, fn, input_map, perturb)
            else:
                if warmup:
                    fn()
                with torch.npu.graph(graph, **self._graph_kwargs()):
                    outputs = fn()
                torch.npu.synchronize()
        except CaptureVerificationError:
            _safe_reset(graph)
            raise
        except Exception as e:                             # noqa: BLE001 - add the graph's name, keep the cause
            _safe_reset(graph)
            raise GraphRunnerError(
                f"capture {name!r} failed on {self.device}: {type(e).__name__}: {e}") from e

        if verify:
            ok = all(c.ok for c in checks)
            report = CaptureReport(
                name=name, mode="graph", verified=True, ok=ok, checks=checks,
                detail="the did-it-capture guard ran: two eager values differed, and two replays at "
                       "the same addresses matched their eager results bit for bit.")
            if not ok:
                _safe_reset(graph)
                raise CaptureVerificationError(report)
        else:
            report = CaptureReport(
                name=name, mode="graph", verified=False, ok=True, checks=(),
                detail="captured with verify=False: the did-it-capture guard was not run.")
        self._graphs[name] = _Entry(name, fn, specs, graph=graph, outputs=outputs, report=report)

    def _verify_npu(self, graph, fn: Callable[[], Any], inputs: "OrderedDict[str, torch.Tensor]",
                    perturb) -> tuple[Any, tuple[Check, ...]]:
        checks: list[Check] = []
        original = _snapshot_inputs(inputs)
        try:
            eager_a = _snapshot_outputs(fn())
            _perturb_inputs(inputs, perturb)
            eager_b = _snapshot_outputs(fn())
        finally:
            _restore_inputs(inputs, original)
        checks.append(_check_differs("eager inputs change output", eager_a, eager_b))

        # Capture with the original values restored: a host value the callable derives from an input
        # is therefore baked at A.  The replay at B below is what exposes that.
        with torch.npu.graph(graph, **self._graph_kwargs()):
            outputs = fn()
        torch.npu.synchronize()

        graph.replay()
        replay_a = _snapshot_outputs(outputs)
        try:
            _perturb_inputs(inputs, perturb)
            graph.replay()
            replay_b = _snapshot_outputs(outputs)
        finally:
            _restore_inputs(inputs, original)

        checks.append(_check_equal("replay A equals eager A", replay_a, eager_a))
        checks.append(_check_equal("replay B equals eager B", replay_b, eager_b))
        checks.append(_check_differs("replay B differs from replay A", replay_b, replay_a))
        return outputs, tuple(checks)

    @staticmethod
    def _npu_index(device: str) -> int:
        parts = device.split(":")
        if len(parts) == 1:
            return 0
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"invalid NPU device {device!r}; expected 'npu' or 'npu:N'")
        return int(parts[1])

    @staticmethod
    def _require_npu():
        try:
            import torch_npu
        except Exception as e:                             # noqa: BLE001 - the real reason is chained
            raise GraphRunnerError(
                "device is an NPU but torch_npu could not be imported; install torch_npu or use "
                "GraphRunner('cpu') for the eager fallback") from e
        if not hasattr(torch, "npu"):
            raise GraphRunnerError(
                "torch_npu imported but torch.npu is not registered; graph capture is unavailable")
        for attr in ("graph", "NPUGraph", "graph_pool_handle"):
            if not hasattr(torch.npu, attr):
                raise GraphRunnerError(
                    f"torch_npu {getattr(torch_npu, '__version__', '?')} has no torch.npu.{attr}; "
                    "graph capture is not available on this stack")
        if not torch.npu.is_available():
            raise GraphRunnerError("torch_npu is installed but no Ascend device is available")
        return torch_npu


# --------------------------------------------------------------------------- guard helpers
def _tensor_leaves(value: Any, prefix: str = "out"):
    """Yield ``(label, tensor)`` for the tensors a callable returned, in a stable order."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _tensor_leaves(item, f"{prefix}.{key}")
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            yield from _tensor_leaves(item, f"{prefix}[{i}]")


def _snapshot_outputs(value: Any) -> "OrderedDict[str, torch.Tensor]":
    """Copy every returned tensor to the host so the next call cannot overwrite the evidence."""
    return OrderedDict(
        (label, tensor.detach().to("cpu", copy=True))
        for label, tensor in _tensor_leaves(value)
    )


def _snapshot_inputs(inputs: "OrderedDict[str, torch.Tensor]") -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((label, tensor.detach().clone()) for label, tensor in inputs.items())


def _restore_inputs(inputs: "OrderedDict[str, torch.Tensor]",
                    snapshots: "OrderedDict[str, torch.Tensor]") -> None:
    for label, tensor in inputs.items():
        tensor.copy_(snapshots[label])


def _perturb_inputs(inputs: "OrderedDict[str, torch.Tensor]", perturb) -> None:
    """Change input *values* in place; refuse anything that changes the declared identity."""
    before = {
        label: (tuple(tensor.shape), tensor.dtype, str(tensor.device), int(tensor.data_ptr()))
        for label, tensor in inputs.items()
    }
    if perturb is None:
        _default_perturb(inputs)
    else:
        perturb(inputs)
    for label, tensor in inputs.items():
        got = (tuple(tensor.shape), tensor.dtype, str(tensor.device), int(tensor.data_ptr()))
        if got != before[label]:
            raise GraphRunnerError(
                f"perturb changed the identity of input {label!r} (shape/dtype/device/address); "
                "it must only change values in place, so the guard tests the same addresses")


def _default_perturb(inputs: "OrderedDict[str, torch.Tensor]") -> None:
    """A deterministic value change that keeps shape/dtype/address.

    It is safe for ordinary numeric buffers, not for every semantic input: an index tensor needs its
    own ``perturb`` to stay in range.  The guard is optional for exactly this reason.
    """
    for label, tensor in inputs.items():
        if tensor.is_floating_point() or tensor.is_complex():
            # mul then add changes a zero buffer and flips the sign of a large one
            tensor.mul_(-0.5).add_(0.125)
        elif tensor.dtype == torch.bool:
            tensor.logical_not_()
        elif tensor.dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
            tensor.bitwise_xor_(1)
        else:
            raise GraphRunnerError(
                f"no default in-place perturbation for input {label!r} with dtype {tensor.dtype}; "
                "pass perturb=... to capture(...)")


def _compare(a: "OrderedDict[str, torch.Tensor]", b: "OrderedDict[str, torch.Tensor]"
             ) -> tuple[bool, str]:
    if not a and not b:
        return True, "the callable returned no tensors"
    if list(a) != list(b):
        return False, f"different tensor sets: {list(a)} vs {list(b)}"
    for label in a:
        x, y = a[label], b[label]
        if tuple(x.shape) != tuple(y.shape):
            return False, f"{label}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        if x.dtype != y.dtype:
            return False, f"{label}: dtype {x.dtype} vs {y.dtype}"
        if not torch.equal(x, y):
            try:
                delta = (x.float() - y.float()).abs().max().item()
                detail = f"{label}: values differ (max |delta| {delta:g})"
            except Exception:                              # noqa: BLE001 - non-float dtypes
                detail = f"{label}: values differ"
            return False, detail
    return True, f"{len(a)} tensor(s) bit-identical"


def _check_equal(name: str, a: "OrderedDict[str, torch.Tensor]",
                 b: "OrderedDict[str, torch.Tensor]") -> Check:
    if not a:
        return Check(name, False, "the callable returned no tensors, so there is nothing to verify")
    same, detail = _compare(a, b)
    return Check(name, same, detail)


def _check_differs(name: str, a: "OrderedDict[str, torch.Tensor]",
                   b: "OrderedDict[str, torch.Tensor]") -> Check:
    if not a:
        return Check(name, False, "the callable returned no tensors, so there is nothing to verify")
    same, detail = _compare(a, b)
    if same:
        return Check(name, False, f"outputs did not change ({detail}); the capture would prove nothing")
    return Check(name, True, f"outputs changed ({detail})")


def _safe_reset(graph: Any) -> None:
    if graph is None:
        return
    try:
        graph.reset()
    except Exception:                                  # noqa: BLE001 - the original error is more useful
        pass
