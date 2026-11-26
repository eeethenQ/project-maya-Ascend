"""Which card a Triton kernel runs on, and the check that it is the card its operands live on.

Triton's Ascend launcher takes the launch card from ``torch.npu.current_device()``, not from the
pointers it is handed (``triton/backends/ascend/driver.py``: ``get_current_device`` is
``torch.npu.current_device()``).  Every torch operator dispatches on its *tensor's* device, so the
current card and the tensor card can differ without anything else noticing - e.g. the expert tier's
split makes ``x.to("npu:3")`` and the last card a loader selected stays current.  A kernel launched
that way dereferences card 3's virtual addresses from card 0 (or the reverse), which the device
reports as ``MTE accesses an invalid GM address or the cross-device memory access times out`` - an
async fault that surfaces at whatever operator happens to be running when it is noticed.

`operand_device` refuses such a call, with the tensors named; `select` makes the operands' card
current for the launch and restores the caller's card afterwards.  The wrappers use both, so they are
correct for any caller, not only for the one that happened to leave the right card current.
"""
from __future__ import annotations

import contextlib

import torch


def operand_device(op: str, **tensors: "torch.Tensor | None") -> torch.device:
    """The one card every named operand is on, or an AssertionError naming the tensors that disagree."""
    named = [(name, t) for name, t in tensors.items() if isinstance(t, torch.Tensor)]
    if not named:
        raise ValueError(f"{op}: needs at least one tensor operand")
    device = named[0][1].device
    wrong = [(name, t.device) for name, t in named[1:] if t.device != device]
    if wrong:
        seen = ", ".join(f"{name}={t.device}" for name, t in named)
        raise AssertionError(
            f"{op}: operands on different cards: {seen}.  A Triton kernel dereferences raw pointers "
            f"and cannot span cards; place the operands on one card before the call")
    return device


@contextlib.contextmanager
def select(device: torch.device):
    """Make ``device`` the current NPU card for the launch, restoring the caller's card afterwards."""
    if device.type != "npu" or not hasattr(torch, "npu"):
        yield
        return
    index = 0 if device.index is None else int(device.index)
    previous = int(torch.npu.current_device())
    if previous == index:
        yield
        return
    torch.npu.set_device(index)
    try:
        yield
    finally:
        torch.npu.set_device(previous)
