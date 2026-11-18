"""ascend/dsv41/kv_planes.py - one conversation's KV planes, addressed by device-side positions.

:class:`ascend.dsv41.model.State` keeps exactly these planes, but every access to them is a host slice:

* ``state.swa[li][:, host_positions[t]] = kv[t]`` writes one window row at a host integer, and
* ``state.swa[li][:, w0:p + 1]`` reads a slice whose *width grows with the conversation* (one row at the
  first token, `sliding_window` rows from token 128 on).

Neither survives capture.  A captured graph bakes the host integer it saw, so a replay keeps writing the
capture-time position (docs/ascend-notes/opt-4-graph.md §2.5, probe D: silently wrong KV, no error), and a
replay does not adopt a grown width (probe F: old shape replayed, no error).  This module replaces both
with fixed-shape ops whose indices are device tensors:

    state.swa[li][:, host_positions[t]] = kv[t]     ->  planes.write_window(li, positions, kv)
    wkeys = state.swa[li][:, w0:p + 1]              ->  keys, valid = planes.read_window(li, pos)
    state.long_kv[src][:, j.clamp_min(0)]           ->  rows = planes.read_groups(src, j)
    state.comp_kv_raw[li][:, positions] = kv        ->  planes.write_raw(li, positions, kv, score)
    state.long_kv[src][:, g0:g0 + n] = lat          ->  planes.write_group(li, groups, lat)

The attribute names and layouts are `State`'s (``swa``, ``long_kv``, ``index_k``, ``comp_kv_raw``,
``comp_score_raw``, ``pos``, ``next_pos``), so replacing the state is mechanical; the dtypes are the ones
the planes already use - the window and compressed rows in the state's dtype (bf16 in the engine), the
compressor's raw rows in fp32.

The window wrap invariant
-------------------------
A read at absolute position ``p`` touches positions ``p - W + 1 .. p`` and finds them at slots
``(p - W + 1 + arange(W)) % max_context``.  That is correct, for **any** ``max_context`` - a multiple of
``W`` or not - exactly when ``W <= max_context``: every position in the span is at most ``W - 1 <=
max_context - 1`` behind ``p``, so under monotonic writes no slot of the span has been reused by a later
token.  Nothing in the read assumes ``max_context % W == 0``, and nothing subtracts without wrapping: the
index is taken per element, so a window that straddles the end of the plane reads its head from slots
``0..`` and its tail from the end, still in ascending position order.  The failure that invariant forbids is
the one a contiguous-slice implementation has for free, and ``__init__`` refuses a window wider than the
plane rather than documenting it away.

The second assumption is the one the forward pass already makes: a window is read at the position that was
just written, so its span lies within the last ``max_context`` writes.  Reading a position that later
positions have already overwritten by more than ``max_context`` tokens returns the slot's current occupant -
the ring has no way to know - which is the same limit `State` has, made explicit rather than a crash.

The compressed planes do not wrap
---------------------------------
Entry ``g`` covers tokens ``[g * r, (g + 1) * r)`` and the plane holds ``max_context // r + 1`` entries,
which is `State`'s layout (one spare over the groups a context of ``max_context`` tokens closes).  A group
index is a *bound*, not a modulo: ``0 <= g < entries`` is checked where it can be - a host integer - and is
the caller's to keep when it is a device tensor, because checking it would need a host read.  The raw rows a
compressor pools are position-addressed and *do* wrap, which is exact for a group because it is read at its
close, at most ``r - 1`` positions behind, and ``r <= max_context`` is asserted.

What is not here
----------------
``State.topk``, ``State.candidates``, ``token_ids`` and the Engram bookkeeping are the caller's; this class
owns the planes and the cursor only.  Nothing here imports the Ascend runtime, and nothing here reads a
tensor's value back to the host: every branch is on a shape or on a config integer.
"""
from __future__ import annotations

import torch

#: The compressed-KV plane, indexed exactly like `State.long_kv`.
KV = "kv"
#: The index-K plane, indexed exactly like `State.index_k` (a subset of the KV sources own one).
INDEX = "index"


class KvPlanes:
    """The window ring, the compressed caches and the compressor rows of one conversation.

    Construct it once per request, like `State`, and replace the state's planes with it.  Writes take an
    absolute position (an ``int`` in eager code, a device tensor inside a graph); reads take the position
    they are asked about and return fixed shapes, so a captured layer replays at any position.
    """

    def __init__(self, cfg, plan, device, dtype, max_context: int, window: int | None = None,
                 batch: int = 1):
        self.cfg = cfg
        self.plan = plan
        self.device = torch.device(device)
        self.dtype = dtype
        self.max_context = int(max_context)
        # the window the model reads is `sliding_window` keys *inclusive of the query*; the parameter exists
        # so a test can make the ring short, not so a caller can invent a different model
        self.window = int(cfg.sliding_window if window is None else window)
        self.batch = int(batch)
        if self.batch != 1:
            raise ValueError(f"KvPlanes addresses one conversation (batch=1), got batch={self.batch}")
        if self.max_context <= 0:
            raise ValueError(f"max_context must be positive, got {self.max_context}")
        if not 0 < self.window <= self.max_context:
            raise ValueError(
                f"window {self.window} is outside (0, max_context={self.max_context}]: a wider window wraps "
                "onto positions that are still live (see the module docstring)")
        D = int(cfg.head_dim)
        hd = int(cfg.index_head_dim)
        self.head_dim, self.index_head_dim = D, hd
        # window ring per layer: `State` allocates the same [batch, max_context, head_dim] plane, so the
        # layout - and therefore anything that inspects the state's memory - is unchanged
        self.swa = {r.layer_idx: torch.zeros(self.batch, self.max_context, D,
                                             device=self.device, dtype=dtype)
                    for r in plan.layers}
        # compressed caches: one per KV source, entry g covers tokens [g*r, g*r+r), as in `State`
        self.long_kv: dict[int, torch.Tensor] = {}
        self.index_k: dict[int, torch.Tensor] = {}
        self.comp_kv_raw: dict[int, torch.Tensor] = {}
        self.comp_score_raw: dict[int, torch.Tensor] = {}
        self.comp_closed: dict[int, int] = {}
        self.group_entries: dict[int, int] = {}
        for lid in plan.kv_source_layer_ids:
            role = plan.layer(lid)
            r = int(role.compress_ratio)
            if r < 1:
                raise ValueError(f"KV source layer {lid} has compress_ratio {r}; a source needs r >= 1")
            if r > self.max_context:
                raise ValueError(f"layer {lid} pools groups of {r} tokens over a {self.max_context}-token "
                                 "ring: a group would alias its own raw rows")
            entries = self.max_context // r + 1
            self.group_entries[lid] = entries
            self.long_kv[lid] = torch.zeros(self.batch, entries, D, device=self.device, dtype=dtype)
            if role.is_index_source:
                self.index_k[lid] = torch.zeros(self.batch, entries, hd, device=self.device, dtype=dtype)
            if r > 1:                       # the gate needs a group to pool; ratio 1 has no open group
                self.comp_kv_raw[lid] = torch.zeros(self.batch, self.max_context, D,
                                                    device=self.device, dtype=torch.float32)
                self.comp_score_raw[lid] = torch.zeros(self.batch, self.max_context, D,
                                                       device=self.device, dtype=torch.float32)
            self.comp_closed[lid] = 0       # `State` keeps it too; its forward pass never advances it today
        # the window read's only index arithmetic that is not derived from `position`: a constant, built once
        self._offsets = torch.arange(self.window, dtype=torch.long, device=self.device)
        # the cursor, as `State` keeps it: `pos` is the device mirror a graph reads, `next_pos` the host
        # mirror the forward pass branches on.  The plane methods below take explicit positions; the cursor
        # is bookkeeping for the caller, moved by `advance`/`rewind`.
        self.pos = torch.zeros(self.batch, dtype=torch.long, device=self.device)
        self.next_pos = 0
        self.tokens = 0

    # -- addressing -----------------------------------------------------------------------------
    def _positions(self, position, n: int) -> torch.Tensor:
        """`n` absolute positions as a device int64 tensor, from a host int or from a device tensor.

        A host int becomes a constant tensor: correct in eager code, but a capture bakes its value, so a
        graph must be handed the step's own position tensor instead.  This is the only place the two forms
        differ - and the reason a test can prove the tensor path is not converted back to an int.
        """
        if isinstance(position, int):
            if n == 1:
                return torch.full((1,), position, dtype=torch.long, device=self.device)
            # several rows and one int: the chunk starts at that position, exactly as `State`'s
            # `range(start, start + T)` does for a prompt
            return torch.arange(position, position + n, dtype=torch.long, device=self.device)
        t = position if isinstance(position, torch.Tensor) else torch.as_tensor(position)
        t = t.to(device=self.device, dtype=torch.long).reshape(-1)
        if t.numel() != n:
            raise ValueError(f"{n} row(s) need {n} position(s), got a tensor holding {t.numel()}")
        return t

    def _as_long(self, x) -> torch.Tensor:
        """An index tensor on this device as int64, keeping its shape (indices, not positions)."""
        t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
        if t.dtype != torch.long:
            t = t.to(dtype=torch.long)
        return t.to(device=self.device)

    @staticmethod
    def _rows(row, width: int, name: str) -> torch.Tensor:
        """``[D]`` or ``[n, D]`` as ``[n, D]``; the widths are the plane's, so a mismatch fails here."""
        t = row if isinstance(row, torch.Tensor) else torch.as_tensor(row)
        if t.dim() not in (1, 2) or t.shape[-1] != width:
            raise ValueError(f"{name} must be [D] or [T, D] with D={width}, got {tuple(t.shape)}")
        return t.reshape(-1, width)

    def _group_plane(self, source_layer: int, which: str) -> torch.Tensor:
        if which == KV:
            planes = self.long_kv
        elif which == INDEX:
            planes = self.index_k
        else:
            raise ValueError(f"plane must be {KV!r} or {INDEX!r}, got {which!r}")
        if source_layer not in planes:
            raise KeyError(f"layer {source_layer} has no {which} plane; "
                           f"{which} sources are {sorted(planes)}")
        return planes[source_layer]

    def _raw_plane(self, layer: int, planes: dict[int, torch.Tensor], name: str) -> torch.Tensor:
        if layer not in planes:
            raise KeyError(f"layer {layer} has no compressor raw {name} rows (compress_ratio 1 has no group)")
        return planes[layer]

    def _group_indices(self, source_layer: int, group, n: int) -> torch.Tensor:
        """`n` group indices as a device int64 tensor; a host int is bounds-checked here, a tensor is not."""
        capacity = self.group_entries[source_layer]
        if isinstance(group, int):
            if n == 1:
                if not 0 <= group < capacity:
                    raise ValueError(f"group {group} is outside [0, {capacity}) on layer {source_layer}")
                return torch.full((1,), group, dtype=torch.long, device=self.device)
            if group < 0 or group + n > capacity:
                raise ValueError(f"groups [{group}, {group + n}) are outside [0, {capacity}) on layer "
                                 f"{source_layer}")
            return torch.arange(group, group + n, dtype=torch.long, device=self.device)
        g = group if isinstance(group, torch.Tensor) else torch.as_tensor(group)
        g = g.to(device=self.device, dtype=torch.long).reshape(-1)
        if g.numel() != n:
            raise ValueError(f"{n} row(s) need {n} group(s), got a tensor holding {g.numel()}")
        # no host read here: a device-side bound check would be data-dependent control flow, which is
        # exactly what a graph cannot replay.  `read_groups`' contract - clamp -1 and mask - applies.
        return g

    # -- the window ring ------------------------------------------------------------------------
    def write_window(self, layer: int, position, row) -> None:
        """Write the chunk's kv rows at their absolute positions, in the state's dtype as `State` does.

        ``row`` is one ``[D]`` row or ``[T, D]`` rows and ``position`` the matching position(s).  One
        `index_copy_` with device-side indices replaces the per-token host slice: its shapes depend only on
        how many rows a chunk has, never on where they land in the conversation.
        """
        plane = self.swa[layer]
        rows = self._rows(row, plane.shape[-1], "row")
        idx = self._positions(position, rows.shape[0]) % self.max_context
        plane[0].index_copy_(0, idx, rows.to(plane.dtype))

    def read_window(self, layer: int, position) -> tuple[torch.Tensor, torch.Tensor]:
        """The ``window`` keys ending at ``position`` (inclusive), ascending, plus which ones exist.

        Always exactly ``window`` rows, whatever the conversation's length: the read gathers
        ``(position - window + 1 + arange(window)) % max_context`` and marks a position ``< 0`` invalid
        rather than dropping it, so the shape a graph captured at one token is the shape it replays at
        100 000.  ``keys`` is the plane's dtype; ``valid`` is the mask `sparse_attn` wants, and the caller
        reproduces today's ``wkeys = swa[:, w0:p + 1]`` as ``keys[valid]`` only in code that may take a host
        branch - inside a graph it passes ``keys`` and ``valid`` straight to `sparse_attn`, whose
        ``masked_fill(~valid, -inf)`` gives the same softmax.

        The content of an invalid row is *unspecified* (it is the slot the modulo landed on, zeros while
        the plane is cold); it is masked, never meant to be read, and not cleared - clearing would add an
        op per read for a value no correct caller consumes.
        """
        plane = self.swa[layer]
        p = self._positions(position, 1)                        # [1]
        positions = p - (self.window - 1) + self._offsets       # [window], ascending
        valid = positions >= 0
        keys = plane[0].index_select(0, positions.clamp_min(0) % self.max_context)
        return keys, valid

    # -- the compressed planes ------------------------------------------------------------------
    def write_group(self, source_layer: int, group, row, plane: str = KV) -> None:
        """Write compressed entries at their group indices; no wrap, the bound is the plane's capacity.

        ``group`` is a host int or a device tensor of group indices, one per row of ``row``; a chunk that
        closes several groups writes ``groups = g0 + torch.arange(n, device=...)`` and its ``[n, D]``
        latents in one call (today's ``long_kv[:, g0:g0 + n] = lat``).  ``plane`` selects the compressed-KV
        plane (``KV``, width ``head_dim``) or the index-K plane (``INDEX``, width ``index_head_dim``); a host
        index outside ``[0, group_entries[source_layer])`` raises here, while a tensor index is the caller's
        to bound - see the module docstring.
        """
        tensor = self._group_plane(source_layer, plane)
        rows = self._rows(row, tensor.shape[-1], "row")
        idx = self._group_indices(source_layer, group, rows.shape[0])
        tensor[0].index_copy_(0, idx, rows.to(tensor.dtype))

    def read_groups(self, source_layer: int, indices: torch.Tensor, plane: str = KV) -> torch.Tensor:
        """The rows the given compressed entries name, with ``-1`` padding folded onto entry 0.

        ``-1`` is the indexer's "no entry" (``torch.where(idx < lens, idx, -1)``); today's attention
        gathers it as ``j.clamp_min(0)`` and drops it from the mask as ``j >= 0``, and this does the same:
        the returned rows have shape ``(*indices.shape, D)``, entry 0's row sits in the padded slots, and
        `group_valid` returns the mask the caller must use.  A row is read per position, not per layer, so
        the batch axis `State` carries is dropped (batch is 1).
        """
        tensor = self._group_plane(source_layer, plane)
        idx = self._as_long(indices)
        rows = tensor[0].index_select(0, idx.reshape(-1).clamp_min(0))
        return rows.reshape(*idx.shape, tensor.shape[-1])

    def group_valid(self, indices: torch.Tensor) -> torch.Tensor:
        """Which entries of ``indices`` are real: ``indices >= 0``, `-1` padding is not."""
        return self._as_long(indices) >= 0

    # -- the compressor's raw rows --------------------------------------------------------------
    def write_raw(self, layer: int, position, kv, score) -> None:
        """The compressor's per-token raw rows (fp32, as `State` keeps them) at absolute positions.

        These are the rows a compressor pools over one group; they are position-addressed and wrapped by
        ``% max_context``, which is exact because a group is read at its close, at most ``r - 1`` positions
        behind its last row, and ``r <= max_context`` is asserted in ``__init__``.  `State` writes the same
        rows at the same absolute positions; the wrap only decides what happens past ``max_context``, where
        `State`'s plane would be indexed out of bounds.
        """
        kv_plane = self._raw_plane(layer, self.comp_kv_raw, "kv")
        score_plane = self._raw_plane(layer, self.comp_score_raw, "score")
        kv_rows = self._rows(kv, kv_plane.shape[-1], "kv")
        score_rows = self._rows(score, score_plane.shape[-1], "score")
        if kv_rows.shape[0] != score_rows.shape[0]:
            raise ValueError(f"kv and score must cover the same positions, got {kv_rows.shape[0]} and "
                             f"{score_rows.shape[0]} rows")
        idx = self._positions(position, kv_rows.shape[0]) % self.max_context
        kv_plane[0].index_copy_(0, idx, kv_rows.to(kv_plane.dtype))
        score_plane[0].index_copy_(0, idx, score_rows.to(score_plane.dtype))

    def read_raw(self, layer: int, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The raw rows at ``positions``, both planes, as ``(*positions.shape, D)`` fp32 tensors.

        A group being closed is the ``r`` consecutive positions ``g * r + arange(r)``::

            kvc, sc = planes.read_raw(layer, g * r + torch.arange(r, device=...))

        which is `State`'s ``comp_kv_raw[:, js]`` / ``comp_score_raw[:, js]`` without the batch axis; the
        compressor pools ``(kvc * sc.softmax(0)).sum(0)`` per dimension, `State`'s ``softmax(1)`` on the
        batched form.
        """
        idx = self._as_long(positions) % self.max_context
        out = []
        for name, planes in (("kv", self.comp_kv_raw), ("score", self.comp_score_raw)):
            tensor = self._raw_plane(layer, planes, name)
            out.append(tensor[0].index_select(0, idx.reshape(-1)).reshape(*idx.shape, tensor.shape[-1]))
        return out[0], out[1]

    # -- the cursor -----------------------------------------------------------------------------
    def context(self) -> int:
        return self.next_pos

    def entries(self, layer: int, position: int | None = None) -> int:
        """How many compressed entries a query at ``position`` can see (`State.entries`, kept for parity)."""
        r = self.plan.layer(layer).compress_ratio
        if not r:
            return 0
        return ((self.next_pos if position is None else position) + 1) // r

    def advance(self, n: int = 1) -> None:
        """Move the cursor ``n`` positions forward: what `Dsv41.forward` does to `State` after a chunk."""
        n = int(n)
        if n < 0:
            raise ValueError(f"cannot advance {n} tokens")
        self.next_pos += n
        self.tokens += n
        self.pos += n

    def rewind(self, n: int = 1) -> None:
        """Step back ``n`` tokens, leaving the planes untouched exactly as `State.rewind` does.

        Every read is addressed by absolute position and every row is written before it is read, so moving
        the cursor is enough: the next pass rewrites the positions past it.  A read of a position past the
        cursor sees the previous pass's row (or zeros), which is the contract `State` has today - rewinding
        is not a rollback of the plane, it is a statement that the next chunk starts earlier.
        """
        n = int(n)
        if n < 0 or n > self.next_pos:
            raise ValueError(f"cannot rewind {n} of {self.next_pos} tokens")
        self.next_pos -= n
        self.tokens -= n
        self.pos -= n
