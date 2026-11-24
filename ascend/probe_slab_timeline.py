"""ascend/probe_slab_timeline.py - the per-layer schedule of the captured decode, host and device.

The landing-zone fill is 8.5 GB of device-to-device copy a step and was priced at 16.2 ms on the step's
critical path.  A second stream could only hide it if, at the moment the layer's routing ids exist, the
device (or the host) still had *other* work in flight that the fill does not depend on.  That is a question
for a timeline, not for the call graph:

* device - a ``torch.npu.Event`` pair around each `g_pre`, the fill and `g_post`, so the per-layer device
  intervals and the **gaps between them** are measured on the card's own clock;
* host - ``perf_counter`` around the same calls plus the tier lookup, the pread, the H2D and the two
  ``cat`` calls inside ``ExpertStreamer.fetch_int8_into``, so the fill's host cost splits into view
  construction, kernel dispatch and device time;
* one anchor event per step puts both clocks on one axis.

It runs the *engine's* own ``Dsv41.forward`` and shadows the methods it calls (``replay_pre``, ``read_ids``,
``fetch``, ``replay_post`` and the streamer's ``_blocks``/``_read_many``/``fetch_int8_into``), so there is no
second implementation of the loop to go stale.  The token stream is greedy - the real serving shape - and
the tier's hit/miss bytes are reported alongside, because the 8.5 GB of D2D does not depend on the tier but
the H2D of the misses does.

    python -m ascend.probe_slab_timeline --device npu:0 --hbm-share 0.65 --steps 3

Writes JSON, a per-layer ASCII schedule and an SVG figure (``--svg``); no plotting library needed.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import Counter

import torch

from .dsv41 import runtime as rt

SLAB_BYTES = 6 * 3 * 2304 * 5120          # one layer's landing-zone fill
PER = 2304 * 5120


def sync() -> None:
    if hasattr(torch, "npu"):
        torch.npu.synchronize()


class Recorder:
    """Host timestamps, device events and per-layer accumulators for one decode step."""

    def __init__(self, layers: int, events: list) -> None:
        self.host: dict[tuple[str, int], float] = {}
        self.dev: dict[tuple[str, int], object] = {}
        self.events = events
        self.used = 0
        self.layers = layers
        self.t0 = time.perf_counter()
        self.anchor = events[0]
        self.anchor.record()
        self.used = 1
        self.cur = 0
        self.pread_ms = [0.0] * layers
        self.pread_n = [0] * layers
        self.cat_ms: dict[str, float] = {}
        self.cat_n: dict[str, int] = {}

    def mark(self, name: str, li: int | None = None, event: bool = True) -> None:
        li = self.cur if li is None else li
        self.host[(name, li)] = time.perf_counter()
        if event:
            ev = self.events[self.used]
            self.used += 1
            ev.record()
            self.dev[(name, li)] = ev

    def at(self, name: str, li: int | None = None) -> float:
        li = self.cur if li is None else li
        return (self.host[(name, li)] - self.t0) * 1e3

    def dev_ms(self, name: str, li: int) -> float:
        ev = self.dev.get((name, li))
        return self.anchor.elapsed_time(ev) if ev is not None else float("nan")


REC: Recorder | None = None


def patch(g, src) -> dict:
    """Shadow the engine's own methods with timing wrappers; returns the originals."""
    real = {"pre": g.replay_pre, "ids": g.read_ids, "fetch": g.fetch, "post": g.replay_post,
            "blocks": src._blocks, "read_many": src._read_many, "fill": src.fetch_int8_into,
            "pool": src._pool}

    def pre(li):
        REC.cur = li
        REC.mark("pre.launch", li)
        real["pre"](li)
        REC.mark("pre.done", li)

    def ids():
        REC.mark("id.start")
        out = real["ids"]()
        REC.mark("id.done")
        return out

    def fetch(li, ids_):
        REC.mark("fetch.launch", li)
        real["fetch"](li, ids_)
        REC.mark("fetch.done", li)

    def post(li):
        REC.mark("post.launch", li)
        real["post"](li)
        REC.mark("post.done", li)

    def blocks(layer, experts, insert=True):
        REC.cur = layer
        REC.mark("blocks.start", layer, event=False)
        out = real["blocks"](layer, experts, insert)
        REC.mark("blocks.done", layer, event=False)
        # the landing-zone fill starts here: the tier lookup, the pread of the misses and their H2D
        # are all finished, and the only thing left before the copy is building its 18 source views
        REC.mark("fill.start", layer)
        return out

    def read_many(layer, experts):
        REC.mark("read_many.start", layer, event=False)
        out = real["read_many"](layer, experts)
        REC.mark("read_many.done", layer, event=False)
        return out

    def fill(*a, **k):
        layer = int(a[0])
        REC.cur = layer
        real["fill"](*a, **k)
        REC.mark("fill.done", layer)

    class PoolProxy:
        """The host pread is the pool's ``map``; the rest of ``_read_many`` is the H2D and the scales."""

        def __init__(self, real_pool):
            self._real = real_pool

        def map(self, fn, it):
            t0 = time.perf_counter()
            try:
                return list(self._real.map(fn, it))
            finally:
                ms = (time.perf_counter() - t0) * 1e3
                REC.pread_ms[REC.cur] += ms
                REC.pread_n[REC.cur] += 1

        def __getattr__(self, name):
            return getattr(self._real, name)

    g.replay_pre, g.read_ids, g.fetch, g.replay_post = pre, ids, fetch, post
    src._blocks, src._read_many, src.fetch_int8_into = blocks, read_many, fill
    src._pool = PoolProxy(real["pool"])
    return real


def unpatch(g, src, real) -> None:
    g.replay_pre, g.read_ids, g.fetch, g.replay_post = real["pre"], real["ids"], real["fetch"], real["post"]
    src._blocks, src._read_many, src.fetch_int8_into = real["blocks"], real["read_many"], real["fill"]
    src._pool = real["pool"]


def cat_probe(real_cat):
    """Time the fill's two ``cat`` calls; the destinations are 212 MB int8 and 233 KB fp32."""
    def cat(tensors, *a, **k):
        t0 = time.perf_counter()
        out = real_cat(tensors, *a, **k)
        ms = (time.perf_counter() - t0) * 1e3
        if REC is not None:
            kind = "fill.cat" if out.dtype == torch.int8 else "fill.scale_cat"
            REC.cat_ms[kind] = REC.cat_ms.get(kind, 0.0) + ms
            REC.cat_n[kind] = REC.cat_n.get(kind, 0) + 1
        return out
    return cat


def layer_rows(rec: Recorder, layers: int) -> list[dict]:
    """Per-layer schedule in ms, device events on the step's anchor axis."""
    rows = []
    for li in range(layers):
        def d(name):
            return rec.dev_ms(name, li)

        def h(name):
            return rec.at(name, li)

        rm = ("read_many.done", li) in rec.host
        row = {"layer": li,
               "pre_d0": d("pre.launch"), "pre_d1": d("pre.done"),
               "fill_d0": d("fill.start"), "fill_d1": d("fill.done"),
               "post_d0": d("post.launch"), "post_d1": d("post.done"),
               "h_pre": h("pre.launch"), "h_pre_done": h("pre.done"),
               "h_id0": h("id.start"), "h_id1": h("id.done"),
               "h_blocks0": h("blocks.start"), "h_blocks1": h("blocks.done"),
               "h_rm0": h("read_many.start") if rm else h("blocks.done"),
               "h_rm1": h("read_many.done") if rm else h("blocks.done"),
               "h_fill0": h("fill.start"), "h_fill1": h("fill.done"),
               "h_post": h("post.launch"), "h_post_done": h("post.done"),
               "pread_host": rec.pread_ms[li]}
        row["pre_dev"] = row["pre_d1"] - row["pre_d0"]
        row["fill_dev"] = row["fill_d1"] - row["fill_d0"]
        row["post_dev"] = row["post_d1"] - row["post_d0"]
        row["read_wait_host"] = row["h_id1"] - row["h_id0"]
        row["blocks_host"] = row["h_blocks1"] - row["h_blocks0"]
        row["readmany_host"] = row["h_rm1"] - row["h_rm0"]
        row["h2d_host"] = max(0.0, row["readmany_host"] - row["pread_host"])
        row["fill_host"] = row["h_fill1"] - row["h_fill0"]
        row["launch_host"] = row["h_post_done"] - row["h_fill1"]
        row["cat_dev_est"] = max(0.0, row["fill_dev"] - row["fill_host"])
        row["fill_exposed"] = max(0.0, row["fill_dev"] - row["launch_host"])
        row["dev_idle_before_fill"] = row["fill_d0"] - row["pre_d1"]
        row["dev_gap_fill_post"] = row["post_d0"] - row["fill_d1"]
        rows.append(row)
    for prev, row in zip(rows, rows[1:]):
        row["dev_idle_between"] = row["pre_d0"] - prev["post_d1"]
        row["host_gap_between"] = row["h_pre"] - prev["h_post_done"]
    rows[0]["dev_idle_between"] = rows[0]["pre_d0"]
    rows[0]["host_gap_between"] = rows[0]["h_pre"]
    return rows


KEYS = ("pre_dev", "fill_dev", "cat_dev_est", "post_dev", "read_wait_host", "blocks_host", "pread_host",
        "h2d_host", "fill_host", "launch_host", "fill_exposed", "dev_idle_before_fill",
        "dev_idle_between", "host_gap_between", "dev_gap_fill_post")


def ascii_schedule(rows: list[dict], width: int, per_char: float) -> list[str]:
    """Two lines a layer - device segments over host segments - on one time axis."""
    out = []
    for row in rows:
        base = row["pre_d0"]
        line = [" "] * width
        for name, ch in (("pre", "#"), ("fill", "*"), ("post", "+")):
            a = int((row[f"{name}_d0"] - base) / per_char)
            b = int((row[f"{name}_d1"] - base) / per_char)
            for i in range(max(0, a), min(width, max(b, a + 1))):
                line[i] = ch
        out.append(f"  L{row['layer']:02d} dev  [{''.join(line)}]")
        line = [" "] * width
        segs = ((row["h_id0"], row["h_rm1"], "p"), (row["h_rm1"], row["h_blocks1"], "h"),
                (row["h_fill0"], row["h_fill1"], "f"), (row["h_fill1"], row["h_post_done"], "l"))
        for a0, a1, ch in segs:
            a = int((a0 - base) / per_char)
            b = int((a1 - base) / per_char)
            for i in range(max(0, a), min(width, max(b, a + 1))):
                line[i] = ch
        out.append(f"  L{row['layer']:02d} host [{''.join(line)}]   (p pread, h H2D, f fill issue, l launch)")
    return out


def svg_figure(rows: list[dict], path: str) -> None:
    """A Gantt: a device band and a host band per layer, shared ms axis."""
    left, top, width, row_h = 92, 44, 1080, 21
    base = rows[0]["pre_d0"]
    span = max(r["post_d1"] for r in rows) - base
    scale = width / max(1e-9, span)
    height = top + row_h * 2 * len(rows) + 130
    col = {"pre": "#4c78a8", "fill": "#f58518", "post": "#54a24b", "pread": "#e45756",
           "h2d": "#b279a2", "launch": "#9d755d", "idle": "#dcdcdc"}
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{left + width + 24}" height="{height}" '
         f'font-family="monospace" font-size="12">',
         '<rect width="100%" height="100%" fill="white"/>',
         f'<text x="10" y="17" font-size="14">captured decode: per-layer schedule, layers '
         f'{rows[0]["layer"]}-{rows[-1]["layer"]} ({span:.1f} ms, device clock)</text>',
         '<text x="10" y="33" fill="#555" font-size="11">device: pre / fill / post   '
         'host: pread / H2D / fill issue / launches   grey: device idle</text>']

    def rect(x0, x1, y, h, fill, label=""):
        w = max(1.0, (x1 - x0) * scale)
        p.append(f'<rect x="{left + (x0 - base) * scale:.1f}" y="{y:.1f}" width="{w:.1f}" '
                 f'height="{h}" fill="{fill}"/>')
        if label and w > 16:
            p.append(f'<text x="{left + (x0 - base) * scale + 2:.1f}" y="{y + h - 3}" font-size="9" '
                     f'fill="white">{label}</text>')

    for k, row in enumerate(rows):
        y = top + k * row_h * 2
        p.append(f'<text x="8" y="{y + 11}" font-size="11">L{row["layer"]:02d} dev</text>')
        p.append(f'<text x="8" y="{y + row_h + 11}" font-size="11">L{row["layer"]:02d} host</text>')
        # idle before the fill is the whole question: mark it, then draw the three segments
        rect(row["pre_d1"], row["fill_d0"], y + 1, row_h - 7, col["idle"], "idle")
        for name in ("pre", "fill", "post"):
            rect(row[f"{name}_d0"], row[f"{name}_d1"], y + 1, row_h - 7, col[name], name)
        y2 = y + row_h
        rect(row["h_id0"], row["h_rm1"], y2 + 1, row_h - 7, col["pread"], "pread")
        rect(row["h_rm1"], row["h_blocks1"], y2 + 1, row_h - 7, col["h2d"], "H2D")
        rect(row["h_fill0"], row["h_fill1"], y2 + 1, row_h - 7, col["fill"], "fill")
        rect(row["h_fill1"], row["h_post_done"], y2 + 1, row_h - 7, col["launch"], "launch")
    # -- zoom on one layer: what is actually in flight while the fill moves ------------------------
    row = rows[0]
    z0, z1 = row["fill_d0"] - 0.15, row["post_d1"] + 0.15
    ztop = top + row_h * 2 * len(rows) + 40
    zscale = width / max(1e-9, z1 - z0)
    p.append(f'<text x="10" y="{ztop - 6}" font-size="13">zoom, L{row["layer"]:02d}: the fill\'s whole '
             f'window ({z1 - z0:.2f} ms) - nothing else is queued beside it</text>')
    p.append(f'<line x1="{left}" y1="{ztop}" x2="{left + width}" y2="{ztop}" stroke="#bbb"/>')

    def zrect(x0, x1, y, h, fill, label=""):
        w = max(2.0, (x1 - x0) * zscale)
        p.append(f'<rect x="{left + (x0 - z0) * zscale:.1f}" y="{y}" width="{w:.1f}" height="{h}" '
                 f'fill="{fill}"/>')
        if label:
            p.append(f'<text x="{left + (x0 - z0) * zscale + 3:.1f}" y="{y + h - 4}" font-size="10" '
                     f'fill="white">{label}</text>')

    p.append(f'<text x="8" y="{ztop + 20}" font-size="11">L{row["layer"]:02d} dev</text>')
    p.append(f'<text x="8" y="{ztop + 44}" font-size="11">L{row["layer"]:02d} host</text>')
    zrect(z0, row["post_d0"], ztop + 8, 18, col["idle"], "device idle")
    zrect(row["fill_d0"], row["fill_d1"], ztop + 8, 18, col["fill"], "fill")
    zrect(row["post_d0"], row["post_d1"], ztop + 8, 18, col["post"], "g_post")
    zrect(row["h_fill0"], row["h_fill1"], ztop + 32, 18, col["fill"], "18 views + cat")
    zrect(row["h_fill1"], row["h_post_done"], ztop + 32, 18, col["launch"], "launch post / pre")
    p.append(f'<text x="10" y="{ztop + 72}" font-size="10" fill="#555">fill {row["fill_dev"]:.2f} ms on '
             f'the device clock, of which the issue is {row["fill_host"]:.2f} ms of host time; '
             f'g_post starts {row["dev_gap_fill_post"]:.3f} ms after it</text>')
    p.append('</svg>')
    with open(path, "w") as handle:
        handle.write("\n".join(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="/mnt/weights/Eco-Tech-DeepSeek-V4.1-Flash-1017-w8a8")
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--expert-devices", default="npu:0")
    ap.add_argument("--hbm-share", type=float, default=0.65)
    ap.add_argument("--expert-slots", type=int, default=None)
    ap.add_argument("--expert-batch", type=int, default=8)
    ap.add_argument("--expert-reserve-gib", type=float, default=3.0)
    ap.add_argument("--prompt-tokens", type=int, default=8)
    ap.add_argument("--warm", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--max-context", type=int, default=2048)
    ap.add_argument("--plot-layers", type=int, default=6)
    ap.add_argument("--no-pin", action="store_true",
                    help="run without pinned host memory (the staging ring unpinned, the id read back on "
                         "`idx.cpu()`).  Card-level pinned host memory is a shared resource on this box: "
                         "when another tenant has exhausted it even a 64-byte pin fails, and the capture "
                         "then falls back to eager.  The fill's own cost is unaffected; the H2D becomes "
                         "synchronous and the id read ~70 us a layer dearer")
    ap.add_argument("--track", type=int, default=0,
                    help="extra decode steps to record the six routing ids of every layer, for the "
                         "per-layer hot-set frequency the 'graph per permutation' idea would live on")
    ap.add_argument("--json", default="")
    ap.add_argument("--svg", default="")
    a = ap.parse_args()

    devices = [d for d in a.expert_devices.split(",") if d] or None
    t_build = time.perf_counter()
    run = rt.build(model=a.model, device=a.device, dtype="bfloat16", hbm_share=a.hbm_share,
                   expert_slots=a.expert_slots, expert_batch=a.expert_batch,
                   expert_reserve_gib=a.expert_reserve_gib, expert_devices=devices, graphs=True)
    model, src, cfg = run.model, run.experts, run.cfg
    print(f"built in {time.perf_counter() - t_build:.0f}s: tier {run.expert_slots} slots {src._slots}, "
          f"allocated {torch.npu.memory_allocated(0) / (1 << 30):.1f} GiB", flush=True)
    if a.no_pin:
        src._pin_enabled = False
    state = run.state(a.max_context)
    prompt = [(i * 7 + 3) % 40000 for i in range(a.prompt_tokens)]
    model.forward(prompt, state, compute_logits=True)
    token, out = prompt[-1], None
    for _ in range(a.warm):
        out = model.forward([token], state, logits_last_only=True)
        token = int(out[0].argmax())
    g = model._decode_graphs
    if g is None:
        print("graphs did not capture")
        return 1
    if a.no_pin:
        g.host_id_read = "cpu"
        print("pinned host memory disabled: staging unpinned, id read on idx.cpu()", flush=True)
    print(g.describe(), flush=True)

    global REC
    n_events = 12 * cfg.num_hidden_layers + 2
    real_cat, real = torch.cat, None
    torch.cat = cat_probe(real_cat)
    real = patch(g, src)
    steps = []
    try:
        for s in range(a.steps):
            REC = Recorder(cfg.num_hidden_layers,
                           [torch.npu.Event(enable_timing=True) for _ in range(n_events)])
            h0 = (src.hits, src.misses, src.bytes_in)
            t0 = time.perf_counter()
            out = model.forward([token], state, logits_last_only=True)
            forward_ms = (time.perf_counter() - t0) * 1e3
            sync()
            wall_ms = (time.perf_counter() - t0) * 1e3
            rows = layer_rows(REC, cfg.num_hidden_layers)
            steps.append({"rows": rows, "wall_ms": wall_ms, "forward_ms": forward_ms,
                          "hits": src.hits - h0[0], "misses": src.misses - h0[1],
                          "h2d_gb": (src.bytes_in - h0[2]) / 1e9,
                          "cats": dict(REC.cat_ms), "cat_n": dict(REC.cat_n),
                          "events": REC.used})
            token = int(out[0].argmax())
            print(f"step {s}: {wall_ms:7.1f} ms wall ({forward_ms:.1f} forward), tier "
                  f"{steps[-1]['hits']} hit / {steps[-1]['misses']} miss, host cats "
                  f"{steps[-1]['cats'].get('fill.cat', 0):.1f} ms slab + "
                  f"{steps[-1]['cats'].get('fill.scale_cat', 0):.1f} ms scales, "
                  f"{steps[-1]['events']} events", flush=True)
    finally:
        unpatch(g, src, real)
        torch.cat = real_cat
        REC = None

    routing: list[list[tuple[int, ...]]] = [[] for _ in range(cfg.num_hidden_layers)]
    if a.track:
        real_fetch = g.fetch

        def track_fetch(li, ids_):
            routing[li].append(tuple(int(e) for e in ids_))
            real_fetch(li, ids_)

        g.fetch = track_fetch
        try:
            for _ in range(a.track):
                out = model.forward([token], state, logits_last_only=True)
                token = int(out[0].argmax())
                sync()
        finally:
            g.fetch = real_fetch
        per_layer = []
        for li in range(cfg.num_hidden_layers):
            got = routing[li]
            if not got:
                continue
            sets = Counter(tuple(sorted(s)) for s in got)
            order = Counter(got)
            per_layer.append((len(got), len(sets), sets.most_common(1)[0][1] / len(got),
                              order.most_common(1)[0][1] / len(got)))
        if per_layer:
            print(f"\nhot-set frequency over {a.track} decode steps "
                  f"({per_layer[0][0]} layer-steps a layer):")
            print(f"  distinct 6-id sets a layer: median {statistics.median(p[1] for p in per_layer):.1f}; "
                  f"the most frequent set covers a median {100 * statistics.median(p[2] for p in per_layer):.1f}%"
                  f" of its layer's steps (the ordered tuple {100 * statistics.median(p[3] for p in per_layer):.1f}%)")
            print(f"  one no-fill graph variant per layer needs its own 212 MB slab: "
                  f"{cfg.num_hidden_layers * 212.0e6 / 1e9:.2f} GB for one set a layer, "
                  f"and one capture a set")
    layers, n = cfg.num_hidden_layers, max(1, a.steps)
    region = {k: sum(st["rows"][li][k] for st in steps for li in range(layers)) / n for k in KEYS}
    wall = statistics.fmean(st["wall_ms"] for st in steps)
    d2d_gb = layers * SLAB_BYTES / 1e9
    h2d_gb = statistics.fmean(st["h2d_gb"] for st in steps)
    print(f"\nregions per step (mean of {a.steps} steps, step wall {wall:.1f} ms):")
    for k in KEYS:
        print(f"  {k:22s} {region[k]:9.2f} ms")
    print(f"  measured cat calls per step:  {statistics.fmean(st['cats'].get('fill.cat', 0) for st in steps):.2f}"
          f" ms slab + {statistics.fmean(st['cats'].get('fill.scale_cat', 0) for st in steps):.2f} ms scales")
    print(f"\n8.5 GB split: D2D tier->slab {d2d_gb:.2f} GB/step (fixed 40 x 212 MB); H2D of misses "
          f"{h2d_gb:.2f} GB/step -> the misses are {100 * h2d_gb / (h2d_gb + d2d_gb):.1f}% of the bytes")
    print(f"  tier {run.expert_slots}: {statistics.fmean(st['hits'] for st in steps):.0f} hits / "
          f"{statistics.fmean(st['misses'] for st in steps):.0f} misses a step; pread "
          f"{region['pread_host']:.1f} ms host, H2D {region['h2d_host']:.1f} ms host "
          f"({h2d_gb * 1e3 / max(1e-9, region['h2d_host']):.1f} GB/s)")
    busy = region["pre_dev"] + region["cat_dev_est"] + region["post_dev"]
    print(f"\nwindow check: device busy {busy:.1f} of {wall:.1f} ms ({100 * busy / wall:.1f}%), "
          f"idle {wall - busy:.1f} ms")
    print(f"  g_pre[L] end -> fill kernel start (device idle): {region['dev_idle_before_fill']:.2f} ms "
          f"({region['dev_idle_before_fill'] / layers * 1e3:.0f} us a layer)")
    print(f"  previous g_post end -> this g_pre start (device idle): {region['dev_idle_between']:.2f} ms")
    print(f"  fill kernel end -> g_post start (device idle): {region['dev_gap_fill_post']:.2f} ms")
    print(f"  host launches that run while the fill is in flight: {region['launch_host']:.2f} ms")
    print(f"  fill's device interval: {region['fill_dev']:.2f} ms, of which ~{region['cat_dev_est']:.2f} ms "
          f"is the copy itself and the rest is the host building its 18 views")
    print(f"  of that copy, the part not hidden by the host's next launches: {region['fill_exposed']:.2f} ms")

    window = min(a.plot_layers, layers)
    first = steps[0]["rows"][:window]
    print(f"\nfirst {window} layers, step 0, ms:")
    print(f"  {'L':>3s} {'pre':>7s} {'fill':>7s} {'copy':>7s} {'post':>7s} {'idle pre->fill':>15s} "
          f"{'read wait':>10s} {'pread':>7s} {'fill host':>10s} {'idle L-1->L':>12s}")
    for row in first:
        print(f"  {row['layer']:3d} {row['pre_dev']:7.3f} {row['fill_dev']:7.3f} {row['cat_dev_est']:7.3f} "
              f"{row['post_dev']:7.3f} {row['dev_idle_before_fill']:15.3f} {row['read_wait_host']:10.3f} "
              f"{row['pread_host']:7.3f} {row['fill_host']:10.3f} {row['dev_idle_between']:12.3f}")
    print()
    per_char = max(0.2, (max(r["post_d1"] for r in first) - first[0]["pre_d0"]) / 110.0)
    for line in ascii_schedule(first, 110, per_char):
        print(line)
    if a.json:
        with open(a.json, "w") as handle:
            json.dump({"tier": run.expert_slots, "shares": src._slots, "wall_ms": wall,
                       "regions_ms": region, "d2d_gb": d2d_gb, "h2d_gb": h2d_gb, "steps": steps},
                      handle, indent=1)
        print(f"wrote {a.json}")
    if a.svg:
        try:
            svg_figure(first, a.svg)
            print(f"wrote {a.svg}")
        except Exception as exc:                     # noqa: BLE001 - the numbers are already on disk
            print(f"figure failed ({type(exc).__name__}: {exc}); the JSON and the table above are intact")
    run.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
