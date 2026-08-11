"""Runtime cascade recognition: query preprocessing + scoring + recognizer.

The codebook (built/loaded via :mod:`cascade_ncc.codebook`) is consumed here:

- ``recognize_cascade`` — functional CPU path: histogram prune -> sparse NCC
- ``CascadeShipRecognizer`` — class interface with GPU-batch (default) or
  vectorized CPU batch, auto-fallback to CPU when no GPU is available.

GPU path: CPU trim-blue -> GpuResize (batch cover-resize) -> shift_y ->
dense GpuSampler(rgb, step) for the 512d histogram + sparse
GpuSampler(gray, ncc_pool) for the exact NCC. Histogram/prune/NCC stay on CPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
from PIL import Image

from ._constants import (
    ALPHA_THRESH,
    EPS,
    GRAY_W,
    K_DEFAULT,
    MAX_QUERIES_DEFAULT,
    OPAQUE_ALPHA,
    REFINE_NCC,
    SHIFT_Y_DEFAULT,
    TOP_N_DEFAULT,
)
from ._gpu import GPU_LOCK, GpuError
from .codebook import CascadeCodebook, _canvas, global_hist, load_cascade_codebook
from .codebook_match import match_codebook
from .primitives import _pooled_multi, preprocess_card

# Shared codebook-agnostic GPU preprocess. GpuPreprocess holds only the query
# working set (src/out/bbox buffers), identical across codebooks that share a
# canvas — so one instance per (max_queries, canvas) serves every codebook.
# On-demand serial inference keeps this safe.
_PREPROCESS_CACHE: dict = {}


def _shared_preprocess(max_queries: int, width: int, height: int,
                       trim_blue: bool = True, align: str = "top-center"):
    """Return the process-wide GpuPreprocess for a canvas/config, shared by codebooks."""
    from .gpu_preprocess import GpuPreprocess
    key = (max_queries, width, height, trim_blue, align)
    pre = _PREPROCESS_CACHE.get(key)
    if pre is None:
        pre = GpuPreprocess(max_images=max_queries, width=width, height=height,
                            trim_blue=trim_blue, align=align)
        _PREPROCESS_CACHE[key] = pre
    return pre


def _query(cb: CascadeCodebook, query: Path | np.ndarray,
           trim_blue: bool, shift_y: int, align: str = "top-center"):
    if isinstance(query, np.ndarray):
        arr = np.asarray(query)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], OPAQUE_ALPHA, np.uint8)])
    else:
        arr = np.asarray(Image.open(query).convert("RGBA"))
    cw, ch = _canvas(cb)
    pre = preprocess_card(arr, trim_blue, shift_y, cw=cw, ch=ch, align=align)
    a = pre.astype(np.float32)
    gray = GRAY_W[0] * a[..., 0] + GRAY_W[1] * a[..., 1] + GRAY_W[2] * a[..., 2]
    step = cb.params["step"]
    # dense grid: alpha + RGB pooled in ONE pass (dense gray is unused)
    d = _pooled_multi([a[..., 3], a[..., 0], a[..., 1], a[..., 2]],
                      cb.xs, cb.ys, step)
    qv = d[0] >= ALPHA_THRESH
    rgb = np.stack([d[1], d[2], d[3]], axis=1)
    f = global_hist(rgb, qv, cb.common, cb.params["bins"])
    # sparse grid: gray + alpha for the exact NCC (ncc_pool pooled)
    s = _pooled_multi([gray, a[..., 3]], cb.xs8, cb.ys8,
                      cb.params["ncc_pool"])
    q8 = np.clip(s[0], 0, 255).astype(np.uint8)
    return q8.astype(np.float32), s[1] >= ALPHA_THRESH, f


def recognize_cascade(cb: CascadeCodebook, query: Path | np.ndarray,
                      k: int = K_DEFAULT, top_n: int = TOP_N_DEFAULT,
                      trim_blue: bool = True, shift_y: int = SHIFT_Y_DEFAULT,
                      refine: int = REFINE_NCC, align: str = "top-center"):
    """Return top-k (index, path, score) via histogram prune -> sparse NCC."""
    q8, qv8, f = _query(cb, query, trim_blue, shift_y, align)
    cand = np.argsort(cb.hist @ f)[::-1][:top_n]
    scores = match_codebook(cb.normed8[cand], cb.samples8[cand],
                            cb.valid8[cand], cb.common8,
                            q8, qv8, refine=refine)
    local = np.argsort(scores)[::-1][:k]      # positions within cand
    order = cand[local]                       # gallery indices
    return [(int(i), cb.paths[i], float(scores[j]))
            for i, j in zip(order, local)]


class CascadeShipRecognizer:
    """GPU-batch (default) or CPU cascade recognizer over numpy arrays.

    Usage:
        r = CascadeShipRecognizer("cascade")            # GPU batch by default
        r = CascadeShipRecognizer("cascade", use_gpu=False)
        top = r.recognize(img_rgba_u8, k=3)             # single -> list
        tops = r.recognize([img1, img2, ...], k=3)      # batch -> list of lists
    """

    def __init__(self, codebook: str | Path | bytes | CascadeCodebook = "cascade",
                 use_gpu: bool = True,
                 max_queries: int = MAX_QUERIES_DEFAULT,
                 trim_blue: bool | None = None, shift_y: int | None = None,
                 top_n: int = TOP_N_DEFAULT, align: str | None = None):
        # A codebook may be a name/path, raw .npz bytes, or an already-loaded
        # CascadeCodebook object.
        self.cb = (codebook if isinstance(codebook, CascadeCodebook)
                   else load_cascade_codebook(codebook))
        p = self.cb.params
        # Default the query preprocessing from the codebook's recorded params so
        # recognition auto-matches how the gallery was laid out. Explicit args
        # (including the old bool/int defaults) win; old codebooks fall back to
        # the canonical top-center + shift-4 + trim config.
        self.trim_blue = (p.get("trim_blue", True) if trim_blue is None
                          else trim_blue)
        self.shift_y = (p.get("shift_y", SHIFT_Y_DEFAULT) if shift_y is None
                        else shift_y)
        self.align = p.get("align", "top-center") if align is None else align
        self.top_n = top_n
        self.use_gpu = use_gpu
        self.max_queries = max_queries
        self._gpu = None
        if use_gpu:
            try:
                self._gpu = self._build_gpu(max_queries)
            except GpuError as exc:
                # wgpu missing / no device / shader compile failed: degrade
                # to CPU. Other exceptions (bad codebook layout, programming
                # errors) propagate instead of being silently swallowed.
                print(f"GPU cascade unavailable ({exc}); falling back to CPU")
                self._gpu = None

    def _build_gpu(self, max_queries: int):
        from ._gpu import default_device
        from .gpu_sampler import GpuSampler
        from .gpu_scorer import CascadeGpuScorer
        cw, ch = _canvas(self.cb)
        # One shared device (process-wide) and one shared GpuPreprocess per
        # canvas: pipelines + the ~30 MB query working set are shared across
        # codebooks. Only the sampler/scorer carry per-codebook data (points,
        # gallery) and are built fresh here.
        device = default_device()
        sd = GpuSampler(self.cb.xs, self.cb.ys, max_images=max_queries,
                        width=cw, height=ch, device=device)
        sd.set_common(self.cb.common)   # codebook constant: upload once, persist
        return {
            "pre": _shared_preprocess(max_queries, cw, ch,
                                      self.trim_blue, self.align),
            "sd": sd,
            "ss": GpuSampler(self.cb.xs8, self.cb.ys8, max_images=max_queries,
                             width=cw, height=ch, device=device),
            "scorer": CascadeGpuScorer(self.cb, max_queries=max_queries,
                                       device=device),
        }

    def recognize(self, images, k: int = K_DEFAULT):
        """Recognize one or many images; returns top-k per image.

        Each input is a (H, W, 3/4) uint8 numpy array or a file path. A single
        input returns one result list; a list/tuple returns a list of results.
        """
        single = not isinstance(images, (list, tuple))
        image_list = [images] if single else list(images)
        if self._gpu is not None:
            results: list = []
            # auto-chunk so a huge input never exceeds the GPU buffer size
            for i in range(0, len(image_list), self.max_queries):
                results.extend(self._gpu_batch(image_list[i:i + self.max_queries], k))
        elif len(image_list) > 1:
            results = self._cpu_batch(image_list, k)
        else:
            results = [self._cpu_one(image_list[0], k)]
        return results[0] if single else results

    def _cpu_one(self, img, k: int):
        return recognize_cascade(self.cb, img, k, self.top_n,
                                 self.trim_blue, self.shift_y,
                                 align=self.align)

    def _cpu_batch(self, image_list, k: int):
        """Vectorized CPU recognition for many images (shared code points)."""
        cb = self.cb
        p = cb.params
        m = len(image_list)
        cw, ch = _canvas(cb)
        arrs = np.stack([preprocess_card(self._to_rgba(img), self.trim_blue,
                                         self.shift_y, cw=cw, ch=ch,
                                         align=self.align)
                         for img in image_list])
        a = arrs.astype(np.float32)
        gray = GRAY_W[0] * a[..., 0] + GRAY_W[1] * a[..., 1] + GRAY_W[2] * a[..., 2]
        # dense: alpha + RGB pooled across the whole batch in one pass
        d = _pooled_multi([a[..., 3], a[..., 0], a[..., 1], a[..., 2]],
                          cb.xs, cb.ys, p["step"])
        qv = d[0] >= ALPHA_THRESH
        rgb = np.stack([d[1], d[2], d[3]], axis=-1)          # (M, P, 3)
        bins = p["bins"]
        shift = round(np.log2(256 // bins))
        qb = (rgb.astype(np.int64) >> shift)
        feats = np.zeros((m, bins ** 3), np.float32)
        for i in range(m):
            idx = np.nonzero(cb.common & qv[i])[0]
            if len(idx):
                flat = (qb[i][idx, 0] * bins * bins
                        + qb[i][idx, 1] * bins + qb[i][idx, 2])
                feats[i] = np.bincount(flat, minlength=bins ** 3)
        fn = np.linalg.norm(feats, axis=1, keepdims=True)
        feats = feats / np.maximum(fn, EPS)
        kth = min(self.top_n, cb.hist.shape[0])   # small codebooks: keep all
        cand = np.argpartition(feats @ cb.hist.T, -kth, axis=1)[:, -self.top_n:]
        s = _pooled_multi([gray, a[..., 3]], cb.xs8, cb.ys8, p["ncc_pool"])
        q8 = np.clip(s[0], 0, 255).astype(np.uint8)
        qv8 = s[1] >= ALPHA_THRESH
        outs = []
        for i in range(m):
            c = cand[i]
            sc = match_codebook(cb.normed8[c], cb.samples8[c], cb.valid8[c],
                                cb.common8, q8[i].astype(np.float32),
                                qv8[i], refine=REFINE_NCC)
            o = np.argsort(sc)[::-1][:k]
            outs.append([(int(c[j]), cb.paths[int(c[j])], float(sc[j]))
                         for j in o])
        return outs

    @staticmethod
    def _to_rgba(img) -> np.ndarray:
        if isinstance(img, (str, Path)):
            return np.asarray(Image.open(img).convert("RGBA"))
        arr = np.asarray(img)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], OPAQUE_ALPHA, np.uint8)])
        return arr

    def _gpu_batch(self, image_list, k: int):
        """GPU batch recognition, serialized by the global GPU lock.

        The shared device, shared GpuPreprocess, and per-stage working buffers
        are not thread-safe, so all GPU inference is serialized by one lock.
        The GPU executes on a single queue anyway, so serializing the Python
        side loses no real throughput — it only prevents cross-thread races.
        """
        with GPU_LOCK:
            return self._gpu_batch_locked(image_list, k)

    def _gpu_batch_locked(self, image_list, k: int):
        g = self._gpu
        p = self.cb.params
        bins = p["bins"]
        pre, sd, ss, sc = g["pre"], g["sd"], g["ss"], g["scorer"]
        arrs = [self._to_rgba(img) for img in image_list]
        m = len(arrs)
        pre.upload(arrs, self.shift_y)
        # ONE command buffer, three passes: bbox -> clear hist -> fused
        # resize/sample/score. Everything stays GPU-resident (processed image
        # in pre.out_buf, hist + sparse gray/valid in the sampler's buffers);
        # only top-k crosses back to the CPU. The fused kernel derives its own
        # resize geometry from the bbox, so no CPU round-trip in the middle.
        encoder = pre.device.create_command_encoder()
        if pre.trim_blue:   # off => bbox sentinel makes the fused kernel full-canvas
            p1 = encoder.begin_compute_pass()
            pre.dispatch_bbox(p1)
            p1.end()
        p2 = encoder.begin_compute_pass()
        sd.enqueue_clear_hist(p2, m)
        p2.end()
        p3 = encoder.begin_compute_pass()
        pre.enqueue_resize(p3)
        sd.enqueue_sample_all(p3, pre.out_buf, ss.pts_buf, sd.common_buf,
                              sd.num_points, ss.num_points,
                              dpool=p["step"], spool=p["ncc_pool"],
                              bins=bins, m=m)
        p3.end()
        p4 = encoder.begin_compute_pass()
        sc.enqueue_prune(p4, sd, m, k, self.top_n)
        p4.end()
        p5 = encoder.begin_compute_pass()
        sc.enqueue_select(p5, sd, m, k, self.top_n)
        p5.end()
        pre.device.queue.submit([encoder.finish()])
        idx, scores = sc.read_topk(m, k)   # readback blocks until the GPU finishes
        outs = []
        for i in range(m):
            outs.append([(int(idx[i, r]), self.cb.paths[int(idx[i, r])],
                          float(scores[i, r])) for r in range(k)])
        return outs


T = TypeVar("T")


class CascadeRecognizer(Generic[T]):
    """High-level ship-card recognizer: codebook (path/bytes) + metadata dict.

    ``meta`` maps each gallery path (as a str) to an arbitrary value; the
    recognizer returns ``(value, confidence, key)`` per match — ``value`` is
    ``meta[key]`` (or ``None`` when there is no metadata for the match) and
    ``key`` is the matched gallery path. The codebook may be a name/path or
    the raw ``.npz`` bytes.

    Usage::

        rec = CascadeRecognizer("data/codebooks/cascade.npz",
                                {".../XM_NORMAL_226.png": "航母 226", ...})
        top = rec.recognize(img_rgba_u8, k=3)     # [(value, conf, key), ...]
        tops = rec.recognize([img1, img2], k=3)   # list of those, one per image
    """

    def __init__(self, codebook: str | Path | bytes,
                 meta: dict[str, T] | None = None,
                 k: int = K_DEFAULT,
                 use_gpu: bool = True,
                 max_queries: int = MAX_QUERIES_DEFAULT,
                 trim_blue: bool | None = None, shift_y: int | None = None,
                 align: str | None = None):
        self.k = k
        self._meta = meta or {}
        self._rec = CascadeShipRecognizer(codebook, use_gpu=use_gpu,
                                          max_queries=max_queries,
                                          trim_blue=trim_blue, shift_y=shift_y,
                                          align=align)
        # The match key is the gallery path RELATIVE to the shared gallery root
        # (the codebook build directory) — short, readable, and unique even when
        # bare filenames repeat across set/id subdirectories.
        self._paths = [str(p) for p in self._rec.cb.paths]
        root = Path(os.path.commonpath(self._paths))
        self._keys = [str(Path(p).relative_to(root)) for p in self._paths]

    @property
    def paths(self) -> list[str]:
        """The absolute gallery paths, in codebook order."""
        return self._paths

    @property
    def keys(self) -> list[str]:
        """The match keys (gallery paths relative to the build directory)."""
        return self._keys

    def recognize(self, images, k: int | None = None):
        """Recognize one or many images; returns (value, confidence, key).

        A single input returns one list of top-k ``(value, confidence, key)``;
        a list/tuple returns a list of those. ``value`` is ``meta[key]`` or
        ``None`` when there is no metadata; ``key`` is the matched gallery path
        relative to the codebook's build directory (e.g. ``1/226/XM_NORMAL_226.png``).
        """
        k = self.k if k is None else k
        single = not isinstance(images, (list, tuple))
        image_list = [images] if single else list(images)
        results = self._rec.recognize(image_list, k=k)
        out = [[(self._meta.get(self._keys[idx]), float(score), self._keys[idx])
                for idx, _, score in top] for top in results]
        return out[0] if single else out
