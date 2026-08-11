"""Cascade codebook: data model, geometry, build, load, cache.

The codebook is a single self-contained ``.npz``:

- dense (step=2) code points drive a 512-dim global color histogram (pruner)
- a sparse (step=8) subset of those points drives the exact per-common-point
  NCC (refine) with a LARGER ncc_pool pixel neighborhood (9x9), so it is both
  ~12x cheaper than the full grid and robust to a few px of misalignment
- the bottom of the canvas is dropped (top_fraction=0.8 keeps the top 80%):
  the bottom 20% is high-variance card-frame/name noise, and dropping it
  widens the tightest top-1/top-2 margin ~5x with no accuracy loss

This module owns building that artifact and loading it back. Runtime
recognition (query preprocessing, scoring, the GPU-batch recognizer) lives in
:mod:`cascade_ncc.recognizer`.

Example:
    cb = build_cascade_codebook(gallery_paths, name="cascade")
    cb2 = load_cascade_codebook("cascade")
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ._constants import (
    BINS,
    CH,
    CW,
    EPS,
    MIN_COMMON_FRAC,
    NCC_POOL,
    NCC_STEP,
    SHIFT_Y_DEFAULT,
    STEP,
    TOP_FRACTION,
)
from .primitives import (
    _normalize,
    _pooled_multi,
    features_from_rgba,
    numpy_resize,
)

CODEBOOK_DIR = (Path(__file__).resolve().parent.parent
                / "data" / "codebooks")


def _canvas(cb) -> tuple[int, int]:
    """Canvas the codebook was built on (older artifacts default to CW/CH)."""
    return cb.params.get("cw", CW), cb.params.get("ch", CH)


def rect_positions(step: int = STEP, cw: int = CW, ch: int = CH):
    """Rectangular grid over the cw x ch canvas, one point every ``step`` px."""
    gx, gy = np.meshgrid(0.5 + step * np.arange(cw // step),
                         0.5 + step * np.arange(ch // step))
    return (gx.ravel().astype(np.float32), gy.ravel().astype(np.float32))


def _pooled_rgb(rgba, xs: np.ndarray, ys: np.ndarray, step: int,
                cw: int = CW, ch: int = CH) -> np.ndarray:
    """Pooled R/G/B per code point from an RGBA image (pure numpy)."""
    if isinstance(rgba, Image.Image):
        rgba = np.asarray(rgba)
    a = np.asarray(rgba)
    if a.shape[1] != cw or a.shape[0] != ch:
        a = numpy_resize(a, cw, ch)
    r, g, b = _pooled_multi([a[:, :, 0].astype(np.float32),
                             a[:, :, 1].astype(np.float32),
                             a[:, :, 2].astype(np.float32)], xs, ys, step)
    return np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255),
                     np.clip(b, 0, 255)], axis=1)


def global_hist(rgb: np.ndarray, v: np.ndarray, common: np.ndarray,
                bins: int = BINS) -> np.ndarray:
    """L2-normalized bins^3 RGB histogram over common+valid code points."""
    idx = np.nonzero(common & v)[0]
    h = np.zeros(bins ** 3, np.float32)
    if len(idx):
        qb = np.clip((rgb[idx] / (256.0 / bins)).astype(int), 0, bins - 1)
        flat = qb[:, 0] * bins * bins + qb[:, 1] * bins + qb[:, 2]
        h = np.bincount(flat, minlength=bins ** 3).astype(np.float32)
    n = np.linalg.norm(h)
    return h / n if n > EPS else h


def _geometry(step: int, ncc_step: int, top_fraction: float,
              cw: int = CW, ch: int = CH):
    """Dense grid positions + sparse NCC subset, keeping the top fraction."""
    xs_all, ys_all = rect_positions(step, cw, ch)
    nx, ny = cw // step, ch // step
    ratio = ncc_step // step
    n_rows = int(np.floor(top_fraction * ny))      # keep this many rows from top
    rows_kept = np.arange(ny) < n_rows
    dense_keep = np.repeat(rows_kept, nx)          # row-major: repeat per column
    xs, ys = xs_all[dense_keep], ys_all[dense_keep]
    sub = np.zeros((ny, nx), bool)
    sub[np.ix_(rows_kept & (np.arange(ny) % ratio == 0),
               np.arange(nx) % ratio == 0)] = True
    xs8, ys8 = xs_all[sub.ravel()], ys_all[sub.ravel()]
    return xs, ys, xs8, ys8


@dataclass
class CascadeCodebook:
    paths: list[Path]
    xs: np.ndarray            # dense grid positions (histogram)
    ys: np.ndarray
    common: np.ndarray        # dense common mask (histogram)
    hist: np.ndarray          # (N, bins^3) gallery color histograms
    xs8: np.ndarray           # sparse NCC positions
    ys8: np.ndarray
    samples8: np.ndarray      # (N, S) sparse NCC samples (ncc_pool pooled)
    valid8: np.ndarray        # (N, S) sparse NCC validity
    common8: np.ndarray       # (S,) sparse NCC common mask
    normed8: np.ndarray       # (N, S_common) normalized sparse NCC vectors
    params: dict


def _cache_key(paths: list[Path], params: dict) -> str:
    payload = ("|".join(str(p.resolve()) for p in sorted(paths))
               + "|".join(f"{k}={params[k]}" for k in sorted(params))
               + "|cascade1")
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _from_npz(data, params: dict | None = None) -> CascadeCodebook:
    """Reconstruct a CascadeCodebook from an open npz file."""
    if params is None:
        params = json.loads(str(data["params_json"][0]))
    return CascadeCodebook(
        paths=[Path(x) for x in data["paths"]],
        xs=np.asarray(data["xs"]), ys=np.asarray(data["ys"]),
        common=np.asarray(data["common"], dtype=bool),
        hist=np.asarray(data["hist"]),
        xs8=np.asarray(data["xs8"]), ys8=np.asarray(data["ys8"]),
        samples8=np.asarray(data["samples8"]),
        valid8=np.asarray(data["valid8"]),
        common8=np.asarray(data["common8"], dtype=bool),
        normed8=np.asarray(data["normed8"]),
        params=params)


def _to_npz(cb: CascadeCodebook, params_json: str, key: str) -> dict:
    """npz payload for a codebook; the key set must stay stable for old caches."""
    return {
        "paths": np.array([str(p) for p in cb.paths]),
        "xs": cb.xs, "ys": cb.ys, "common": cb.common, "hist": cb.hist,
        "xs8": cb.xs8, "ys8": cb.ys8, "samples8": cb.samples8,
        "valid8": cb.valid8, "common8": cb.common8, "normed8": cb.normed8,
        "params_json": np.array([params_json]), "key": key,
    }


def build_cascade_codebook(
    paths: list[Path],
    step: int = STEP,
    ncc_step: int = NCC_STEP,
    ncc_pool: int = NCC_POOL,
    bins: int = BINS,
    min_common_frac: float = MIN_COMMON_FRAC,
    top_fraction: float = TOP_FRACTION,
    cw: int = CW,
    ch: int = CH,
    trim_blue: bool = True,
    shift_y: int = SHIFT_Y_DEFAULT,
    align: str = "top-center",
    name: str | None = None,
    cache_path: Path | None = None,
    force: bool = False,
) -> CascadeCodebook:
    """Build (or load cached) the single-artifact cascade codebook.

    ``cw``/``ch`` is the canvas the code points live on; it is recorded in the
    artifact and every recognizer path resizes queries to it. ``trim_blue``,
    ``shift_y`` and ``align`` record the QUERY preprocessing that matches this
    gallery's canonical layout — the recognizer reads them back so recognition
    auto-uses the same config instead of silently mismatching.
    """
    paths = [Path(p).resolve() for p in paths]
    params = {"step": step, "ncc_step": ncc_step, "ncc_pool": ncc_pool,
              "bins": bins, "min_common_frac": min_common_frac,
              "top_fraction": top_fraction, "cw": cw, "ch": ch,
              "trim_blue": trim_blue, "shift_y": shift_y, "align": align}
    # Explicit cache_path wins; otherwise fall back to a named default.
    # (A bare ``or`` here parses wrong when name is None and silently disables
    # an explicit cache_path — keep the precedence explicit.)
    cache = (cache_path if cache_path is not None
             else CODEBOOK_DIR / f"{name}.npz" if name else None)
    key = _cache_key(paths, params)

    if not force and cache is not None and cache.exists():
        data = np.load(cache, allow_pickle=False)
        if str(data["key"]) == key:
            return _from_npz(data, params=params)

    xs, ys, xs8, ys8 = _geometry(step, ncc_step, top_fraction, cw, ch)
    t0 = time.perf_counter()
    vrows = []
    s8r = []
    v8r = []
    for p in paths:
        im = Image.open(p).convert("RGBA")
        _, v = features_from_rgba(im, xs, ys, step, cw, ch)      # dense validity
        s8, v8 = features_from_rgba(im, xs8, ys8, ncc_pool, cw, ch)  # sparse NCC
        vrows.append(v.astype(np.uint8))
        s8r.append(s8)
        v8r.append(v8.astype(np.uint8))
    valid = np.stack(vrows)
    common = valid.mean(axis=0) >= min_common_frac
    hist = np.stack([global_hist(_pooled_rgb(Image.open(p).convert("RGBA"),
                                             xs, ys, step, cw, ch),
                                 valid[i], common, bins)
                     for i, p in enumerate(paths)])
    samples8 = np.stack(s8r)
    valid8 = np.stack(v8r)
    common8 = valid8.mean(axis=0) >= min_common_frac
    normed8 = _normalize(samples8, common8)
    print(f"cascade codebook built: {len(paths)} imgs, hist {hist.shape[1]}d, "
          f"NCC {len(xs8)} pts/{int(common8.sum())} common "
          f"({time.perf_counter() - t0:.1f}s)")

    cb = CascadeCodebook(paths, xs, ys, common, hist, xs8, ys8,
                         samples8, valid8, common8, normed8, params)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **_to_npz(cb, json.dumps(params), key))
        print(f"wrote {cache} ({cache.stat().st_size / 1e6:.2f} MB)")
    return cb


def load_cascade_codebook(name_or_path: str | Path | bytes) -> CascadeCodebook:
    """Load a codebook from a name, a .npz path, or the raw .npz bytes."""
    if isinstance(name_or_path, bytes):
        return _from_npz(np.load(io.BytesIO(name_or_path), allow_pickle=False))
    p = Path(name_or_path)
    if p.suffix != ".npz":
        p = CODEBOOK_DIR / f"{name_or_path}.npz"
    if not p.exists():
        raise FileNotFoundError(f"cascade codebook not found: {p}")
    return _from_npz(np.load(p, allow_pickle=False))
