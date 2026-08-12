"""CPU codebook build/recognize tests on synthetic images (no GPU/data)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cascade_ncc._constants import SCORE_SENTINEL
from cascade_ncc.codebook import (
    build_cascade_codebook,
    load_cascade_codebook,
    region_codebook,
)
from cascade_ncc.codebook_match import match_codebook, normalize_common
from cascade_ncc.recognizer import CascadeShipRecognizer, recognize_cascade
from tests.conftest import _write_gallery

_BUILD = {"step": 2, "ncc_step": 4, "ncc_pool": 3}   # small + fast


def _build(tmp_path, paths):
    return build_cascade_codebook(paths, cache_path=tmp_path / "cb.npz",
                                  **_BUILD)


def test_build_roundtrip(tmp_path):
    paths = _write_gallery(tmp_path, 4)
    cb = _build(tmp_path, paths)
    cb2 = load_cascade_codebook(tmp_path / "cb.npz")
    assert cb2.hist.shape == cb.hist.shape == (4, 576)
    assert cb2.samples8.shape == cb.samples8.shape
    assert cb2.common8.dtype == bool
    assert cb2.params == cb.params
    for field in ("xs", "ys", "common", "hist", "xs8", "ys8", "samples8",
                  "valid8", "common8", "normed8"):
        assert np.array_equal(getattr(cb2, field), getattr(cb, field))


def test_build_cache_hit_and_force(tmp_path):
    paths = _write_gallery(tmp_path, 3)
    cb1 = _build(tmp_path, paths)
    # Second build serves the cache even after the sources are gone.
    for p in paths:
        p.unlink()
    cb2 = _build(tmp_path, paths)
    assert np.array_equal(cb1.hist, cb2.hist)
    # force=True rebuilds and re-reads sources -> clean FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        build_cascade_codebook(paths, cache_path=tmp_path / "cb.npz",
                               force=True, **_BUILD)


def test_recognize_cascade_top1_recovers_gallery(tmp_path):
    paths = _write_gallery(tmp_path, 4)
    cb = _build(tmp_path, paths)
    for i, p in enumerate(paths):
        top = recognize_cascade(cb, p, k=3, top_n=20,
                                trim_blue=False, shift_y=0)
        assert top[0][0] == i, f"card {i} top-1 mismatch: {top[0]}"
        assert top[0][1].resolve() == p.resolve()
        scores = [t[2] for t in top]
        assert scores == sorted(scores, reverse=True)


def test_recognize_cascade_returns_k(tmp_path):
    paths = _write_gallery(tmp_path, 4)
    cb = _build(tmp_path, paths)
    top = recognize_cascade(cb, paths[0], k=3, top_n=20,
                            trim_blue=False, shift_y=0)
    assert len(top) == 3
    assert len({t[0] for t in top}) == 3   # distinct gallery indices


def test_cascade_recognizer_path_bytes_and_meta(tmp_path):
    """CascadeRecognizer: codebook (path/bytes) + meta dict -> (value, conf)."""
    from cascade_ncc.recognizer import CascadeRecognizer
    paths = _write_gallery(tmp_path, 4)
    cb_path = tmp_path / "cb.npz"
    build_cascade_codebook(paths, cache_path=cb_path,
                           trim_blue=False, shift_y=0, **_BUILD)
    blob = cb_path.read_bytes()

    rec = CascadeRecognizer(blob, use_gpu=False)
    assert rec.keys[0].endswith("card_0.png")   # key = relative to build dir
    meta = {key: f"卡{i + 1}" for i, key in enumerate(rec.keys)}
    rec = CascadeRecognizer(blob, meta=meta, use_gpu=False)
    arrs = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
    top = rec.recognize(arrs, k=1)
    for i, row in enumerate(top):
        value, conf, key = row[0]
        assert value == f"卡{i + 1}", f"card {i} got {value}"
        assert isinstance(conf, float)
        assert key == rec.keys[i]

    # from a path, with an object (dict) meta value
    meta_obj = {key: {"idx": i} for i, key in enumerate(rec.keys)}
    rec2 = CascadeRecognizer(str(cb_path), meta=meta_obj, use_gpu=False)
    top2 = rec2.recognize(arrs[1], k=1)
    assert top2[0][0] == {"idx": 1}

    # no meta -> value is None, key is the relative gallery path
    rec3 = CascadeRecognizer(blob, use_gpu=False)
    top3 = rec3.recognize(arrs[0], k=1)
    assert top3[0][0] is None
    assert top3[0][2] == rec3.keys[0]


def test_build_recognize_custom_canvas(tmp_path):
    """A codebook on a non-default canvas (62x120) recognizes itself."""
    paths = _write_gallery(tmp_path, 4, size=(62, 120))
    cb = build_cascade_codebook(paths, cache_path=tmp_path / "cb62.npz",
                                cw=62, ch=120, **_BUILD)
    assert cb.params["cw"] == 62 and cb.params["ch"] == 120
    cb2 = load_cascade_codebook(tmp_path / "cb62.npz")
    assert cb2.params["cw"] == 62 and cb2.params["ch"] == 120
    for i, p in enumerate(paths):
        top = recognize_cascade(cb, p, k=3, top_n=20,
                                trim_blue=False, shift_y=0)
        assert top[0][0] == i, f"card {i} top-1 mismatch: {top[0]}"
        assert top[0][1].resolve() == p.resolve()


def test_recognizer_unmask_codebook_default_and_override(tmp_path):
    """Recognizer defaults to codebook unmask; explicit 0.0 disables it."""
    paths = _write_gallery(tmp_path, 2)
    cb_path = tmp_path / "cb.npz"
    build_cascade_codebook(paths, cache_path=cb_path,
                           trim_blue=False, shift_y=0, unmask=0.4, **_BUILD)
    rec_default = CascadeShipRecognizer(str(cb_path), use_gpu=False,
                                        trim_blue=False, shift_y=0)
    assert rec_default.unmask == 0.4
    rec_override = CascadeShipRecognizer(str(cb_path), use_gpu=False,
                                         trim_blue=False, shift_y=0,
                                         unmask=0.0)
    assert rec_override.unmask == 0.0


def test_region_activation_top50_buckets(tmp_path):
    """Region (0,50,0,100) activates the top 2 rows (6 of 9 histogram cells)."""
    paths = _write_gallery(tmp_path, 4)
    cb = build_cascade_codebook(paths, cache_path=tmp_path / "cb.npz",
                                trim_blue=False, shift_y=0, **_BUILD)
    rc = region_codebook(cb, (0, 50, 0, 100))
    color_bins = 64
    assert rc.hist_mask is not None and rc.hist_mask.shape == (576,)
    assert rc.hist_mask[:6 * color_bins].all()
    assert not rc.hist_mask[6 * color_bins:].any()
    assert rc.common8.sum() < cb.common8.sum()
    top = recognize_cascade(cb, paths[0], k=1, top_n=20,
                            trim_blue=False, shift_y=0,
                            region=(0, 50, 0, 100))
    assert top[0][0] == 0


def test_recognize_cascade_array_input(tmp_path):
    paths = _write_gallery(tmp_path, 4)
    cb = _build(tmp_path, paths)
    arr = np.asarray(Image.open(paths[0]).convert("RGBA"))
    top_path = recognize_cascade(cb, paths[0], k=1, top_n=20,
                                 trim_blue=False, shift_y=0)
    top_arr = recognize_cascade(cb, arr, k=1, top_n=20,
                                trim_blue=False, shift_y=0)
    assert top_path[0][0] == top_arr[0][0]


def test_match_codebook_ranks_identity_first():
    n = 60
    rng = np.random.RandomState(1)
    pattern = rng.randint(0, 256, n).astype(np.float32)
    common = np.ones(n, bool)
    samples = np.stack([pattern, pattern[::-1]]).astype(np.uint8)
    valid = np.ones((2, n), np.uint8)
    rows = np.stack([pattern - pattern.mean(), pattern[::-1] - pattern[::-1].mean()])
    normed = rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6)
    q = pattern.astype(np.float32)
    qv = np.ones(n, bool)
    sc = match_codebook(normed, samples, valid, common, q, qv, refine=50)
    assert sc[0] > sc[1]
    assert abs(sc[0] - 1.0) < 1e-4
    assert np.isneginf(SCORE_SENTINEL)        # sentinel used for no-survivor
    assert not np.isneginf(sc[0])


def test_normalize_common_unit_norm():
    rng = np.random.RandomState(2)
    q = (rng.rand(50) * 255).astype(np.float32)
    common = np.zeros(50, bool)
    common[10:40] = True
    qn = normalize_common(q, common)
    assert qn.shape == (30,)
    assert abs(np.linalg.norm(qn) - 1.0) < 1e-5
    assert abs(qn.mean()) < 1e-5


def test_recognizer_cpu_batch_equals_single(tmp_path):
    paths = _write_gallery(tmp_path, 4)
    _build(tmp_path, paths)
    r = CascadeShipRecognizer(str(tmp_path / "cb.npz"), use_gpu=False,
                              trim_blue=False, shift_y=0)
    a1 = np.asarray(Image.open(paths[0]).convert("RGBA"))
    a2 = np.asarray(Image.open(paths[1]).convert("RGBA"))
    batch = r.recognize([a1, a2], k=3)
    single1 = r.recognize(a1, k=3)
    single2 = r.recognize(a2, k=3)
    assert [x[1] for x in batch[0]] == [x[1] for x in single1]
    assert [x[1] for x in batch[1]] == [x[1] for x in single2]
    for b, s in ((batch[0], single1), (batch[1], single2)):
        assert [round(x[2], 6) for x in b] == [round(x[2], 6) for x in s]
