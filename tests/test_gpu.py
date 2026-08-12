"""Cascade GPU kernel correctness tests against the CPU reference.

Catches the silent-correctness bugs this codebase hit historically (bbox
off-by-one, src_start bytes-vs-pixels, Metal uchar3 misreads, atomic types).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from cascade_ncc._constants import BLUE_TRIM_THRESH, HIST_DIM
from cascade_ncc.primitives import (
    features_from_rgba,
    features_rgb_from_rgba,
    preprocess_card,
)
from tests.conftest import ship_name_of

PREPROCESS_RGB_TOL = 50   # fused preprocess is bilinear, CPU _preprocess LANCZOS


def _load_cards(cards, n: int) -> list[np.ndarray]:
    return [np.asarray(Image.open(c).convert("RGBA")) for c in cards[:n]]


def _synthetic_124x240(seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (240, 124, 4), np.uint8)
    img[:, :, 3] = 255
    img[:20, :20, 3] = 0
    img[220:, 100:, 3] = 0
    return img


def _cpu_bbox(arr: np.ndarray):
    rgb = arr[:, :, :3].astype(np.int16)
    blue = (rgb[:, :, 2] > rgb[:, :, 0] + BLUE_TRIM_THRESH) & \
           (rgb[:, :, 2] > rgb[:, :, 1] + BLUE_TRIM_THRESH)
    if not blue.any():
        return (0, 0, arr.shape[1] - 1, arr.shape[0] - 1)
    ys, xs = np.where(~blue)
    if not ys.size:
        return (0, 0, arr.shape[1] - 1, arr.shape[0] - 1)
    if (xs.min() > 0 or ys.min() > 0
            or xs.max() < arr.shape[1] - 1 or ys.max() < arr.shape[0] - 1):
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return (0, 0, arr.shape[1] - 1, arr.shape[0] - 1)


def _pooled_rgb(im, xs, ys, pool=2):
    from cascade_ncc.primitives import _pooled_multi
    a = np.asarray(im.resize((124, 240), Image.LANCZOS))
    rgb = a[:, :, :3].astype(np.float32)
    return np.stack([np.clip(c, 0, 255) for c in
                     _pooled_multi([rgb[..., c] for c in range(3)],
                                   xs, ys, pool)], axis=1)


def test_gpu_preprocess_bbox_matches_cpu(gpu_context, cards):
    """Blue-border bounding box computed by the GPU atomic reduction."""
    from cascade_ncc._gpu import download
    imgs = _load_cards(cards, 6)
    cpu_bbox = [_cpu_bbox(a) for a in imgs]
    pre = gpu_context["GpuPreprocess"](max_images=8,
                                       device=gpu_context["device"])
    m = pre.upload(imgs, 4)
    pre.dispatch_bbox()
    gpu_bbox = np.frombuffer(download(pre.device, pre.bbox_buf, m * 16),
                             np.uint32).copy().reshape(m, 4)
    for i, (cbx, gb) in enumerate(zip(cpu_bbox, gpu_bbox)):
        assert tuple(int(v) for v in gb) == cbx, f"card {i} bbox mismatch"


def test_gpu_preprocess_matches_cpu(gpu_context, cards):
    """Fused trim+resize+shift output close to CPU preprocess_card."""
    pre = gpu_context["GpuPreprocess"](max_images=8)
    imgs = _load_cards(cards, 6)
    out = pre.run(imgs, shift_y=4)
    for i, a in enumerate(imgs):
        ref = preprocess_card(a, True, 4)
        d = np.abs(out[i].astype(int) - ref.astype(int))
        assert d[:, :, :3].max() <= PREPROCESS_RGB_TOL, f"card {i} rgb"
        assert d[:, :, 3].max() <= 1, f"card {i} alpha"


def test_gpu_preprocess_unmask_matches_cpu(gpu_context, cards):
    """GPU unmask (RGB / unmask) matches the CPU preprocess within bilinear tol."""
    UNMASK_RGB_TOL = 120
    imgs = _load_cards(cards, 2)
    for trim_blue in (True, False):
        pre = gpu_context["GpuPreprocess"](max_images=4, trim_blue=trim_blue,
                                           unmask=0.5,
                                           device=gpu_context["device"])
        out = pre.run(imgs, shift_y=4)
        for i, a in enumerate(imgs):
            ref = preprocess_card(a, trim_blue, 4, unmask=0.5)
            d = np.abs(out[i].astype(int) - ref.astype(int))
            assert d[:, :, :3].max() <= UNMASK_RGB_TOL, f"trim={trim_blue} card {i} rgb"
            assert d[:, :, 3].max() <= 1, f"trim={trim_blue} card {i} alpha"


def test_gpu_concurrent_inference_serialized(gpu_context, cards):
    """Global GPU lock: concurrent recognize() calls stay race-free."""
    import threading

    from cascade_ncc.recognizer import CascadeShipRecognizer
    recs = [CascadeShipRecognizer("cascade"), CascadeShipRecognizer("cascade")]
    sub = cards[:6]
    arrs = [np.asarray(Image.open(c).convert("RGBA")) for c in sub]
    ref = [[str(x[1]) for x in top] for top in recs[0].recognize(arrs, k=3)]
    errors: list = []

    def worker(ri: int):
        try:
            for _ in range(3):
                out = recs[ri].recognize(arrs, k=3)
                got = [[str(x[1]) for x in top] for top in out]
                assert got == ref, f"thread {ri}: result mismatch"
        except Exception as exc:   # noqa: BLE001 - surface any thread error
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i % 2,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_gpu_preprocess_alignment_matches_cpu(gpu_context, cards):
    """Configurable align/trim: crop-vs-fill edges match the CPU preprocess."""
    # Non-default alignment pushes different content regions to the canvas
    # edges, where bilinear-vs-LANCZOS differs most — use the wider tolerance
    # that the custom-canvas test already needs for hard cover-resizes.
    ALIGN_TOL = 120
    imgs = _load_cards(cards, 3)
    for align in ("top-left", "top-right", "bottom-right", "center-center"):
        pre = gpu_context["GpuPreprocess"](max_images=4, align=align,
                                           device=gpu_context["device"])
        out = pre.run(imgs, shift_y=4)
        for i, a in enumerate(imgs):
            ref = preprocess_card(a, True, 4, align=align)
            d = np.abs(out[i].astype(int) - ref.astype(int))
            assert d[:, :, :3].max() <= ALIGN_TOL, \
                f"{align} card {i} rgb"
            assert d[:, :, 3].max() <= 1, f"{align} card {i} alpha"
    # trim_blue off -> full canvas (bbox sentinel), matches CPU
    pre = gpu_context["GpuPreprocess"](max_images=4, trim_blue=False,
                                       device=gpu_context["device"])
    out = pre.run(imgs, shift_y=4)
    for i, a in enumerate(imgs):
        ref = preprocess_card(a, False, 4)
        d = np.abs(out[i].astype(int) - ref.astype(int))
        assert d[:, :, :3].max() <= PREPROCESS_RGB_TOL, f"notrim card {i} rgb"


def test_gpu_sampler_sample_all_matches_cpu(gpu_context, cards):
    """Fused dense-hist + sparse sampling matches the CPU reference."""
    from cascade_ncc._gpu import download
    from cascade_ncc.codebook import load_cascade_codebook
    cb = load_cascade_codebook("cascade")
    dev = gpu_context["device"]
    # Fusing stages into ONE encoder requires a shared device.
    pre = gpu_context["GpuPreprocess"](max_images=2, device=dev)
    sd = gpu_context["GpuSampler"](cb.xs, cb.ys, max_images=2, device=dev)
    ss = gpu_context["GpuSampler"](cb.xs8, cb.ys8, max_images=2, device=dev)
    arrs = _load_cards(cards, 2)
    pre.upload(arrs, 4)
    sd.set_common(cb.common)
    encoder = dev.create_command_encoder()
    p1 = encoder.begin_compute_pass()
    pre.dispatch_bbox(p1)
    p1.end()
    p2 = encoder.begin_compute_pass()
    sd.enqueue_clear_hist(p2, 2)
    p2.end()
    p3 = encoder.begin_compute_pass()
    pre.enqueue_resize(p3)
    sd.enqueue_sample_all(p3, pre.out_buf, ss.pts_buf, sd.common_buf,
                          sd.num_points, ss.num_points,
                          dpool=cb.params["step"], spool=cb.params["ncc_pool"],
                          m=2)
    p3.end()
    dev.queue.submit([encoder.finish()])
    n3 = 2 * ss.num_points * 3
    srgb = np.frombuffer(download(dev, sd.sa_srgb_buf, n3 * 4),
                         np.uint32).copy().astype(np.uint8).reshape(
        2, ss.num_points, 3)
    svalid = np.frombuffer(download(dev, sd.sa_svalid_buf, n3 * 4),
                           np.uint32).copy().astype(np.uint8).reshape(
        2, ss.num_points, 3)
    hist = np.frombuffer(download(dev, sd.hist_buf, 2 * HIST_DIM * 4),
                         np.uint32).copy().reshape(2, HIST_DIM)
    # CPU reference on the SAME processed image the GPU sampled (pre.out_buf)
    out_arr = np.frombuffer(download(dev, pre.out_buf, 2 * 240 * 124 * 4),
                            np.uint8).copy().reshape(2, 240, 124, 4)
    imgs = [Image.fromarray(out_arr[i]) for i in range(2)]
    # sparse RGB/valid
    for i, im in enumerate(imgs):
        sg, sv = features_rgb_from_rgba(im, cb.xs8, cb.ys8,
                                        cb.params["ncc_pool"])
        assert np.abs(srgb[i].astype(int) - sg.astype(int)).max() <= 5, \
            f"card {i} sparse rgb"
        assert np.array_equal(svalid[i][:, 0].astype(bool), sv), \
            f"card {i} sparse valid"
    # dense HSL spatial histogram
    from cascade_ncc.codebook import codebook_hist
    for i, im in enumerate(imgs):
        _, v = features_from_rgba(im, cb.xs, cb.ys, cb.params["step"])
        rgb = _pooled_rgb(im, cb.xs, cb.ys, cb.params["step"])
        cpu = codebook_hist(cb, rgb, v, normalize=False)
        # HSL bins sit on float boundaries: pooled RGB can differ by 1 unit
        # between CPU/GPU, occasionally flipping a bin. A small uniform
        # region can flip together, so allow a handful of counts (+-5).
        assert np.abs(hist[i].astype(int) - cpu).max() <= 5, f"card {i} hist"


def test_gpu_scorer_matches_cpu(gpu_context, cards):
    """Fused prune+top-N+NCC scorer matches the CPU cascade scoring."""
    from cascade_ncc.codebook import codebook_hist, load_cascade_codebook
    from cascade_ncc.codebook_match import match_codebook
    cb = load_cascade_codebook("cascade")
    imgs = [Image.fromarray(preprocess_card(a, True, 4))
            for a in _load_cards(cards, 3)]
    # raw hist + sparse features via CPU
    feats_raw = np.zeros((3, HIST_DIM), np.uint32)
    srgb = np.zeros((3, len(cb.xs8) * 3), np.uint8)
    svalid = np.zeros((3, len(cb.xs8) * 3), np.uint8)
    for i, im in enumerate(imgs):
        _, v = features_from_rgba(im, cb.xs, cb.ys, cb.params["step"])
        rgb = _pooled_rgb(im, cb.xs, cb.ys, cb.params["step"])
        feats_raw[i] = codebook_hist(cb, rgb, v, normalize=False)
        sg, sv = features_rgb_from_rgba(im, cb.xs8, cb.ys8,
                                        cb.params["ncc_pool"])
        srgb[i] = sg.reshape(-1)
        svalid[i] = np.repeat(sv.astype(np.uint8), 3)
    # CPU reference
    feats_n = feats_raw.astype(np.float32)
    feats_n /= np.maximum(np.linalg.norm(feats_n, axis=1, keepdims=True), 1e-6)
    cpu_idx, cpu_sc = [], []
    for i in range(3):
        cand = np.argpartition(cb.hist @ feats_n[i], -20)[-20:]
        sc = match_codebook(cb.normed8[cand], cb.samples8[cand],
                            cb.valid8[cand], cb.common8,
                            srgb[i].astype(np.float32),
                            svalid[i].astype(bool), refine=50)
        o = np.argsort(sc)[::-1][:3]
        cpu_idx.append(cand[o]); cpu_sc.append(sc[o])
    scorer = gpu_context["CascadeGpuScorer"](cb, max_queries=4)
    gpu_idx, gpu_sc = scorer.score(feats_raw, srgb, svalid, k=3, top_n=20)
    assert np.array_equal(gpu_idx, np.stack(cpu_idx))
    assert np.abs(gpu_sc - np.stack(cpu_sc)).max() < 1e-4


def test_cascade_gpu_matches_cpu(gpu_context, cards):
    """GPU-batch cascade top-1 names equal the CPU cascade per card."""
    from cascade_ncc.codebook import load_cascade_codebook
    from cascade_ncc.recognizer import CascadeShipRecognizer, recognize_cascade
    cb = load_cascade_codebook("cascade")
    rec = CascadeShipRecognizer("cascade")
    cards = cards[:6]
    arrs = [np.asarray(Image.open(c).convert("RGBA")) for c in cards]
    gpu_out = rec.recognize(arrs, k=3)
    for i, card in enumerate(cards):
        cpu = recognize_cascade(cb, card, k=3, top_n=20, shift_y=4)
        assert ship_name_of(cpu[0][1]) == ship_name_of(gpu_out[i][0][1]), \
            f"card {card.name}"


def test_gpu_trim_blue_override_applied(gpu_context, cards):
    """Recognizer trim_blue=False must reach the shared GPU preprocess.

    Regression: the override used to be silently ignored because
    _apply_gpu_config never synced pre.trim_blue, so the GPU kept trimming
    while the CPU path honored the flag.
    """
    from cascade_ncc._gpu import download
    from cascade_ncc.primitives import preprocess_card
    from cascade_ncc.recognizer import CascadeShipRecognizer
    rec = CascadeShipRecognizer("cascade", use_gpu=True, trim_blue=False)
    assert rec._gpu is not None
    cards = cards[:4]
    arrs = [np.asarray(Image.open(c).convert("RGBA")) for c in cards]
    rec._gpu_batch(arrs, k=1)   # applies the config lazily on the first batch
    pre = rec._gpu["pre"]
    assert pre.trim_blue is False
    out = np.frombuffer(
        download(pre.device, pre.out_buf, len(cards) * 240 * 124 * 4),
        np.uint8).copy().reshape(len(cards), 240, 124, 4)
    for i, a in enumerate(arrs):
        ref = preprocess_card(a, False, 4, fit_width=True)
        d = np.abs(out[i].astype(int) - ref.astype(int))
        assert d[:, :, :3].max() <= PREPROCESS_RGB_TOL, f"card {i} rgb"
        assert d[:, :, 3].max() <= 1, f"card {i} alpha"


def test_gpu_batch_matches_single(gpu_context, cards):
    """Batch recognition == processing each card individually on the GPU."""
    from cascade_ncc.recognizer import CascadeShipRecognizer
    rec = CascadeShipRecognizer("cascade")
    cards = cards[:4]
    arrs = [np.asarray(Image.open(c).convert("RGBA")) for c in cards]
    batch = rec.recognize(arrs, k=3)
    singles = [rec.recognize(a, k=3) for a in arrs]
    for b, s in zip(batch, singles):
        assert [x[1] for x in b] == [x[1] for x in s]


def test_gpu_preprocess_custom_canvas(gpu_context, cards):
    """Fused preprocess at a non-default canvas matches the CPU reference.

    The bilinear-vs-LANCZOS difference scales with how hard the cover-resize
    downsamples, so the small canvas uses a wider tolerance than the 124x240
    default (which caps at PREPROCESS_RGB_TOL).
    """
    cw, ch = 62, 120
    pre = gpu_context["GpuPreprocess"](max_images=4, width=cw, height=ch)
    imgs = _load_cards(cards, 4)
    out = pre.run(imgs, shift_y=4)
    assert out.shape == (4, ch, cw, 4)
    for i, a in enumerate(imgs):
        ref = preprocess_card(a, True, 4, cw=cw, ch=ch)
        d = np.abs(out[i].astype(int) - ref.astype(int))
        assert d[:, :, :3].max() <= 120, f"card {i} rgb"
        assert d[:, :, 3].max() <= 1, f"card {i} alpha"


def test_cascade_gpu_large_sparse_grid(gpu_context, tmp_path):
    """GPU recognizes a sparse grid well above the old 512-point cap."""
    from cascade_ncc.codebook import build_cascade_codebook
    from cascade_ncc.recognizer import CascadeShipRecognizer, recognize_cascade
    from tests.conftest import _write_gallery
    paths = _write_gallery(tmp_path, 4)
    # Default 124x240 canvas with ncc_step=4 -> 1488 sparse points (> 512).
    cb = build_cascade_codebook(paths, cache_path=tmp_path / "cb_large.npz",
                                step=2, ncc_step=4, ncc_pool=3)
    assert len(cb.common8) > 512
    rec = CascadeShipRecognizer(str(tmp_path / "cb_large.npz"), use_gpu=True,
                                trim_blue=False, shift_y=0)
    assert rec._gpu is not None          # GPU path active, not a CPU fallback
    arrs = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
    gpu_out = rec.recognize(arrs, k=3)
    for i, p in enumerate(paths):
        cpu = recognize_cascade(cb, p, k=3, top_n=20, trim_blue=False,
                                shift_y=0)
        assert cpu[0][0] == i, f"card {i} not self-match on CPU"
        assert gpu_out[i][0][1].resolve() == p.resolve(), \
            f"card {i} GPU top-1 mismatch"


def test_cascade_gpu_custom_canvas(gpu_context, tmp_path):
    """End-to-end GPU recognition on a codebook built at 62x120 (synthetic).

    Synthetic distinct-color cards make self-match reliable, so GPU top-1 must
    equal the CPU top-1 (the card itself) at the non-default canvas.
    """
    from cascade_ncc.codebook import build_cascade_codebook
    from cascade_ncc.recognizer import CascadeShipRecognizer, recognize_cascade
    from tests.conftest import _write_gallery
    paths = _write_gallery(tmp_path, 4, size=(62, 120))
    cb = build_cascade_codebook(paths, cache_path=tmp_path / "cb62.npz",
                                cw=62, ch=120, step=2, ncc_step=4, ncc_pool=3)
    rec = CascadeShipRecognizer(str(tmp_path / "cb62.npz"), use_gpu=True,
                                trim_blue=False, shift_y=0)
    assert rec._gpu is not None
    arrs = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
    gpu_out = rec.recognize(arrs, k=3)
    for i, p in enumerate(paths):
        cpu = recognize_cascade(cb, p, k=3, top_n=20, trim_blue=False,
                                shift_y=0)
        assert cpu[0][0] == i, f"card {i} not self-match on CPU"
        assert gpu_out[i][0][1].resolve() == p.resolve(), \
            f"card {i} GPU top-1 mismatch"


def test_region_activation_gpu_matches_cpu(gpu_context, tmp_path):
    """Region-restricted GPU recognition matches the CPU region path."""
    from cascade_ncc.codebook import build_cascade_codebook
    from cascade_ncc.recognizer import CascadeShipRecognizer
    from tests.conftest import _write_gallery
    paths = _write_gallery(tmp_path, 4)
    cb_path = tmp_path / "cb.npz"
    build_cascade_codebook(paths, cache_path=cb_path,
                           trim_blue=False, shift_y=0,
                           step=2, ncc_step=4, ncc_pool=3)
    region = (0, 50, 0, 100)
    gpu = CascadeShipRecognizer(str(cb_path), use_gpu=True,
                                trim_blue=False, shift_y=0, region=region)
    cpu = CascadeShipRecognizer(str(cb_path), use_gpu=False,
                                trim_blue=False, shift_y=0, region=region)
    arrs = [np.asarray(Image.open(p).convert("RGBA")) for p in paths]
    assert gpu._gpu is not None
    for i, arr in enumerate(arrs):
        got = gpu.recognize(arr, k=1)[0][0]
        ref = cpu.recognize(arr, k=1)[0][0]
        assert got == ref == i, f"card {i}: gpu={got} cpu={ref}"
