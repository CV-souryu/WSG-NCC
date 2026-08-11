"""CPU-path primitives tests — no GPU and no real data required."""

from __future__ import annotations

import numpy as np

from cascade_ncc.primitives import (
    _trim_blue,
    features_from_rgba,
    numpy_resize,
    preprocess_card,
    refine_query,
)


def test_numpy_resize_output_shape():
    a = np.zeros((240, 124, 3), np.uint8)
    out = numpy_resize(a, 62, 31)
    assert out.shape == (31, 62, 3)
    assert out.dtype == np.float32


def test_numpy_resize_preserves_constant():
    a = np.full((240, 124, 3), 77, np.uint8)
    out = numpy_resize(a, 62, 31)
    assert np.allclose(out, 77.0, atol=0.5)


def test_preprocess_card_custom_canvas():
    img = np.full((200, 200, 3), 100, np.uint8)
    out = preprocess_card(img, trim_blue=False, shift_y=0, cw=62, ch=120)
    assert out.shape == (120, 62, 4)
    assert (out[..., 3] == 255).all()


def test_preprocess_card_adds_alpha():
    img = np.full((200, 124, 3), 100, np.uint8)
    out = preprocess_card(img, trim_blue=False, shift_y=0)
    assert out.shape == (240, 124, 4)
    assert (out[..., 3] == 255).all()


def test_preprocess_card_trims_blue_border():
    # Gray card with a blue band on top and left; trims to the gray content.
    img = np.full((240, 200, 3), 128, np.uint8)
    img[:30, :, 0] = 0
    img[:30, :, 2] = 255
    img[:, :20, 0] = 0
    img[:, :20, 2] = 255
    img = np.dstack([img, np.full((240, 200), 255, np.uint8)])
    trimmed = preprocess_card(img, trim_blue=True, shift_y=0)
    untrimmed = preprocess_card(img, trim_blue=False, shift_y=0)
    assert trimmed.shape == (240, 124, 4)
    assert not np.array_equal(trimmed, untrimmed)
    # blue bands are gone from the trimmed output's top rows
    top = trimmed[:3, :, :3].astype(int)
    assert not ((top[:, :, 2] > top[:, :, 0] + 20)
                & (top[:, :, 2] > top[:, :, 1] + 20)).any()


def test_preprocess_card_shift_y_blank_top():
    img = np.full((240, 124, 3), 200, np.uint8)
    out = preprocess_card(img, trim_blue=False, shift_y=4)
    assert out.shape == (240, 124, 4)
    assert (out[:4, :, :3] == 0).all()        # blank top-4 rows
    assert (out[4:, :, :3] == 200).all()      # content shifted down 4
    assert (out[4:, :, 3] == 255).all()


def test_trim_blue_shared_helper():
    # No-op on a full-bleed image.
    gray = np.full((50, 50, 4), 128, np.uint8)
    assert _trim_blue(gray).shape == gray.shape
    # Crops a blue frame to the non-blue content bbox.
    framed = gray.copy()
    framed[:10] = (0, 0, 255, 255)
    framed[:, :10] = (0, 0, 255, 255)
    out = _trim_blue(framed)
    assert out.shape == (40, 40, 4)
    assert (out == 128).all()


def test_features_from_rgba_gray_and_valid():
    img = np.full((240, 124, 4), 128, np.uint8)
    img[0, 0, 3] = 0
    s, v = features_from_rgba(img, np.array([62.0]), np.array([120.0]), 1)
    assert s.shape == (1,)
    assert abs(float(s[0]) - 128) <= 1
    assert v[0]
    _, v2 = features_from_rgba(img, np.array([0.0]), np.array([0.0]), 1)
    assert not v2[0]


def test_refine_query_scores_identity():
    n = 60
    rng = np.random.RandomState(0)
    pattern = rng.randint(0, 256, n).astype(np.float32)
    samples = np.stack([pattern, pattern[::-1]]).astype(np.uint8)
    valid = np.ones((2, n), np.uint8)
    common = np.ones(n, bool)
    q = pattern.astype(np.float32)
    qv = np.ones(n, bool)
    exact, order = refine_query(samples, valid, common, q, qv,
                                np.array([1.0, 0.5]), refine=50)
    assert order[0] == 0
    assert abs(exact[0] - 1.0) < 1e-4
