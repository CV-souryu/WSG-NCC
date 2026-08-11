"""Pytest setup: make the package importable and skip GPU tests off-macOS.

Data-dependent globals (testset, ship names) degrade to empty values when the
data tree is absent so a fresh clone can still run the CPU tests.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cascade_ncc import ROOT as PKG_ROOT

_DATA = PKG_ROOT / "data"
_TEST = _DATA / "groups" / "group1" / "testset"
TEST_CARDS = sorted((_TEST / "cards").glob("*.png")) if (_TEST / "cards").is_dir() else []
MANIFEST = (_TEST / "manifest.jsonl").read_text(encoding="utf-8") \
    if (_TEST / "manifest.jsonl").exists() else ""
SHIP_NAMES = json.loads((_DATA / "ship_names.json").read_text(encoding="utf-8")) \
    if (_DATA / "ship_names.json").exists() else {}


def ship_name_of(path: Path) -> str:
    base = path.parent.name.split("_")[0]
    return SHIP_NAMES.get(base, path.parent.name)


def _card_image(seed: int) -> np.ndarray:
    """Synthetic RGBA card: distinct random colors, fully opaque."""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (240, 124, 3), np.uint8)
    return np.dstack([img, np.full((240, 124), 255, np.uint8)])


def _write_gallery(tmp_path, n: int = 4) -> list:
    """Write n synthetic cards into tmp_path; returns their paths."""
    paths = []
    for i in range(n):
        p = tmp_path / f"card_{i}.png"
        Image.fromarray(_card_image(i)).save(p)
        paths.append(p)
    return paths


@pytest.fixture(scope="session")
def cards():
    """Portrait-card testset images; skip cleanly when the data is absent."""
    if not TEST_CARDS:
        pytest.skip("data/portrait-card-testset not present")
    return TEST_CARDS


@pytest.fixture(scope="session")
def gpu():
    """A usable wgpu adapter; skips the whole session when none is available."""
    try:
        import wgpu
        try:  # wgpu-py <= 0.31
            import wgpu.backends.rs
        except ImportError:
            try:  # wgpu-py >= 0.32
                import wgpu.backends.wgpu_native
            except ImportError:
                import wgpu.backends.auto
    except ImportError:
        pytest.skip("wgpu not available (pip install wgpu)")
    try:
        if hasattr(wgpu.gpu, "request_adapter_sync"):
            adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        else:
            adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
    except Exception as exc:  # noqa: BLE001 - any adapter failure means no GPU -> skip
        pytest.skip(f"no usable wgpu adapter: {exc}")
    if adapter is None:
        pytest.skip("no wgpu adapter (no Vulkan/Metal/DX12 device)")
    return adapter


@pytest.fixture(scope="session")
def gpu_device(gpu):
    if hasattr(gpu, "request_device_sync"):
        return gpu.request_device_sync()
    return gpu.request_device()


@pytest.fixture(scope="session")
def gpu_context(gpu_device):
    """Build the cascade GPU classes once; skip if a kernel fails to compile."""
    from cascade_ncc.gpu_preprocess import GpuPreprocess
    from cascade_ncc.gpu_sampler import GpuSampler
    from cascade_ncc.gpu_scorer import CascadeGpuScorer
    return {"device": gpu_device, "GpuPreprocess": GpuPreprocess,
            "GpuSampler": GpuSampler, "CascadeGpuScorer": CascadeGpuScorer}
