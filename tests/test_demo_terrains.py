from pathlib import Path

import numpy as np
import pytest

from trajectory_app.terrain_io import load_terrain


PRESETS = ("multi_hills", "big_top", "canyon", "ridge_pass")


@pytest.mark.parametrize("name", PRESETS)
def test_packaged_demo_terrain_loads(name: str) -> None:
    root = Path(__file__).resolve().parents[1]
    config, heights = load_terrain(root / "terrains" / f"{name}.terrain")

    assert heights.shape == (config.grid_size, config.grid_size)
    assert np.isfinite(heights).all()
    assert float(np.max(heights)) > float(np.min(heights))


def test_big_top_has_prominent_peak() -> None:
    root = Path(__file__).resolve().parents[1]
    _, heights = load_terrain(root / "terrains" / "big_top.terrain")
    assert float(np.max(heights) - np.median(heights)) > 50.0


def test_canyon_has_deep_relief() -> None:
    root = Path(__file__).resolve().parents[1]
    _, heights = load_terrain(root / "terrains" / "canyon.terrain")
    assert float(np.max(heights) - np.min(heights)) > 40.0
