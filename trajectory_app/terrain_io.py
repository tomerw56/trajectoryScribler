from __future__ import annotations

from .logging_config import get_logger
import json
from pathlib import Path
import numpy as np

from .models import TerrainConfig

TERRAIN_VERSION = 1


logger = get_logger(__name__)

def save_terrain(path: str | Path, config: TerrainConfig, heights: np.ndarray) -> Path:
    path = Path(path)
    if path.suffix.lower() != ".terrain":
        path = path.with_suffix(".terrain")
    payload = {
        "version": TERRAIN_VERSION,
        "config": config.__dict__,
        "heights": np.asarray(heights, dtype=float).tolist(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_terrain(path: str | Path) -> tuple[TerrainConfig, np.ndarray]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != TERRAIN_VERSION:
        raise ValueError(f"Unsupported terrain version: {payload.get('version')}")
    config = TerrainConfig(**payload["config"])
    heights = np.asarray(payload["heights"], dtype=np.float64)
    if heights.shape != (config.grid_size, config.grid_size):
        raise ValueError(
            f"Terrain grid shape {heights.shape} does not match grid_size={config.grid_size}."
        )
    return config, heights
