from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from trajectory_app.core import build_dubins_trajectory, derive_velocity_model
from trajectory_app.dubins import (
    DubinsPose,
    fit_freehand_dubins,
    sample_dubins_path,
    shortest_dubins_path,
)
from trajectory_app.models import (
    TerrainConfig,
    TrajectoryConfig,
    TrajectoryGenerationMode,
)


ROOT = Path(__file__).resolve().parents[1]


def _flat_terrain():
    cfg = TerrainConfig(
        x_min=-100.0,
        x_max=100.0,
        y_min=-100.0,
        y_max=100.0,
        grid_size=80,
        base_height=0.0,
        mountain_count=0,
        valley_count=0,
        noise_amplitude=0.0,
    )
    return cfg, np.zeros((cfg.grid_size, cfg.grid_size), dtype=float)


def test_shortest_dubins_sampler_reaches_requested_pose():
    start = DubinsPose(-20.0, -10.0, math.radians(15.0))
    end = DubinsPose(35.0, 22.0, math.radians(120.0))

    path = shortest_dubins_path(start, end, radius_m=8.0)
    xy = sample_dubins_path(path, spacing_m=0.25)

    assert path.family in {"LSL", "LSR", "RSL", "RSR", "RLR", "LRL"}
    assert np.allclose(xy[0], [start.x, start.y], atol=1e-9)
    assert np.allclose(xy[-1], [end.x, end.y], atol=1e-8)
    assert path.length_m > 0.0


def test_freehand_dubins_fit_keeps_endpoints_and_creates_anchor_poses():
    raw = np.array(
        [
            [-70.0, -30.0],
            [-50.0, -28.0],
            [-30.0, -15.0],
            [-10.0, 10.0],
            [15.0, 25.0],
            [40.0, 22.0],
            [65.0, 5.0],
        ],
        dtype=float,
    )

    fit = fit_freehand_dubins(
        raw,
        min_turn_radius_m=10.0,
        path_resolution_m=0.25,
        smoothing_passes=2,
    )

    assert len(fit.xy) > 20
    assert len(fit.anchor_poses) >= 2
    assert len(fit.leg_families) == len(fit.anchor_poses) - 1
    assert np.allclose(fit.xy[0], raw[0], atol=1e-6)
    assert np.allclose(fit.xy[-1], raw[-1], atol=1e-6)
    assert fit.mean_fit_error_m >= 0.0
    assert fit.max_fit_error_m >= fit.mean_fit_error_m


def test_dubins_build_uses_existing_altitude_pipeline():
    terrain_cfg, terrain = _flat_terrain()
    raw = [
        [-60.0, -20.0],
        [-40.0, -15.0],
        [-20.0, 0.0],
        [0.0, 20.0],
        [25.0, 30.0],
        [55.0, 20.0],
    ]
    cfg = TrajectoryConfig(
        path_resolution_m=0.25,
        clearance_m=3.0,
        generation_mode=TrajectoryGenerationMode.DUBINS,
        dubins_min_turn_radius_m=8.0,
    )

    trajectory, fit = build_dubins_trajectory(
        raw,
        terrain,
        terrain_cfg,
        cfg,
    )

    assert len(trajectory.xy) == len(trajectory.xyz)
    assert np.allclose(trajectory.xyz[:, 2], 3.0)
    assert fit.min_turn_radius_m == 8.0

    model = derive_velocity_model(
        trajectory,
        speed_mps=4.0,
        path_resolution_m=cfg.path_resolution_m,
    )
    # The analysis path is intentionally smoothed/coarsened, so use a modest
    # tolerance rather than demanding the exact commanded Rmin.
    assert model.min_turn_radius_m >= 6.0


def test_dubins_path_outside_terrain_is_rejected():
    cfg = TerrainConfig(
        x_min=-10.0,
        x_max=10.0,
        y_min=-10.0,
        y_max=10.0,
        grid_size=30,
        mountain_count=0,
        valley_count=0,
        noise_amplitude=0.0,
    )
    terrain = np.zeros((cfg.grid_size, cfg.grid_size), dtype=float)
    trajectory_cfg = TrajectoryConfig(
        generation_mode=TrajectoryGenerationMode.DUBINS,
        dubins_min_turn_radius_m=20.0,
        path_resolution_m=0.25,
    )

    raw = [
        [-9.0, -8.0],
        [-7.0, -4.0],
        [-4.0, 0.0],
        [-7.0, 4.0],
        [-9.0, 8.0],
    ]

    try:
        build_dubins_trajectory(raw, terrain, cfg, trajectory_cfg)
    except ValueError as exc:
        assert "terrain bounds" in str(exc)
    else:
        raise AssertionError("Expected out-of-bounds Dubins fit to be rejected")


def test_target_manager_contains_radius_prompt_and_two_drawing_modes():
    source = (
        ROOT / "trajectory_app" / "target_manager.py"
    ).read_text(encoding="utf-8")

    assert "Begin freehand drawing / redraw" in source
    assert "Begin Dubins drawing / redraw" in source
    assert "QInputDialog.getDouble" in source
    assert "beginDubinsDrawingRequested" in source


def test_project_and_dash_support_dubins_generation_mode():
    main = (
        ROOT / "trajectory_app" / "main_window.py"
    ).read_text(encoding="utf-8")
    runtime = (
        ROOT / "trajectory_dash" / "runtime.py"
    ).read_text(encoding="utf-8")

    assert '"version": 7' in main
    assert "TrajectoryGenerationMode.DUBINS" in main
    assert "build_dubins_trajectory" in main

    assert "TrajectoryGenerationMode.DUBINS" in runtime
    assert "build_dubins_trajectory" in runtime
