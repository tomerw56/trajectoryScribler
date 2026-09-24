from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from trajectory_app.core import (
    build_trajectory,
    sample_motion,
    sample_velocity_covariance,
)
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_app.models import (
    AltitudeMode,
    MotionConfig,
    TrajectoryConfig,
)
from trajectory_app.terrain_io import load_terrain


ROOT = Path(__file__).resolve().parents[1]


def _configs(target_data: dict) -> tuple[TrajectoryConfig, MotionConfig]:
    trajectory = target_data["trajectory_config"]
    motion = target_data["motion_config"]

    return (
        TrajectoryConfig(
            path_resolution_m=trajectory["path_resolution_m"],
            altitude_mode=AltitudeMode(trajectory["altitude_mode"]),
            clearance_m=trajectory["clearance_m"],
            cruise_altitude_m=trajectory["cruise_altitude_m"],
            max_climb_angle_deg=trajectory["max_climb_angle_deg"],
            smoothing_passes=trajectory.get("smoothing_passes", 2),
        ),
        MotionConfig(**motion),
    )


def test_all_example_projects_reference_existing_terrain_and_generate():
    projects = sorted((ROOT / "examples").glob("*.trajectory"))
    assert len(projects) >= 11

    for project_path in projects:
        data = json.loads(project_path.read_text(encoding="utf-8"))
        assert data["version"] == 6
        CovarianceScope(data["covariance_scope"])
        assert len(data["targets"]) >= 4
        assert len(data["cameramen"]) >= 3

        terrain_path = (project_path.parent / data["terrain_file"]).resolve()
        assert terrain_path.exists()

        terrain_cfg, terrain = load_terrain(terrain_path)

        for target_data in data["targets"]:
            trajectory_cfg, motion_cfg = _configs(target_data)
            trajectory = build_trajectory(
                target_data["scribble_xy"],
                terrain,
                terrain_cfg,
                trajectory_cfg,
            )
            samples = sample_motion(
                trajectory,
                terrain,
                terrain_cfg,
                motion_cfg,
            )
            covariance = sample_velocity_covariance(samples, motion_cfg)

            assert len(samples) > 2
            assert len(covariance.time_sec) > 2
            assert np.all(np.isfinite(covariance.matrices))


def test_flat_covariance_showcase_contains_small_and_large_xy_covariance():
    path = ROOT / "examples" / "01_covariance_showcase_flat.trajectory"
    data = json.loads(path.read_text(encoding="utf-8"))
    terrain_path = (path.parent / data["terrain_file"]).resolve()
    terrain_cfg, terrain = load_terrain(terrain_path)

    max_traces: dict[str, float] = {}
    for target_data in data["targets"]:
        trajectory_cfg, motion_cfg = _configs(target_data)
        trajectory = build_trajectory(
            target_data["scribble_xy"],
            terrain,
            terrain_cfg,
            trajectory_cfg,
        )
        samples = sample_motion(trajectory, terrain, terrain_cfg, motion_cfg)
        covariance = sample_velocity_covariance(samples, motion_cfg)
        xy_trace = covariance.cxx + covariance.cyy
        max_traces[target_data["id"]] = float(np.max(xy_trace))

    assert max_traces["demo_straight"] < 1e-8
    assert max_traces["demo_near_straight"] < 0.01
    assert max_traces["demo_s_curve"] > 1.0
    assert max_traces["demo_zigzag"] > 10.0



def test_scenario_manifest_matches_generated_projects():
    manifest_path = ROOT / "examples" / "scenario_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    projects = sorted((ROOT / "examples").glob("*.trajectory"))

    assert len(manifest) == len(projects)
    assert {item["project"] for item in manifest} == {path.name for path in projects}

    for item in manifest:
        assert item["bird_count"] == len(item["birds"])
        assert item["cameraman_count"] == len(item["cameramen"])
        assert item["bird_count"] >= 4
        assert item["cameraman_count"] >= 3
