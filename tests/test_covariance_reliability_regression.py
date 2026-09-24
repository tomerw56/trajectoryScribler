from __future__ import annotations

import json
from pathlib import Path

from trajectory_app.core import (
    build_trajectory,
    sample_motion,
    sample_velocity_covariance,
)
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_app.covariance_reliability import (
    summarize_covariance_reliability,
)
from trajectory_app.covariance_stability import CovarianceStabilityCalculator
from trajectory_app.models import AltitudeMode, MotionConfig, TrajectoryConfig
from trajectory_app.terrain_io import load_terrain


ROOT = Path(__file__).resolve().parents[1]


def _target_reliability(project_name: str) -> dict[str, float]:
    project_path = ROOT / "examples" / project_name
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    terrain_path = (project_path.parent / payload["terrain_file"]).resolve()
    terrain_cfg, terrain = load_terrain(terrain_path)
    scope = CovarianceScope(payload["covariance_scope"])
    calculator = CovarianceStabilityCalculator()

    result: dict[str, float] = {}
    for item in payload["targets"]:
        tc = item["trajectory_config"]
        mc = item["motion_config"]
        trajectory_cfg = TrajectoryConfig(
            path_resolution_m=tc["path_resolution_m"],
            altitude_mode=AltitudeMode(tc["altitude_mode"]),
            clearance_m=tc["clearance_m"],
            cruise_altitude_m=tc["cruise_altitude_m"],
            max_climb_angle_deg=tc["max_climb_angle_deg"],
            smoothing_passes=tc.get("smoothing_passes", 2),
        )
        motion_cfg = MotionConfig(**mc)

        trajectory = build_trajectory(
            item["scribble_xy"],
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

        history = calculator.evaluate_history(
            covariance,
            last_n_seconds=float(covariance.time_sec[-1]) + 1.0,
            end_time_sec=float(covariance.time_sec[-1]),
            evaluation_window_sec=motion_cfg.covariance_stability_window_sec,
            source_warmup_sec=motion_cfg.covariance_window_sec,
            scope=scope,
        )
        result[item["name"]] = summarize_covariance_reliability(
            history.results
        ).usable_fraction

    return result


def test_big_top_smooth_routes_remain_operationally_reliable():
    reliability = _target_reliability("06_big_top_orbits.trajectory")

    # These routes are deliberately smooth. A covariance implementation change
    # must not make expected rotation/drift look like temporal unreliability.
    assert reliability["Outer mountain arc"] >= 0.95
    assert reliability["Inner mountain arc"] >= 0.95
    assert reliability["Summit crossing"] >= 0.95
    assert reliability["High cruise orbit"] >= 0.85


def test_high_variance_stress_routes_are_still_less_reliable():
    reliability = _target_reliability(
        "11_high_variance_maneuvers_flat.trajectory"
    )

    # The regression should not become so permissive that deliberate covariance
    # thrashing is accepted as reliable.
    assert reliability["Sawtooth fast"] <= 0.40
    assert reliability["Repeated box turns"] <= 0.25
    assert reliability["Reverse S maneuver"] <= 0.50
