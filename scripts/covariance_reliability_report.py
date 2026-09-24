from __future__ import annotations

import argparse
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
from trajectory_app.models import (
    AltitudeMode,
    MotionConfig,
    TrajectoryConfig,
)
from trajectory_app.terrain_io import load_terrain


ROOT = Path(__file__).resolve().parents[1]


def _trajectory_config(data: dict) -> TrajectoryConfig:
    return TrajectoryConfig(
        path_resolution_m=data["path_resolution_m"],
        altitude_mode=AltitudeMode(data["altitude_mode"]),
        clearance_m=data["clearance_m"],
        cruise_altitude_m=data["cruise_altitude_m"],
        max_climb_angle_deg=data["max_climb_angle_deg"],
        smoothing_passes=data.get("smoothing_passes", 2),
    )


def analyze_project(project_path: Path) -> list[dict]:
    project_path = project_path.resolve()
    payload = json.loads(project_path.read_text(encoding="utf-8"))
    terrain_path = (project_path.parent / payload["terrain_file"]).resolve()
    terrain_cfg, terrain = load_terrain(terrain_path)
    scope = CovarianceScope(payload["covariance_scope"])
    calculator = CovarianceStabilityCalculator()

    report: list[dict] = []
    for item in payload["targets"]:
        trajectory_cfg = _trajectory_config(item["trajectory_config"])
        motion_cfg = MotionConfig(**item["motion_config"])

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
            last_n_seconds=max(
                motion_cfg.covariance_stability_window_sec,
                float(covariance.time_sec[-1]) + 1.0,
            ),
            end_time_sec=float(covariance.time_sec[-1]),
            evaluation_window_sec=motion_cfg.covariance_stability_window_sec,
            source_warmup_sec=motion_cfg.covariance_window_sec,
            scope=scope,
        )
        summary = summarize_covariance_reliability(history.results)

        report.append(
            {
                "target_id": item["id"],
                "target_name": item["name"],
                "classified_samples": summary.classified_samples,
                "stable_fraction": summary.stable_fraction,
                "changing_fraction": summary.changing_fraction,
                "unstable_fraction": summary.unstable_fraction,
                "usable_fraction": summary.usable_fraction,
                "mean_roughness": summary.mean_roughness,
                "max_roughness": summary.max_roughness,
            }
        )

    return report


def print_report(project_path: Path, report: list[dict]) -> None:
    print(f"Covariance reliability report: {project_path}")
    print(
        "Operational reliability = STABLE + CHANGING. "
        "This is temporal usability, not NEES/calibration."
    )
    print()
    print(
        f"{'Target':30s} {'Stable':>8s} {'Changing':>9s} "
        f"{'Unstable':>9s} {'Usable':>8s} {'Mean rough':>11s}"
    )
    print("-" * 84)
    for row in report:
        print(
            f"{row['target_name'][:30]:30s} "
            f"{row['stable_fraction']:8.1%} "
            f"{row['changing_fraction']:9.1%} "
            f"{row['unstable_fraction']:9.1%} "
            f"{row['usable_fraction']:8.1%} "
            f"{row['mean_roughness']:11.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "project",
        nargs="?",
        default=str(ROOT / "examples" / "06_big_top_orbits.trajectory"),
        help="Path to a .trajectory project",
    )
    parser.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=None,
        help="Optional JSON report output path",
    )
    args = parser.parse_args()

    project_path = Path(args.project)
    if not project_path.is_absolute():
        project_path = (Path.cwd() / project_path).resolve()

    report = analyze_project(project_path)
    print_report(project_path, report)

    if args.json_path is not None:
        args.json_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
