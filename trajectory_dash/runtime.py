from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from trajectory_app.core import (
    build_dubins_trajectory,
    build_trajectory,
    derive_velocity_model,
    fixed_velocity_covariance_from_model,
    sample_motion,
    sample_velocity_covariance,
)
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_app.logging_config import get_logger
from trajectory_app.models import (
    AltitudeMode,
    CameramanState,
    MotionConfig,
    TargetState,
    TerrainConfig,
    TrajectoryConfig,
    TrajectoryGenerationMode,
)
from trajectory_app.terrain_io import load_terrain

logger = get_logger(__name__)


@dataclass(frozen=True)
class LoadedScenario:
    path: Path
    terrain_path: Path
    terrain_config: TerrainConfig
    terrain: np.ndarray
    covariance_scope: CovarianceScope
    targets: tuple[TargetState, ...]
    cameramen: tuple[CameramanState, ...]

    @property
    def max_time_sec(self) -> float:
        values = [
            float(target.samples.time_sec[-1])
            for target in self.targets
            if target.samples is not None and len(target.samples.time_sec)
        ]
        return max(values, default=0.0)

    @property
    def playback_step_sec(self) -> float:
        values = [
            float(target.motion_config.sample_dt_sec)
            for target in self.targets
            if target.samples is not None
        ]
        return min(values, default=0.25)

    def target_by_id(self, target_id: str | None) -> TargetState | None:
        if target_id is None:
            return None
        return next((target for target in self.targets if target.id == target_id), None)

    def target_index_at_time(self, target: TargetState, time_sec: float) -> int:
        if target.samples is None or len(target.samples.time_sec) == 0:
            return 0
        index = int(np.searchsorted(target.samples.time_sec, time_sec, side="right") - 1)
        return int(np.clip(index, 0, len(target.samples.time_sec) - 1))

    def covariance_at_time(self, target: TargetState, time_sec: float) -> np.ndarray:
        if target.covariance is None or len(target.covariance.time_sec) == 0:
            return np.zeros((3, 3), dtype=float)
        index = int(np.searchsorted(target.covariance.time_sec, time_sec, side="right") - 1)
        index = int(np.clip(index, 0, len(target.covariance.time_sec) - 1))
        return np.asarray(target.covariance.matrices[index], dtype=float)


class ScenarioRepository:
    """Discover and cache local .trajectory scenarios for the Dash player."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.examples_dir = self.root / "examples"

    def choices(self) -> list[dict[str, str]]:
        paths = sorted(self.examples_dir.glob("*.trajectory"))
        return [{"label": path.stem.replace("_", " "), "value": str(path)} for path in paths]

    def default_path(self) -> Path:
        paths = sorted(self.examples_dir.glob("*.trajectory"))
        if not paths:
            raise FileNotFoundError(f"No .trajectory projects found under {self.examples_dir}")
        return paths[0]

    def load(self, path: str | Path) -> LoadedScenario:
        return self._load_cached(str(Path(path).resolve()))

    @staticmethod
    @lru_cache(maxsize=32)
    def _load_cached(path_text: str) -> LoadedScenario:
        project_path = Path(path_text)
        logger.info("Loading Dash scenario: %s", project_path)
        payload = json.loads(project_path.read_text(encoding="utf-8"))
        version = int(payload.get("version", 0))
        if version not in (2, 3, 4, 5, 6, 7):
            raise ValueError(f"Unsupported trajectory project version: {version}")

        terrain_path = (project_path.parent / payload["terrain_file"]).resolve()
        terrain_config, terrain = load_terrain(terrain_path)
        scope = CovarianceScope(
            payload.get("covariance_scope", CovarianceScope.PREDICTION_XY.value)
        )

        targets: list[TargetState] = []
        for item in payload.get("targets", []):
            trajectory_data = item["trajectory_config"]
            motion_data = item["motion_config"]
            trajectory_config = TrajectoryConfig(
                path_resolution_m=float(trajectory_data["path_resolution_m"]),
                altitude_mode=AltitudeMode(trajectory_data["altitude_mode"]),
                clearance_m=float(trajectory_data["clearance_m"]),
                cruise_altitude_m=float(trajectory_data["cruise_altitude_m"]),
                max_climb_angle_deg=float(trajectory_data["max_climb_angle_deg"]),
                smoothing_passes=int(trajectory_data.get("smoothing_passes", 2)),
                generation_mode=TrajectoryGenerationMode(
                    trajectory_data.get(
                        "generation_mode",
                        TrajectoryGenerationMode.FREEHAND.value,
                    )
                ),
                dubins_min_turn_radius_m=float(
                    trajectory_data.get("dubins_min_turn_radius_m", 10.0)
                ),
            )
            motion_config = MotionConfig(
                speed_mps=float(motion_data["speed_mps"]),
                sample_dt_sec=float(motion_data["sample_dt_sec"]),
                covariance_sample_rate_sec=float(
                    motion_data.get("covariance_sample_rate_sec", 0.5)
                ),
                covariance_window_sec=float(motion_data.get("covariance_window_sec", 5.0)),
                prediction_dt_sec=float(motion_data.get("prediction_dt_sec", 2.0)),
                covariance_stability_window_sec=float(
                    motion_data.get("covariance_stability_window_sec", 6.0)
                ),
            )
            target = TargetState(
                name=item["name"],
                color=item["color"],
                id=item.get("id", item["name"]),
                trajectory_config=trajectory_config,
                motion_config=motion_config,
                scribble_xy=item.get("scribble_xy", []),
            )
            if len(target.scribble_xy) >= 2:
                if (
                    trajectory_config.generation_mode
                    == TrajectoryGenerationMode.DUBINS
                ):
                    target.trajectory, fit = build_dubins_trajectory(
                        target.scribble_xy,
                        terrain,
                        terrain_config,
                        trajectory_config,
                    )
                    target.dubins_anchor_poses = fit.anchor_poses
                    target.dubins_leg_families = fit.leg_families
                    target.dubins_mean_fit_error_m = fit.mean_fit_error_m
                    target.dubins_max_fit_error_m = fit.max_fit_error_m
                else:
                    target.trajectory = build_trajectory(
                        target.scribble_xy,
                        terrain,
                        terrain_config,
                        trajectory_config,
                    )
                target.samples = sample_motion(
                    target.trajectory,
                    terrain,
                    terrain_config,
                    motion_config,
                )
                target.covariance = sample_velocity_covariance(target.samples, motion_config)
                target.derived_velocity_model = derive_velocity_model(
                    target.trajectory,
                    motion_config.speed_mps,
                    trajectory_config.path_resolution_m,
                )
                target.fixed_velocity_covariance = fixed_velocity_covariance_from_model(
                    target.derived_velocity_model,
                    reference_dt_sec=1.0,
                    sigma_factor=3.0,
                )
            targets.append(target)

        cameramen: list[CameramanState] = []
        for item in payload.get("cameramen", []):
            position = item.get("position_xyz")
            cameramen.append(
                CameramanState(
                    name=item.get("name", "Cameraman"),
                    color=item.get("color", "#ffffff"),
                    id=item.get("id", item.get("name", "cameraman")),
                    position_xyz=(
                        [float(value) for value in position]
                        if position is not None
                        else None
                    ),
                    view_radius_m=float(item.get("view_radius_m", 8.0)),
                    view_start_angle_deg=float(item.get("view_start_angle_deg", -30.0)),
                    view_end_angle_deg=float(item.get("view_end_angle_deg", 30.0)),
                )
            )

        scenario = LoadedScenario(
            path=project_path,
            terrain_path=terrain_path,
            terrain_config=terrain_config,
            terrain=np.asarray(terrain, dtype=float),
            covariance_scope=scope,
            targets=tuple(targets),
            cameramen=tuple(cameramen),
        )
        logger.info(
            "Dash scenario ready: targets=%d cameramen=%d max_time=%.2fs scope=%s",
            len(scenario.targets),
            len(scenario.cameramen),
            scenario.max_time_sec,
            scenario.covariance_scope.value,
        )
        return scenario
