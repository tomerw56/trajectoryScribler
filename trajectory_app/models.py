from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import uuid
import numpy as np


class AltitudeMode(str, Enum):
    TERRAIN_CLEARANCE = "terrain_clearance"
    CRUISE_ALTITUDE = "cruise_altitude"


class TrajectoryGenerationMode(str, Enum):
    FREEHAND = "freehand"
    DUBINS = "dubins"


@dataclass
class TerrainConfig:
    x_min: float = -15.0
    x_max: float = 25.0
    y_min: float = -15.0
    y_max: float = 15.0
    grid_size: int = 80
    seed: int = 7
    base_height: float = 0.0
    valley_count: int = 2
    mountain_count: int = 4
    valley_depth_min: float = 1.0
    valley_depth_max: float = 4.0
    mountain_height_min: float = 2.0
    mountain_height_max: float = 8.0
    min_sigma: float = 2.0
    max_sigma: float = 7.0
    noise_amplitude: float = 0.15

    @property
    def x_cell_size(self) -> float:
        return (self.x_max - self.x_min) / max(1, self.grid_size - 1)

    @property
    def y_cell_size(self) -> float:
        return (self.y_max - self.y_min) / max(1, self.grid_size - 1)


@dataclass
class TrajectoryConfig:
    path_resolution_m: float = 0.25
    altitude_mode: AltitudeMode = AltitudeMode.TERRAIN_CLEARANCE
    clearance_m: float = 3.0
    cruise_altitude_m: float = 8.0
    max_climb_angle_deg: float = 18.0
    smoothing_passes: int = 2
    generation_mode: TrajectoryGenerationMode = TrajectoryGenerationMode.FREEHAND
    dubins_min_turn_radius_m: float = 10.0


@dataclass
class MotionConfig:
    speed_mps: float = 4.0
    sample_dt_sec: float = 0.25
    covariance_sample_rate_sec: float = 0.5
    covariance_window_sec: float = 5.0
    prediction_dt_sec: float = 2.0
    covariance_stability_window_sec: float = 6.0


@dataclass
class TrajectoryResult:
    xy: np.ndarray
    xyz: np.ndarray
    terrain_z: np.ndarray
    cumulative_distance_3d: np.ndarray


@dataclass
class MotionSamples:
    time_sec: np.ndarray
    distance_3d: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    terrain_z: np.ndarray

    @property
    def speed(self) -> np.ndarray:
        return np.linalg.norm(self.velocity, axis=1)

    def __len__(self) -> int:
        return int(self.position.shape[0])


@dataclass
class CovarianceSamples:
    time_sec: np.ndarray
    matrices: np.ndarray  # [N,3,3]

    @property
    def cxx(self) -> np.ndarray: return self.matrices[:, 0, 0]
    @property
    def cyy(self) -> np.ndarray: return self.matrices[:, 1, 1]
    @property
    def czz(self) -> np.ndarray: return self.matrices[:, 2, 2]
    @property
    def trace(self) -> np.ndarray: return np.trace(self.matrices, axis1=1, axis2=2)



@dataclass
class SolverSampleEvaluation:
    sample_index: int
    time_sec: float
    actual_position: np.ndarray
    estimated_position: np.ndarray
    error_vector: np.ndarray
    squared_error_m2: float

    @property
    def error_m(self) -> float:
        return float(np.sqrt(self.squared_error_m2))


@dataclass
class SolverStatistics:
    evaluations: dict[int, SolverSampleEvaluation]

    @classmethod
    def empty(cls) -> "SolverStatistics":
        return cls(evaluations={})

    def put(
        self,
        sample_index: int,
        time_sec: float,
        actual_position: np.ndarray,
        estimated_position: np.ndarray,
    ) -> SolverSampleEvaluation:
        actual = np.asarray(actual_position, dtype=np.float64).copy()
        estimated = np.asarray(estimated_position, dtype=np.float64).copy()
        error = estimated - actual
        squared = float(np.dot(error, error))
        evaluation = SolverSampleEvaluation(
            sample_index=int(sample_index),
            time_sec=float(time_sec),
            actual_position=actual,
            estimated_position=estimated,
            error_vector=error,
            squared_error_m2=squared,
        )
        self.evaluations[int(sample_index)] = evaluation
        return evaluation

    def get(self, sample_index: int) -> SolverSampleEvaluation | None:
        return self.evaluations.get(int(sample_index))

    def clear(self) -> None:
        self.evaluations.clear()

    @property
    def count(self) -> int:
        return len(self.evaluations)

    @property
    def mse_m2(self) -> float:
        if not self.evaluations:
            return 0.0
        return float(np.mean([e.squared_error_m2 for e in self.evaluations.values()]))

    @property
    def rmse_m(self) -> float:
        return float(np.sqrt(self.mse_m2))


@dataclass
class DerivedVelocityModel:
    speed_mps: float
    min_turn_radius_m: float
    max_turn_rate_deg_s: float
    max_lateral_accel_mps2: float
    max_climb_angle_deg: float
    max_descent_angle_deg: float
    max_climb_rate_mps: float
    max_descent_rate_mps: float

    @property
    def has_finite_turn(self) -> bool:
        return bool(np.isfinite(self.min_turn_radius_m))


@dataclass
class CameramanState:
    name: str
    color: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    position_xyz: list[float] | None = None
    view_radius_m: float = 8.0
    view_start_angle_deg: float = -30.0
    view_end_angle_deg: float = 30.0

    @property
    def is_placed(self) -> bool:
        return self.position_xyz is not None


@dataclass
class TargetState:
    name: str
    color: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    trajectory_config: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    motion_config: MotionConfig = field(default_factory=MotionConfig)
    scribble_xy: list[list[float]] = field(default_factory=list)
    trajectory: TrajectoryResult | None = None
    samples: MotionSamples | None = None
    covariance: CovarianceSamples | None = None
    current_index: int = 0
    # Runtime-only analysis of the generated/smoothed trajectory. It is
    # deliberately not serialized or exported.
    derived_velocity_model: DerivedVelocityModel | None = None
    # Fixed all-purpose velocity covariance derived from the read-only
    # velocity model. Runtime-only; never serialized.
    fixed_velocity_covariance: np.ndarray | None = None

    # Runtime-only solver outputs. The same configured solver is evaluated
    # twice: rolling/observed covariance and fixed/model covariance.
    estimated_position: np.ndarray | None = None
    fixed_cov_estimated_position: np.ndarray | None = None
    solver_metadata: dict[str, Any] = field(default_factory=dict)
    fixed_cov_solver_metadata: dict[str, Any] = field(default_factory=dict)
    solver_statistics: SolverStatistics = field(default_factory=SolverStatistics.empty)

    # Runtime-only Dubins fit diagnostics. Rebuilt from the scribble on load.
    dubins_anchor_poses: np.ndarray | None = None
    dubins_leg_families: tuple[str, ...] = field(default_factory=tuple)
    dubins_mean_fit_error_m: float | None = None
    dubins_max_fit_error_m: float | None = None


def dataclass_dict(value: Any) -> dict[str, Any]:
    data = asdict(value)
    if isinstance(value, TrajectoryConfig):
        data["altitude_mode"] = value.altitude_mode.value
        data["generation_mode"] = value.generation_mode.value
    return data
