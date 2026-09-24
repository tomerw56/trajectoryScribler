from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .covariance_policy import CovarianceScope


@dataclass(frozen=True)
class CnCBirdSnapshot:
    """One bird at the scenario time captured by the CnC Manager."""

    target_id: str
    target_name: str
    sample_index: int
    sample_time_sec: float
    position_xyz: np.ndarray
    velocity_xyz_mps: np.ndarray
    rolling_velocity_covariance: np.ndarray
    fixed_velocity_covariance: np.ndarray | None


@dataclass(frozen=True)
class CnCCameramanSnapshot:
    cameraman_id: str
    cameraman_name: str
    position_xyz: np.ndarray | None
    view_radius_m: float
    view_start_angle_deg: float
    view_end_angle_deg: float


@dataclass(frozen=True)
class CnCTerrainSnapshot:
    heightmap: np.ndarray
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    x_cell_size: float
    y_cell_size: float


def prediction_offsets_sec(
    prediction_time_sec: float,
    solver_dt_sec: float,
) -> tuple[float, ...]:
    """Return t0 + regular dt jumps + the exact prediction horizon.

    Examples:
        prediction=15, dt=5 -> (0, 5, 10, 15)
        prediction=12, dt=5 -> (0, 5, 10, 12)

    Including the exact horizon makes the meaning of prediction_time_sec
    unambiguous even when it is not divisible by solver_dt_sec.
    """

    horizon = float(prediction_time_sec)
    dt = float(solver_dt_sec)
    if not math.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("prediction_time_sec must be finite and > 0")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("solver_dt_sec must be finite and > 0")

    offsets = [0.0]
    step = dt
    epsilon = max(1e-12, abs(horizon) * 1e-12)

    while step < horizon - epsilon:
        offsets.append(float(step))
        step += dt

    if not math.isclose(offsets[-1], horizon, rel_tol=0.0, abs_tol=epsilon):
        offsets.append(horizon)

    return tuple(offsets)


@dataclass(frozen=True)
class CnCSolverRequest:
    """Complete immutable scenario snapshot presented to a scenario solver."""

    scenario_name: str
    scenario_time_sec: float
    prediction_time_sec: float
    solver_dt_sec: float
    solution_time_sec: float
    covariance_scope: CovarianceScope
    birds: tuple[CnCBirdSnapshot, ...]
    cameramen: tuple[CnCCameramanSnapshot, ...]
    terrain: CnCTerrainSnapshot

    @property
    def prediction_offsets_sec(self) -> tuple[float, ...]:
        return prediction_offsets_sec(
            self.prediction_time_sec,
            self.solver_dt_sec,
        )


@dataclass(frozen=True)
class CnCObservationProbability:
    """Probability of one cameraman observing one bird at one prediction offset."""

    cameraman_id: str
    cameraman_name: str
    target_id: str
    target_name: str
    prediction_offset_sec: float
    observation_probability: float

    def __post_init__(self) -> None:
        probability = float(self.observation_probability)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise ValueError(
                "observation_probability must be finite and within [0, 1]"
            )


@dataclass(frozen=True)
class CnCSolverResult:
    """Scenario solver output.

    `observation_probabilities` is the primary result shape:
    cameraman × bird × prediction-offset -> probability.
    """

    status: str
    summary: str = ""
    observation_probabilities: tuple[CnCObservationProbability, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)

    def probabilities_for(
        self,
        cameraman_id: str,
        target_id: str,
    ) -> tuple[CnCObservationProbability, ...]:
        return tuple(
            item
            for item in self.observation_probabilities
            if item.cameraman_id == cameraman_id
            and item.target_id == target_id
        )


@runtime_checkable
class CnCSolver(Protocol):
    """Scenario-level solver used by CnCManager."""

    name: str

    def solve(self, request: CnCSolverRequest) -> CnCSolverResult:
        ...


class SnapshotInspectionSolver:
    """Contract/demo solver that fills the observation-probability shape.

    The probability calculation is intentionally a placeholder. Its purpose is
    to exercise the cameraman × bird × dt contract end-to-end before the real
    observation algorithm is implemented.
    """

    name = "snapshot-observation-demo"

    @staticmethod
    def _placeholder_probability(
        cameraman: CnCCameramanSnapshot,
        bird: CnCBirdSnapshot,
        offset_sec: float,
    ) -> float:
        if cameraman.position_xyz is None:
            return 0.0

        camera = np.asarray(cameraman.position_xyz, dtype=float)
        future = (
            np.asarray(bird.position_xyz, dtype=float)
            + np.asarray(bird.velocity_xyz_mps, dtype=float) * float(offset_sec)
        )
        horizontal_distance = float(np.linalg.norm(future[:2] - camera[:2]))

        # Placeholder only: smooth deterministic distance proxy.
        scale = max(1.0, float(cameraman.view_radius_m))
        probability = math.exp(-horizontal_distance / scale)
        return float(np.clip(probability, 0.0, 1.0))

    def solve(self, request: CnCSolverRequest) -> CnCSolverResult:
        probabilities: list[CnCObservationProbability] = []

        for cameraman in request.cameramen:
            for bird in request.birds:
                for offset in request.prediction_offsets_sec:
                    probabilities.append(
                        CnCObservationProbability(
                            cameraman_id=cameraman.cameraman_id,
                            cameraman_name=cameraman.cameraman_name,
                            target_id=bird.target_id,
                            target_name=bird.target_name,
                            prediction_offset_sec=float(offset),
                            observation_probability=self._placeholder_probability(
                                cameraman,
                                bird,
                                float(offset),
                            ),
                        )
                    )

        return CnCSolverResult(
            status="ready",
            summary=(
                f"{len(request.cameramen)} cameramen × "
                f"{len(request.birds)} birds × "
                f"{len(request.prediction_offsets_sec)} prediction times"
            ),
            observation_probabilities=tuple(probabilities),
            data={
                "prediction_offsets_sec": list(request.prediction_offsets_sec),
                "prediction_time_sec": float(request.prediction_time_sec),
                "solver_dt_sec": float(request.solver_dt_sec),
                "solution_time_sec": float(request.solution_time_sec),
                "probability_model": "placeholder-distance-proxy",
            },
        )
