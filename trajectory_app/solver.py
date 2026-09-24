from __future__ import annotations

from .logging_config import get_logger
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .covariance_policy import CovarianceScope
from .models import MotionConfig, TrajectoryConfig



logger = get_logger(__name__)

@dataclass(frozen=True)

class SolverRequest:
    """Everything a position-estimation solver receives for one target/sample.

    ``velocity_covariance`` is the rolling covariance of [Vx, Vy, Vz], so its
    units are (m/s)^2.  The solver is free to ignore fields it does not need.
    """

    target_id: str
    target_name: str
    sample_index: int
    time_sec: float
    current_position: np.ndarray
    current_velocity: np.ndarray
    velocity_covariance: np.ndarray
    terrain_height_m: float
    motion_config: MotionConfig
    trajectory_config: TrajectoryConfig
    covariance_scope: CovarianceScope = CovarianceScope.PREDICTION_XY
    previous_estimated_position: np.ndarray | None = None


@dataclass(frozen=True)
class SolverResult:
    estimated_position: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Solver(Protocol):
    """Replaceable API used by the GUI to estimate the current position."""

    name: str

    def solve(self, request: SolverRequest) -> SolverResult:
        ...



class PrincipalSigmaDemoSolver:
    """Visible demonstration of the solver API.

    Covariance alone does not imply a biased mean.  To make the integration
    visible, this demo returns one deterministic 1-sigma hypothesis along the
    dominant velocity-covariance axis over a short prediction horizon.

    This is intentionally a *hypothesis marker*, not a claim that the expected
    position has moved.  Replace this class with the real consumer of your
    covariance when that algorithm is ready.
    """

    name = "1σ principal-covariance demo"

    def __init__(self, sigma_scale: float = 1.0, horizon_sec: float | None = None):
        self.sigma_scale = float(sigma_scale)
        self.horizon_sec = horizon_sec

    def solve(self, request: SolverRequest) -> SolverResult:
        position = np.asarray(request.current_position, dtype=float).reshape(3)
        velocity = np.asarray(request.current_velocity, dtype=float).reshape(3)
        horizon = (
            float(self.horizon_sec)
            if self.horizon_sec is not None
            else float(request.motion_config.prediction_dt_sec)
        )
        nominal_future_position = position + velocity * horizon
        covariance = np.asarray(request.velocity_covariance, dtype=float).reshape(3, 3)
        covariance = 0.5 * (covariance + covariance.T)

        values, vectors = np.linalg.eigh(covariance)
        dominant_index = int(np.argmax(values))
        dominant_variance = max(0.0, float(values[dominant_index]))

        if dominant_variance <= 1e-15:
            return SolverResult(
                estimated_position=nominal_future_position.copy(),
                metadata={
                    "mode": "principal_sigma",
                    "sigma_velocity_mps": 0.0,
                    "horizon_sec": horizon,
                    "offset_m": 0.0,
                    "covariance_scope": request.covariance_scope.value,
                },
            )

        axis = vectors[:, dominant_index].astype(float)
        # Eigenvectors have arbitrary sign.  Normalize the sign so playback is
        # deterministic and the marker does not randomly flip sides.
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            axis = -axis

        sigma_velocity = float(np.sqrt(dominant_variance))
        offset = axis * sigma_velocity * horizon * self.sigma_scale

        return SolverResult(
            estimated_position=nominal_future_position + offset,
            metadata={
                "mode": "principal_sigma",
                "sigma_velocity_mps": sigma_velocity,
                "horizon_sec": horizon,
                "offset_m": float(np.linalg.norm(offset)),
                "axis": axis.tolist(),
                "covariance_scope": request.covariance_scope.value,
            },
        )
