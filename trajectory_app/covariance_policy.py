from __future__ import annotations

from enum import Enum

import numpy as np


class CovarianceScope(str, Enum):
    """System-wide interpretation of velocity covariance."""

    PREDICTION_XY = "prediction_xy"
    FULL_3D = "full_3d"

    @property
    def display_name(self) -> str:
        if self == CovarianceScope.PREDICTION_XY:
            return "Prediction XY"
        return "Full 3D"

    @property
    def ellipse_label(self) -> str:
        """How the active scope is represented on the 2D terrain canvas."""
        if self == CovarianceScope.PREDICTION_XY:
            return "XY"
        return "3D→XY marginal"


def scoped_velocity_covariance(
    covariance: np.ndarray,
    scope: CovarianceScope = CovarianceScope.PREDICTION_XY,
) -> np.ndarray:
    """Return a symmetric 3x3 covariance honoring the selected system scope.

    PREDICTION_XY keeps only the [Vx, Vy] covariance block and zeros all Z
    variance/cross-covariance terms. This lets every downstream consumer
    (solver, status, prediction ellipse) operate on the same interpretation.

    FULL_3D preserves the complete symmetric 3x3 covariance.
    """
    matrix = np.asarray(covariance, dtype=float).reshape(3, 3)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("velocity covariance must be finite")

    matrix = 0.5 * (matrix + matrix.T)
    selected = CovarianceScope(scope)

    if selected == CovarianceScope.FULL_3D:
        return matrix.copy()

    result = np.zeros((3, 3), dtype=float)
    result[:2, :2] = matrix[:2, :2]
    return result


def projected_covariance_for_canvas(
    covariance: np.ndarray,
    scope: CovarianceScope = CovarianceScope.PREDICTION_XY,
) -> np.ndarray:
    """Return the 2x2 covariance represented by the terrain-view ellipse.

    The terrain canvas is an XY view. In FULL_3D mode the correct 2D drawing
    is therefore the XY marginal of the selected 3D covariance.
    """
    return scoped_velocity_covariance(covariance, scope)[:2, :2].copy()


def covariance_trace_for_scope(
    covariance: np.ndarray,
    scope: CovarianceScope = CovarianceScope.PREDICTION_XY,
) -> float:
    return float(np.trace(scoped_velocity_covariance(covariance, scope)))
