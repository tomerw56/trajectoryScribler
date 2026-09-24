from __future__ import annotations

import numpy as np

from trajectory_app.covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityConfig,
    CovarianceStabilityScope,
    CovarianceStabilityState,
)
from trajectory_app.models import CovarianceSamples


def _calculator() -> CovarianceStabilityCalculator:
    return CovarianceStabilityCalculator(
        CovarianceStabilityConfig(
            window_sec=4.0,
            min_samples=3,
            min_window_coverage_fraction=0.75,
            absolute_change_deadband=1e-6,
            stable_max_drift=0.10,
            stable_max_rms_change=0.06,
            stable_max_single_jump=0.08,
            changing_max_drift=0.35,
            changing_max_rms_change=0.18,
            changing_max_single_jump=0.25,
        )
    )


def test_default_scope_is_xy_and_ignores_large_z_only_changes():
    times = np.arange(0.0, 5.0, 1.0)
    matrices = []
    for z_variance in [0.2, 3.0, 0.4, 4.5, 0.1]:
        matrix = np.array(
            [
                [1.0, 0.05, 0.0],
                [0.05, 0.8, 0.0],
                [0.0, 0.0, z_variance],
            ],
            dtype=float,
        )
        matrices.append(matrix)

    samples = CovarianceSamples(times, np.asarray(matrices))
    result = _calculator().evaluate(samples, end_time_sec=4.0)

    assert result.scope == CovarianceStabilityScope.PREDICTION_XY
    assert result.state == CovarianceStabilityState.STABLE
    assert result.drift == 0.0
    assert result.rms_change == 0.0
    assert result.max_single_jump == 0.0


def test_full_3d_scope_detects_same_z_changes_as_unstable():
    times = np.arange(0.0, 5.0, 1.0)
    matrices = np.array(
        [
            np.diag([1.0, 0.8, z])
            for z in [0.2, 3.0, 0.4, 4.5, 0.1]
        ],
        dtype=float,
    )
    samples = CovarianceSamples(times, matrices)

    result = _calculator().evaluate(
        samples,
        end_time_sec=4.0,
        scope=CovarianceStabilityScope.FULL_3D,
    )

    assert result.scope == CovarianceStabilityScope.FULL_3D
    assert result.state == CovarianceStabilityState.UNSTABLE


def test_absolute_deadband_suppresses_near_zero_numerical_changes():
    times = np.arange(0.0, 5.0, 1.0)
    matrices = np.zeros((5, 3, 3), dtype=float)
    matrices[1, 0, 0] = 2e-8
    matrices[2, 1, 1] = -3e-8
    matrices[3, 0, 1] = 4e-8
    matrices[3, 1, 0] = 4e-8
    matrices[4, 0, 0] = 5e-8

    samples = CovarianceSamples(times, matrices)
    result = _calculator().evaluate(samples, end_time_sec=4.0)

    assert result.state == CovarianceStabilityState.STABLE
    assert result.drift == 0.0
    assert result.rms_change == 0.0
    assert result.max_single_jump == 0.0


def test_single_xy_jump_is_detected_as_changing_by_default_scope():
    times = np.arange(0.0, 5.0, 1.0)
    matrices = np.array(
        [
            np.diag([x, y, 0.1])
            for x, y in [
                (1.0, 0.8),
                (1.0, 0.8),
                (1.0, 0.8),
                (1.0, 0.8),
                (2.5, 0.2),
            ]
        ],
        dtype=float,
    )
    samples = CovarianceSamples(times, matrices)

    result = _calculator().evaluate(samples, end_time_sec=4.0)

    assert result.scope == CovarianceStabilityScope.PREDICTION_XY
    assert result.state == CovarianceStabilityState.CHANGING
    assert result.max_single_jump > 0.0
