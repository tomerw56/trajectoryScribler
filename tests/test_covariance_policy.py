from __future__ import annotations

import numpy as np

from trajectory_app.covariance_policy import (
    CovarianceScope,
    projected_covariance_for_canvas,
    scoped_velocity_covariance,
)
from trajectory_app.covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityScope,
    CovarianceStabilityState,
)
from trajectory_app.models import CovarianceSamples


def test_prediction_xy_scope_zeroes_every_z_term_but_keeps_xy_block():
    covariance = np.array(
        [
            [4.0, 0.7, 1.5],
            [0.7, 2.0, -0.9],
            [1.5, -0.9, 25.0],
        ]
    )

    selected = scoped_velocity_covariance(
        covariance,
        CovarianceScope.PREDICTION_XY,
    )

    assert np.allclose(selected[:2, :2], covariance[:2, :2])
    assert np.allclose(selected[2, :], 0.0)
    assert np.allclose(selected[:, 2], 0.0)


def test_full_3d_scope_preserves_complete_symmetric_covariance():
    covariance = np.array(
        [
            [4.0, 0.6, 1.2],
            [0.8, 2.0, -0.4],
            [1.0, -0.8, 9.0],
        ]
    )

    selected = scoped_velocity_covariance(
        covariance,
        CovarianceScope.FULL_3D,
    )

    assert np.allclose(selected, 0.5 * (covariance + covariance.T))


def test_canvas_projection_is_xy_marginal_for_both_modes():
    covariance = np.array(
        [
            [4.0, 0.7, 8.0],
            [0.7, 2.0, -5.0],
            [8.0, -5.0, 30.0],
        ]
    )

    xy = projected_covariance_for_canvas(
        covariance,
        CovarianceScope.PREDICTION_XY,
    )
    full = projected_covariance_for_canvas(
        covariance,
        CovarianceScope.FULL_3D,
    )

    # A top-down 2D covariance ellipse represents the XY marginal.
    assert np.allclose(xy, covariance[:2, :2])
    assert np.allclose(full, covariance[:2, :2])


def test_stability_scope_alias_is_the_system_covariance_scope():
    assert CovarianceStabilityScope is CovarianceScope


def test_stability_xy_and_full_3d_use_same_shared_scope_enum():
    times = np.arange(7.0)
    matrices = np.array(
        [np.diag([1.0, 1.0, z]) for z in [0.1, 4.0, 0.2, 5.0, 0.1, 4.5, 0.1]]
    )
    samples = CovarianceSamples(times, matrices)
    calculator = CovarianceStabilityCalculator()

    xy = calculator.evaluate(samples, end_time_sec=6.0)
    full = calculator.evaluate(
        samples,
        end_time_sec=6.0,
        scope=CovarianceScope.FULL_3D,
    )

    assert xy.scope == CovarianceScope.PREDICTION_XY
    assert xy.state == CovarianceStabilityState.STABLE
    assert full.scope == CovarianceScope.FULL_3D
    assert full.state != CovarianceStabilityState.STABLE
