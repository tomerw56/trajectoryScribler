from __future__ import annotations

import numpy as np

from trajectory_app.covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityConfig,
    CovarianceStabilityState,
)
from trajectory_app.models import CovarianceSamples


def _samples(times, scalar_values):
    matrices = np.array(
        [np.diag([value, value * 0.8, value * 0.5]) for value in scalar_values],
        dtype=float,
    )
    return CovarianceSamples(np.asarray(times, dtype=float), matrices)


def _calculator():
    return CovarianceStabilityCalculator(
        CovarianceStabilityConfig(
            window_sec=4.0,
            min_samples=3,
            min_window_coverage_fraction=0.75,
            stable_max_drift=0.10,
            stable_max_rms_change=0.06,
            stable_max_single_jump=0.08,
            changing_max_drift=0.35,
            changing_max_rms_change=0.18,
            changing_max_single_jump=0.25,
        )
    )


def test_returns_insufficient_data_without_enough_window_history():
    samples = _samples([0.0, 1.0], [1.0, 1.0])
    result = _calculator().evaluate(samples, end_time_sec=1.0)
    assert result.state == CovarianceStabilityState.INSUFFICIENT_DATA
    assert not result.is_trustworthy


def test_stable_covariance_returns_stable_enum():
    samples = _samples(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.00, 1.01, 1.00, 1.02, 1.01],
    )
    result = _calculator().evaluate(samples, end_time_sec=4.0)
    assert result.state == CovarianceStabilityState.STABLE
    assert result.is_trustworthy
    assert result.max_single_jump < 0.08


def test_smooth_material_drift_returns_changing():
    samples = _samples(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.00, 1.06, 1.12, 1.18, 1.24],
    )
    result = _calculator().evaluate(samples, end_time_sec=4.0)
    assert result.state == CovarianceStabilityState.CHANGING
    assert result.is_usable_with_caution
    assert not result.is_trustworthy


def test_single_large_recent_jump_returns_changing():
    samples = _samples(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [1.0, 1.0, 1.0, 1.0, 2.2],
    )
    result = _calculator().evaluate(samples, end_time_sec=4.0)
    assert result.state == CovarianceStabilityState.CHANGING
    assert result.max_single_jump > 0.25
    assert "isolated" in result.reason or "caution" in result.reason


def test_old_instability_outside_window_does_not_poison_current_state():
    samples = _samples(
        [-10.0, -9.0, 0.0, 1.0, 2.0, 3.0, 4.0],
        [8.0, 0.2, 1.0, 1.01, 1.0, 1.01, 1.0],
    )
    result = _calculator().evaluate(samples, end_time_sec=4.0)
    assert result.state == CovarianceStabilityState.STABLE


def test_history_returns_only_requested_last_n_seconds():
    samples = _samples(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [1.0, 1.0, 1.0, 1.0, 1.1, 1.2, 1.3],
    )
    history = _calculator().evaluate_history(
        samples,
        last_n_seconds=2.0,
        end_time_sec=6.0,
    )
    assert history.start_time_sec == 4.0
    assert history.end_time_sec == 6.0
    assert [item.end_time_sec for item in history.results] == [4.0, 5.0, 6.0]


def test_normalized_distance_is_zero_for_equal_covariances():
    matrix = np.diag([1.0, 2.0, 3.0])
    distance = CovarianceStabilityCalculator.normalized_frobenius_distance(
        matrix, matrix
    )
    assert distance == 0.0
