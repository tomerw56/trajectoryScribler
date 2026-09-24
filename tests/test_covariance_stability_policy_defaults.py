from __future__ import annotations

import numpy as np

from trajectory_app.covariance_stability import (
    CovarianceStabilityCalculator,
    CovarianceStabilityConfig,
    CovarianceStabilityState,
)
from trajectory_app.models import CovarianceSamples


def _constant_xy_with_small_noise() -> CovarianceSamples:
    times = np.arange(0.0, 12.5, 0.5)
    matrices = []
    for index, _ in enumerate(times):
        tiny = (index % 3 - 1) * 2e-5
        matrix = np.zeros((3, 3), dtype=float)
        matrix[0, 0] = 2e-4 + tiny
        matrix[1, 1] = 1e-4 - tiny * 0.5
        matrices.append(matrix)
    return CovarianceSamples(times, np.asarray(matrices))


def test_default_thresholds_are_deliberately_less_strict():
    config = CovarianceStabilityConfig()

    assert config.normalization_floor == 0.005
    assert config.window_sec == 6.0
    assert config.stable_max_drift == 0.40
    assert config.stable_max_rms_change == 0.15
    assert config.stable_max_single_jump == 0.35
    assert config.changing_max_drift == 1.50
    assert config.changing_max_rms_change == 0.35
    assert config.changing_max_single_jump == 0.80
    assert config.unstable_large_change_fraction == 0.25
    assert config.unstable_min_roughness == 0.35


def test_normalization_floor_keeps_tiny_covariance_variation_stable():
    result = CovarianceStabilityCalculator().evaluate(
        _constant_xy_with_small_noise(),
        end_time_sec=12.0,
        source_warmup_sec=3.0,
    )

    assert result.state == CovarianceStabilityState.STABLE
    assert result.max_single_jump < 0.25


def test_source_warmup_plus_stability_window_returns_insufficient_data():
    samples = _constant_xy_with_small_noise()
    calculator = CovarianceStabilityCalculator()

    before_ready = calculator.evaluate(
        samples,
        end_time_sec=10.5,
        source_warmup_sec=5.0,
    )
    ready = calculator.evaluate(
        samples,
        end_time_sec=11.0,
        source_warmup_sec=5.0,
    )

    assert before_ready.state == CovarianceStabilityState.INSUFFICIENT_DATA
    assert "warming up" in before_ready.reason
    assert before_ready.source_warmup_sec == 5.0

    assert ready.state != CovarianceStabilityState.INSUFFICIENT_DATA


def test_one_isolated_large_jump_is_changing_not_unstable():
    times = np.arange(0.0, 16.0, 0.5)
    matrices = np.array(
        [
            np.diag([0.05, 0.04, 0.0]) if time < 14.5
            else np.diag([2.0, 1.5, 0.0])
            for time in times
        ],
        dtype=float,
    )
    result = CovarianceStabilityCalculator().evaluate(
        CovarianceSamples(times, matrices),
        end_time_sec=15.5,
        source_warmup_sec=5.0,
    )

    assert result.state == CovarianceStabilityState.CHANGING
    assert result.max_single_jump > 0.80
    assert result.large_change_fraction < 0.25
    assert "isolated" in result.reason


def test_repeated_large_changes_are_unstable():
    times = np.arange(0.0, 16.0, 0.5)
    matrices = []
    for time in times:
        if time < 11.0:
            matrices.append(np.diag([0.05, 0.04, 0.0]))
        else:
            # Alternate between materially different covariance shapes.
            if int(round(time * 2)) % 2:
                matrices.append(np.diag([2.0, 0.15, 0.0]))
            else:
                matrices.append(np.diag([0.15, 2.0, 0.0]))

    result = CovarianceStabilityCalculator().evaluate(
        CovarianceSamples(times, np.asarray(matrices, dtype=float)),
        end_time_sec=15.5,
        source_warmup_sec=5.0,
    )

    assert result.state == CovarianceStabilityState.UNSTABLE
    assert result.large_change_fraction >= 0.25
    assert "repeated" in result.reason or "sustained" in result.reason
