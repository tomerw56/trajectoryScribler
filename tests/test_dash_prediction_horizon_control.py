from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.analysis import solver_snapshot
from trajectory_dash.plotting import build_terrain_figure
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def _scenario_target():
    repository = ScenarioRepository(ROOT)
    scenario = repository.load(
        ROOT / "examples" / "06_big_top_orbits.trajectory"
    )
    return scenario, scenario.targets[0]


def _final_rolling_ellipse_span(figure) -> float:
    traces = [
        trace for trace in figure.data
        if getattr(trace, "name", None) == "Rolling covariance 1σ"
    ]
    assert traces
    final = traces[-1]
    x = np.asarray(final.x, dtype=float)
    y = np.asarray(final.y, dtype=float)
    return float(np.hypot(np.ptp(x), np.ptp(y)))


def test_layout_has_forward_prediction_numeric_updown():
    source = (
        ROOT / "trajectory_dash" / "layout.py"
    ).read_text(encoding="utf-8")

    assert 'id="prediction-horizon-input"' in source
    assert 'type="number"' in source
    assert 'step=0.25' in source
    assert 'min=0.25' in source
    assert 'max=30.0' in source


def test_prediction_horizon_changes_covariance_envelope_size():
    scenario, target = _scenario_target()

    short = build_terrain_figure(
        scenario,
        time_sec=12.0,
        active_target_id=target.id,
        covariance_scope=CovarianceScope.FULL_3D,
        show_rolling_prediction=True,
        show_fixed_prediction=False,
        prediction_horizon_sec=1.0,
    )
    long = build_terrain_figure(
        scenario,
        time_sec=12.0,
        active_target_id=target.id,
        covariance_scope=CovarianceScope.FULL_3D,
        show_rolling_prediction=True,
        show_fixed_prediction=False,
        prediction_horizon_sec=4.0,
    )

    short_span = _final_rolling_ellipse_span(short)
    long_span = _final_rolling_ellipse_span(long)

    assert short_span > 0.0
    assert np.isclose(long_span / short_span, 4.0, rtol=0.03)


def test_prediction_horizon_changes_solver_future_estimate():
    scenario, target = _scenario_target()

    short = solver_snapshot(
        scenario,
        target,
        time_sec=12.0,
        scope=CovarianceScope.FULL_3D,
        prediction_horizon_sec=1.0,
    )
    long = solver_snapshot(
        scenario,
        target,
        time_sec=12.0,
        scope=CovarianceScope.FULL_3D,
        prediction_horizon_sec=4.0,
    )

    assert short.rolling_estimate is not None
    assert long.rolling_estimate is not None
    assert not np.allclose(short.rolling_estimate, long.rolling_estimate)


def test_app_wires_horizon_into_plot_and_sample_panel():
    source = (
        ROOT / "trajectory_dash" / "app.py"
    ).read_text(encoding="utf-8")

    assert 'Input("prediction-horizon-input", "value")' in source
    assert "prediction_horizon_sec=horizon" in source
    assert 'Output("prediction-horizon-input", "value")' in source
