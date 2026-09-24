from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.plotting import build_terrain_figure
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def _scenario_and_target():
    scenario = ScenarioRepository(ROOT).load(
        ROOT / "examples" / "06_big_top_orbits.trajectory"
    )
    return scenario, scenario.targets[0]


def _figure(track: bool, time_sec: float = 12.0):
    scenario, target = _scenario_and_target()
    figure = build_terrain_figure(
        scenario,
        time_sec=time_sec,
        active_target_id=target.id,
        covariance_scope=CovarianceScope.FULL_3D,
        show_rolling_prediction=True,
        show_fixed_prediction=True,
        prediction_horizon_sec=2.0,
        track_active_target=track,
    )
    return scenario, target, figure


def test_track_mode_centers_ranges_on_current_selected_bird():
    scenario, target, figure = _figure(True, 12.0)
    index = scenario.target_index_at_time(target, 12.0)
    position = target.samples.position[index]

    x_range = np.asarray(figure.layout.xaxis.range, dtype=float)
    y_range = np.asarray(figure.layout.yaxis.range, dtype=float)

    np.testing.assert_allclose(np.mean(x_range), position[0], atol=1e-9)
    np.testing.assert_allclose(np.mean(y_range), position[1], atol=1e-9)

    full_x_span = scenario.terrain_config.x_max - scenario.terrain_config.x_min
    full_y_span = scenario.terrain_config.y_max - scenario.terrain_config.y_min
    assert np.ptp(x_range) < full_x_span
    assert np.ptp(y_range) < full_y_span


def test_track_mode_moves_viewport_when_bird_moves():
    scenario, target, first = _figure(True, 8.0)
    _, _, second = _figure(True, 16.0)

    first_x_center = float(np.mean(np.asarray(first.layout.xaxis.range, dtype=float)))
    first_y_center = float(np.mean(np.asarray(first.layout.yaxis.range, dtype=float)))
    second_x_center = float(np.mean(np.asarray(second.layout.xaxis.range, dtype=float)))
    second_y_center = float(np.mean(np.asarray(second.layout.yaxis.range, dtype=float)))

    first_index = scenario.target_index_at_time(target, 8.0)
    second_index = scenario.target_index_at_time(target, 16.0)

    np.testing.assert_allclose(
        [first_x_center, first_y_center],
        target.samples.position[first_index, :2],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        [second_x_center, second_y_center],
        target.samples.position[second_index, :2],
        atol=1e-9,
    )
    assert not np.allclose(
        [first_x_center, first_y_center],
        [second_x_center, second_y_center],
    )


def test_track_off_uses_full_terrain_ranges():
    scenario, _target, figure = _figure(False, 12.0)

    assert list(figure.layout.xaxis.range) == [
        scenario.terrain_config.x_min,
        scenario.terrain_config.x_max,
    ]
    assert list(figure.layout.yaxis.range) == [
        scenario.terrain_config.y_min,
        scenario.terrain_config.y_max,
    ]


def test_tracking_uses_changing_uirevision_so_following_is_not_blocked_by_manual_zoom():
    _scenario, _target, tracked = _figure(True, 12.0)
    _scenario, _target, normal = _figure(False, 12.0)

    assert str(tracked.layout.uirevision).startswith("track:")
    assert not str(normal.layout.uirevision).startswith("track:")


def test_dash_has_track_toggle_state_and_button():
    layout = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert 'id="track-state"' in layout
    assert 'id="track-button"' in layout
    assert 'Output("track-state", "data")' in app
    assert 'Input("track-button", "n_clicks")' in app
    assert 'Input("track-state", "data")' in app
    assert "track_active_target=track_enabled" in app
