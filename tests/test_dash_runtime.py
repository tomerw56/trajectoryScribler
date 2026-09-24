from __future__ import annotations

from pathlib import Path

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.controller import PlaybackState, apply_playback_event
from trajectory_dash.plotting import build_terrain_figure, stability_for_target
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def test_dash_repository_loads_every_packaged_scenario():
    repository = ScenarioRepository(ROOT)
    choices = repository.choices()
    assert len(choices) >= 11

    for choice in choices:
        scenario = repository.load(choice["value"])
        assert scenario.targets
        assert scenario.max_time_sec > 0.0
        assert scenario.playback_step_sec > 0.0
        assert scenario.terrain.size > 0


def test_dash_plot_contains_terrain_targets_cameras_and_predictions():
    repository = ScenarioRepository(ROOT)
    scenario = repository.load(ROOT / "examples" / "06_big_top_orbits.trajectory")
    target = scenario.targets[0]

    figure = build_terrain_figure(
        scenario,
        time_sec=12.0,
        active_target_id=target.id,
        covariance_scope=CovarianceScope.FULL_3D,
        show_rolling_prediction=True,
        show_fixed_prediction=True,
    )

    assert len(figure.data) > len(scenario.targets) + len(scenario.cameramen)
    assert "Reliability:" in figure.layout.title.text
    assert "Full 3D" in figure.layout.title.text


def test_dash_stability_uses_same_backend_calculator():
    repository = ScenarioRepository(ROOT)
    scenario = repository.load(ROOT / "examples" / "06_big_top_orbits.trajectory")
    target = scenario.targets[0]

    result = stability_for_target(
        scenario,
        target,
        12.0,
        CovarianceScope.FULL_3D,
    )

    assert result is not None
    assert result.scope == CovarianceScope.FULL_3D


def test_playback_controller_is_bounded_and_pause_safe():
    state = PlaybackState(0.0, False)
    state = apply_playback_event(
        state,
        "play-button",
        step_sec=0.25,
        max_time_sec=1.0,
    )
    assert state.playing

    for _ in range(5):
        state = apply_playback_event(
            state,
            "playback-interval",
            step_sec=0.25,
            max_time_sec=1.0,
        )

    assert state.time_sec == 1.0
    assert not state.playing

    state = apply_playback_event(
        state,
        "time-slider",
        step_sec=0.25,
        max_time_sec=1.0,
        slider_value=0.4,
    )
    assert state.time_sec == 0.4
    assert not state.playing
