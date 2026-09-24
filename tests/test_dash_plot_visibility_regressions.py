from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.plotting import build_terrain_figure
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def test_terrain_has_no_colorbar_or_grid():
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

    heatmap = figure.data[0]
    assert heatmap.type == "heatmap"
    assert heatmap.showscale is False
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.yaxis.showgrid is False
    assert figure.layout.xaxis.zeroline is False
    assert figure.layout.yaxis.zeroline is False


def test_covariance_ellipses_are_non_degenerate_and_visibly_styled():
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

    rolling = [
        trace for trace in figure.data
        if getattr(trace, "name", None) == "Rolling covariance 1σ"
    ]
    fixed = [
        trace for trace in figure.data
        if getattr(trace, "name", None) == "Fixed model covariance 1σ"
    ]

    assert len(rolling) == 5
    assert len(fixed) == 5

    rolling_final = rolling[-1]
    fixed_final = fixed[-1]

    assert np.ptp(np.asarray(rolling_final.x, dtype=float)) > 0.1
    assert np.ptp(np.asarray(rolling_final.y, dtype=float)) > 0.1
    assert np.ptp(np.asarray(fixed_final.x, dtype=float)) > 0.1
    assert np.ptp(np.asarray(fixed_final.y, dtype=float)) > 0.1

    assert rolling_final.fill == "toself"
    assert fixed_final.fill == "toself"
    assert rolling_final.line.width >= 3.0
    assert fixed_final.line.width >= 3.0


def test_dropdown_selected_value_is_dark_on_light_surface():
    css = (ROOT / "trajectory_dash" / "assets" / "style.css").read_text(
        encoding="utf-8"
    )

    assert ".friendly-dropdown .Select-value-label" in css
    assert "color: #1f2937 !important;" in css
    assert "background-color: #ffffff !important;" in css
