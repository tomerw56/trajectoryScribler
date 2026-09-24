from __future__ import annotations

from pathlib import Path

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.plotting import build_terrain_figure
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def test_dash_uses_friendly_light_dropdowns():
    layout = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert 'className="dark-dropdown"' not in layout
    assert layout.count('className="friendly-dropdown"') == 4
    assert ".friendly-dropdown" in css
    assert "background-color: #ffffff !important;" in css
    assert "color: #1f2937 !important;" in css


def test_light_theme_checkboxes_are_visible():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert '.prediction-checklist input[type="checkbox"]' in css
    assert "accent-color: var(--accent);" in css


def test_plotly_figure_uses_white_theme_and_no_grid():
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

    assert figure.layout.paper_bgcolor == "#ffffff"
    assert figure.layout.plot_bgcolor == "#ffffff"
    assert figure.layout.xaxis.showgrid is False
    assert figure.layout.yaxis.showgrid is False
    assert figure.data[0].showscale is False
