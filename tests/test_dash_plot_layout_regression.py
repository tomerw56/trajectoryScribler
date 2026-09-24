from __future__ import annotations

from pathlib import Path

from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.plotting import build_terrain_figure
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def _figure():
    repository = ScenarioRepository(ROOT)
    scenario = repository.load(
        ROOT / "examples" / "02_basic_multi_target_patrol.trajectory"
    )
    target = scenario.targets[0]
    return build_terrain_figure(
        scenario,
        time_sec=12.0,
        active_target_id=target.id,
        covariance_scope=CovarianceScope.PREDICTION_XY,
        show_rolling_prediction=True,
        show_fixed_prediction=True,
    )


def test_plot_is_not_locked_to_equal_xy_aspect():
    figure = _figure()

    # Dashboard map should stretch to fill the available plot viewport.
    assert figure.layout.xaxis.scaleanchor is None
    assert figure.layout.xaxis.scaleratio is None


def test_legend_is_vertical_and_aligned_to_the_right():
    figure = _figure()

    assert figure.layout.legend.orientation == "v"
    assert figure.layout.legend.x > 1.0
    assert figure.layout.legend.xanchor == "left"
    assert figure.layout.legend.yanchor == "top"


def test_right_margin_reserves_space_for_legend():
    figure = _figure()

    assert figure.layout.margin.r >= 200
