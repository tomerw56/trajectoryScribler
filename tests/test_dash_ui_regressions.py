from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_debug_entry_points_are_available():
    assert (ROOT / "run_dash_debug.ps1").exists()
    assert (ROOT / ".vscode" / "launch.json").exists()

    app_source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")
    assert "use_reloader=False" in app_source
    assert 'parser.add_argument(' in app_source
    assert '"--debug"' in app_source


def test_all_dash_dropdowns_use_friendly_dropdown_class():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    for dropdown_id in (
        "scenario-select",
        "active-target-select",
        "covariance-scope-select",
    ):
        marker = f'id="{dropdown_id}"'
        start = source.index(marker)
        nearby = source[start:start + 350]
        assert 'className="friendly-dropdown"' in nearby


def test_plot_is_allocated_a_large_viewport():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert "height: 72vh;" in css
    assert "min-height: 650px;" in css
    assert ".plot-panel .dash-graph" in css


def test_slider_and_dropdown_have_explicit_readable_light_styles():
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert ".friendly-dropdown .Select-value-label" in css
    assert "color: #1f2937 !important;" in css
    assert "background-color: #ffffff !important;" in css
    assert ".timeline-row .rc-slider-mark-text" in css
    assert "color: #475467 !important;" in css

def test_graph_is_responsive():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    assert "responsive=True" in source
    assert '"responsive": True' in source
