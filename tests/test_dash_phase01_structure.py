from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dash_phase01_files_exist():
    for relative in [
        "dash_app.py",
        "run_dash.ps1",
        "trajectory_dash/app.py",
        "trajectory_dash/layout.py",
        "trajectory_dash/runtime.py",
        "trajectory_dash/plotting.py",
        "trajectory_dash/controller.py",
        "trajectory_dash/analysis.py",
        "trajectory_dash/assets/style.css",
        "DASH_PHASE_01.md",
    ]:
        assert (ROOT / relative).exists(), relative


def test_dash_layout_contains_requested_regions_and_controls():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    assert 'html.Div("CnC Manager", className="panel-title")' in source
    assert 'html.Div("Control", className="dock-title")' in source
    assert 'label="Active Target"' in source
    assert 'label="Telemetry (next)"' in source
    assert 'id="active-target-select"' in source
    assert 'id="covariance-scope-select"' in source
    assert 'id="prediction-visibility"' in source
    assert 'id="terrain-graph"' in source
    assert 'id="play-button"' in source
    assert 'id="time-slider"' in source


def test_dash_dependencies_are_declared():
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "dash>=2.18" in requirements
    assert "plotly>=5.24" in requirements
