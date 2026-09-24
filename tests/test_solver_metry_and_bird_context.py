from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_solver_metry_tab_exists_and_is_backed_by_last_execution():
    layout = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert 'label="Solver Metry"' in layout
    assert 'value="solver-metry"' in layout
    assert 'id="solver-metry-panel"' in layout

    assert 'Output("solver-metry-panel", "children")' in app
    assert 'Input("cnc-last-execution", "data")' in app
    assert "def _solver_metry_panel" in app


def test_cnc_decision_panel_no_longer_renders_solver_performance_rows():
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    panel_start = app.index("def _cnc_execution_panel")
    panel_end = app.index("\ndef create_app", panel_start)
    panel_source = app[panel_start:panel_end]

    assert '"Measured solve()"' not in panel_source
    assert '"Solution budget"' not in panel_source
    assert '"Scenario time"' not in panel_source
    assert "cnc-result-status" not in panel_source

    assert "cnc-accepted-summary" in panel_source
    assert "cnc-probability-section" in panel_source


def test_solver_metry_contains_requested_execution_metrics():
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    start = app.index("def _solver_metry_panel")
    end = app.index("\ndef _cnc_execution_panel", start)
    source = app[start:end]

    for label in (
        "Status",
        "Solver",
        "Scenario time",
        "Birds",
        "Cameramen",
        "Prediction times",
        "Prediction time",
        "Solver dt",
        "Solution budget",
        "Measured solve()",
        "Budget",
    ):
        assert f'"{label}"' in source


def test_bird_cells_expose_context_menu_targets_and_actions():
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")
    js = (
        ROOT / "trajectory_dash" / "assets" / "cnc_context_menu.js"
    ).read_text(encoding="utf-8")

    assert "cnc-bird-context-target" in app
    assert "cnc-bird-context-hint" in app
    assert '"cnc-context-active"' in app
    assert '"cnc-context-track"' in app
    assert '"cnc-context-best"' in app

    assert 'addEventListener("contextmenu"' in js
    assert "Make active bird" in js
    assert "Track bird" in js
    assert "Select best observation" in js


def test_bird_context_actions_are_wired_to_dash_state():
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert 'Output("active-target-select", "value", allow_duplicate=True)' in app
    assert 'Output("track-state", "data", allow_duplicate=True)' in app
    assert 'Output("cnc-selection-state", "data", allow_duplicate=True)' in app
    assert "def handle_bird_context_action" in app

    # "best" is selection only; acceptance remains explicit.
    start = app.index("def handle_bird_context_action")
    end = app.index("\n    @app.callback(", start)
    source = app[start:end]
    assert "cnc-context-best" in source
    assert "observation_probability" in source
    assert "cnc-accepted-state" not in source


def test_track_button_visuals_follow_track_state_not_only_button_click():
    app = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert "def render_track_button" in app
    assert 'Input("track-state", "data")' in app
