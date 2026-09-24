from __future__ import annotations

from pathlib import Path

from trajectory_app.cnc_manager import CnCManager
from trajectory_app.cnc_solver import SnapshotInspectionSolver
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


def _scenario():
    return ScenarioRepository(ROOT).load(
        ROOT / "examples" / "06_big_top_orbits.trajectory"
    )


def _solve(manager: CnCManager):
    scenario = _scenario()
    return manager.solve(
        scenario_name=scenario.path.stem,
        scenario_time_sec=12.0,
        prediction_time_sec=15.0,
        solver_dt_sec=5.0,
        solution_time_sec=1.0,
        covariance_scope=CovarianceScope.FULL_3D,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )


def test_each_cnc_solve_has_a_unique_solution_id():
    manager = CnCManager(SnapshotInspectionSolver())
    first = _solve(manager)
    second = _solve(manager)

    assert first.solution_id
    assert second.solution_id
    assert first.solution_id != second.solution_id


def test_dash_layout_has_selection_and_acceptance_stores():
    source = (ROOT / "trajectory_dash" / "layout.py").read_text(encoding="utf-8")

    assert 'id="cnc-selection-state"' in source
    assert 'id="cnc-accepted-state"' in source


def test_dash_probability_cells_are_clickable_pattern_components():
    source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert '"type": "cnc-probability-cell"' in source
    assert '"n_clicks"' in source
    assert 'Output("cnc-selection-state", "data")' in source
    assert 'Input(' in source
    assert '"camera": ALL' in source
    assert '"target": ALL' in source
    assert '"offset": ALL' in source


def test_dash_has_one_accept_action_per_cameraman_block():
    source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert '"type": "cnc-accept-button"' in source
    assert 'Output("cnc-accepted-state", "data")' in source
    assert '"Accept selection"' in source
    assert '"Change acceptance"' in source
    assert '"Clear accepted"' in source


def test_acceptance_is_bound_to_solution_identity():
    source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert '"solution_id": execution.solution_id' in source
    assert 'selected.get("solution_id") != execution_data["solution_id"]' in source
    assert 'candidate.get("solution_id") != solution_id' in source


def test_new_solve_resets_ui_selection_and_acceptance():
    source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")

    assert 'Output("cnc-selection-state", "data", allow_duplicate=True)' in source
    assert 'Output("cnc-accepted-state", "data", allow_duplicate=True)' in source
    assert 'return _serialize_cnc_execution(execution), {}, {}' in source


def test_probability_percentage_remains_primary_visual():
    app_source = (ROOT / "trajectory_dash" / "app.py").read_text(encoding="utf-8")
    css = (
        ROOT / "trajectory_dash" / "assets" / "style.css"
    ).read_text(encoding="utf-8")

    assert 'f"{100.0 * probability:.1f}%"' in app_source
    assert '"backgroundColor"' in app_source
    assert "cnc-probability-cell-selected" in css
    assert "cnc-probability-cell-accepted" in css
    assert "cnc-probability-row-selected" in css
    assert "cnc-accepted-summary" in css
