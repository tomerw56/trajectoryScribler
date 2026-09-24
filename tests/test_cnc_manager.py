from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trajectory_app.cnc_manager import CnCManager
from trajectory_app.cnc_solver import (
    CnCObservationProbability,
    CnCSolverRequest,
    CnCSolverResult,
    SnapshotInspectionSolver,
    prediction_offsets_sec,
)
from trajectory_app.covariance_policy import CovarianceScope
from trajectory_dash.runtime import ScenarioRepository


ROOT = Path(__file__).resolve().parents[1]


class CapturingSolver:
    name = "capturing-test-solver"

    def __init__(self):
        self.request: CnCSolverRequest | None = None

    def solve(self, request: CnCSolverRequest) -> CnCSolverResult:
        self.request = request
        probabilities = tuple(
            CnCObservationProbability(
                cameraman_id=cameraman.cameraman_id,
                cameraman_name=cameraman.cameraman_name,
                target_id=bird.target_id,
                target_name=bird.target_name,
                prediction_offset_sec=offset,
                observation_probability=0.5,
            )
            for cameraman in request.cameramen
            for bird in request.birds
            for offset in request.prediction_offsets_sec
        )
        return CnCSolverResult(
            status="ok",
            summary="captured",
            observation_probabilities=probabilities,
            data={"birds": len(request.birds)},
        )


class IncompleteSolver:
    name = "incomplete-test-solver"

    def solve(self, request: CnCSolverRequest) -> CnCSolverResult:
        return CnCSolverResult(
            status="bad",
            summary="intentionally incomplete",
            observation_probabilities=(),
        )


class FakeClock:
    def __init__(self, values):
        self._values = iter(values)

    def __call__(self):
        return next(self._values)


def _scenario():
    repository = ScenarioRepository(ROOT)
    return repository.load(
        ROOT / "examples" / "06_big_top_orbits.trajectory"
    )


def test_prediction_offsets_include_t0_regular_dt_and_exact_horizon():
    assert prediction_offsets_sec(15.0, 5.0) == (0.0, 5.0, 10.0, 15.0)
    assert prediction_offsets_sec(12.0, 5.0) == (0.0, 5.0, 10.0, 12.0)
    assert prediction_offsets_sec(2.0, 5.0) == (0.0, 2.0)


def test_cnc_manager_passes_complete_current_scenario_snapshot():
    scenario = _scenario()
    solver = CapturingSolver()
    manager = CnCManager(
        solver,
        clock_ns=FakeClock([1_000_000_000, 1_025_000_000]),
    )

    execution = manager.solve(
        scenario_name=scenario.path.stem,
        scenario_time_sec=12.0,
        prediction_time_sec=15.0,
        solver_dt_sec=5.0,
        solution_time_sec=0.5,
        covariance_scope=CovarianceScope.FULL_3D,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )

    request = solver.request
    assert request is not None
    assert request.scenario_time_sec == 12.0
    assert request.prediction_time_sec == 15.0
    assert request.solver_dt_sec == 5.0
    assert request.prediction_offsets_sec == (0.0, 5.0, 10.0, 15.0)
    assert request.solution_time_sec == 0.5
    assert request.covariance_scope == CovarianceScope.FULL_3D
    assert len(request.birds) == len(scenario.targets)
    assert len(request.cameramen) == len(scenario.cameramen)
    assert request.terrain.heightmap.shape == scenario.terrain.shape

    first = request.birds[0]
    assert first.position_xyz.shape == (3,)
    assert first.velocity_xyz_mps.shape == (3,)
    assert first.rolling_velocity_covariance.shape == (3, 3)
    assert (
        first.fixed_velocity_covariance is None
        or first.fixed_velocity_covariance.shape == (3, 3)
    )

    expected_probability_count = (
        len(request.cameramen)
        * len(request.birds)
        * len(request.prediction_offsets_sec)
    )
    assert len(execution.result.observation_probabilities) == expected_probability_count

    assert execution.solve_elapsed_sec == pytest.approx(0.025)
    assert execution.exceeded_solution_time is False
    assert manager.last_execution is execution


def test_cnc_timing_is_only_around_solver_call_and_detects_budget_overrun():
    scenario = _scenario()
    solver = CapturingSolver()
    manager = CnCManager(
        solver,
        clock_ns=FakeClock([5_000_000_000, 5_250_000_000]),
    )

    execution = manager.solve(
        scenario_name="timing",
        scenario_time_sec=0.0,
        prediction_time_sec=2.0,
        solver_dt_sec=1.0,
        solution_time_sec=0.1,
        covariance_scope=CovarianceScope.PREDICTION_XY,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )

    assert execution.solve_elapsed_sec == pytest.approx(0.25)
    assert execution.exceeded_solution_time is True


def test_solver_snapshots_are_read_only_copies():
    scenario = _scenario()
    solver = CapturingSolver()
    manager = CnCManager(
        solver,
        clock_ns=FakeClock([1, 2]),
    )

    manager.solve(
        scenario_name="readonly",
        scenario_time_sec=12.0,
        prediction_time_sec=2.0,
        solver_dt_sec=1.0,
        solution_time_sec=1.0,
        covariance_scope=CovarianceScope.PREDICTION_XY,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )

    request = solver.request
    assert request is not None
    assert request.terrain.heightmap.flags.writeable is False
    assert request.birds[0].position_xyz.flags.writeable is False
    assert request.birds[0].velocity_xyz_mps.flags.writeable is False
    assert request.birds[0].rolling_velocity_covariance.flags.writeable is False

    with pytest.raises(ValueError):
        request.birds[0].position_xyz[0] = 999.0


def test_prediction_dt_and_solution_times_are_validated():
    scenario = _scenario()
    manager = CnCManager(
        CapturingSolver(),
        clock_ns=FakeClock([1, 2]),
    )

    common = dict(
        scenario_name="validation",
        scenario_time_sec=0.0,
        covariance_scope=CovarianceScope.PREDICTION_XY,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )

    with pytest.raises(ValueError, match="prediction_time_sec"):
        manager.solve(
            prediction_time_sec=0.0,
            solver_dt_sec=1.0,
            solution_time_sec=1.0,
            **common,
        )

    with pytest.raises(ValueError, match="solver_dt_sec"):
        manager.solve(
            prediction_time_sec=2.0,
            solver_dt_sec=0.0,
            solution_time_sec=1.0,
            **common,
        )

    with pytest.raises(ValueError, match="solution_time_sec"):
        manager.solve(
            prediction_time_sec=2.0,
            solver_dt_sec=1.0,
            solution_time_sec=0.0,
            **common,
        )


def test_manager_rejects_incomplete_probability_grid():
    scenario = _scenario()
    manager = CnCManager(
        IncompleteSolver(),
        clock_ns=FakeClock([1, 2]),
    )

    with pytest.raises(ValueError, match="probability grid"):
        manager.solve(
            scenario_name="incomplete",
            scenario_time_sec=0.0,
            prediction_time_sec=2.0,
            solver_dt_sec=1.0,
            solution_time_sec=1.0,
            covariance_scope=CovarianceScope.PREDICTION_XY,
            targets=scenario.targets,
            cameramen=scenario.cameramen,
            terrain=scenario.terrain,
            terrain_config=scenario.terrain_config,
        )


def test_reference_solver_returns_probability_for_every_tuple():
    scenario = _scenario()
    manager = CnCManager(
        SnapshotInspectionSolver(),
        clock_ns=FakeClock([1, 2]),
    )
    execution = manager.solve(
        scenario_name="reference",
        scenario_time_sec=12.0,
        prediction_time_sec=15.0,
        solver_dt_sec=5.0,
        solution_time_sec=1.0,
        covariance_scope=CovarianceScope.PREDICTION_XY,
        targets=scenario.targets,
        cameramen=scenario.cameramen,
        terrain=scenario.terrain,
        terrain_config=scenario.terrain_config,
    )

    expected = (
        len(execution.request.cameramen)
        * len(execution.request.birds)
        * 4
    )
    assert len(execution.result.observation_probabilities) == expected
    assert all(
        0.0 <= item.observation_probability <= 1.0
        for item in execution.result.observation_probabilities
    )


def test_dash_cnc_panel_wires_dt_display_count_and_probability_tables():
    layout = (
        ROOT / "trajectory_dash" / "layout.py"
    ).read_text(encoding="utf-8")
    app = (
        ROOT / "trajectory_dash" / "app.py"
    ).read_text(encoding="utf-8")

    assert 'id="prediction-horizon-input"' in layout
    assert layout.count('id="prediction-horizon-input"') == 1
    assert 'id="solver-dt-input"' in layout
    assert 'id="cnc-display-steps"' in layout
    assert 'id="solution-time-input"' in layout
    assert 'id="cnc-solve-button"' in layout
    assert 'id="cnc-result-panel"' in layout
    assert 'id="cnc-last-execution"' in layout

    assert 'Input("cnc-solve-button", "n_clicks")' in app
    assert 'State("prediction-horizon-input", "value")' in app
    assert 'State("solver-dt-input", "value")' in app
    assert 'State("solution-time-input", "value")' in app
    assert 'Input("cnc-display-steps", "value")' in app
    assert "cnc_manager.solve(" in app
    assert "cnc-probability-table" in app
