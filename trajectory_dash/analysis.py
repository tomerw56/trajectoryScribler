from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from trajectory_app.covariance_policy import CovarianceScope, scoped_velocity_covariance
from trajectory_app.models import TargetState
from trajectory_app.solver import PrincipalSigmaDemoSolver, Solver, SolverRequest

from .runtime import LoadedScenario


@dataclass(frozen=True)
class SolverSnapshot:
    solver_name: str
    rolling_estimate: np.ndarray | None
    fixed_estimate: np.ndarray | None
    future_truth: np.ndarray | None
    rolling_error_m: float | None
    estimate_difference_m: float | None
    mse_m2: float | None
    rmse_m: float | None
    evaluated_samples: int


def future_truth_for_target(
    target: TargetState,
    index: int,
    prediction_horizon_sec: float | None = None,
) -> np.ndarray | None:
    if target.samples is None or len(target.samples.time_sec) == 0:
        return None
    start_time = float(target.samples.time_sec[index])
    horizon = (
        float(target.motion_config.prediction_dt_sec)
        if prediction_horizon_sec is None
        else max(0.0, float(prediction_horizon_sec))
    )
    target_time = start_time + horizon
    times = np.asarray(target.samples.time_sec, dtype=float)
    if target_time < float(times[0]) or target_time > float(times[-1]):
        return None
    positions = np.asarray(target.samples.position, dtype=float)
    return np.array(
        [np.interp(target_time, times, positions[:, axis]) for axis in range(3)],
        dtype=float,
    )


def solve_at_index(
    scenario: LoadedScenario,
    target: TargetState,
    index: int,
    covariance: np.ndarray,
    scope: CovarianceScope,
    solver: Solver,
    prediction_horizon_sec: float | None = None,
) -> np.ndarray:
    if target.samples is None:
        raise ValueError("target has no motion samples")
    samples = target.samples
    active_covariance = scoped_velocity_covariance(covariance, scope)
    request_motion_config = (
        target.motion_config
        if prediction_horizon_sec is None
        else replace(
            target.motion_config,
            prediction_dt_sec=max(0.0, float(prediction_horizon_sec)),
        )
    )
    request = SolverRequest(
        target_id=target.id,
        target_name=target.name,
        sample_index=index,
        time_sec=float(samples.time_sec[index]),
        current_position=samples.position[index].copy(),
        current_velocity=samples.velocity[index].copy(),
        velocity_covariance=active_covariance,
        terrain_height_m=float(samples.terrain_z[index]),
        motion_config=request_motion_config,
        trajectory_config=target.trajectory_config,
        covariance_scope=scope,
        previous_estimated_position=None,
    )
    result = solver.solve(request)
    estimate = np.asarray(result.estimated_position, dtype=float).reshape(3)
    if not np.all(np.isfinite(estimate)):
        raise ValueError("solver returned a non-finite estimate")
    return estimate


def solver_snapshot(
    scenario: LoadedScenario,
    target: TargetState,
    time_sec: float,
    scope: CovarianceScope,
    solver: Solver | None = None,
    prediction_horizon_sec: float | None = None,
) -> SolverSnapshot:
    solver = solver or PrincipalSigmaDemoSolver()
    if target.samples is None or len(target.samples.time_sec) == 0:
        return SolverSnapshot(solver.name, None, None, None, None, None, None, None, 0)

    index = scenario.target_index_at_time(target, time_sec)
    sample_time = float(target.samples.time_sec[index])
    rolling_covariance = scenario.covariance_at_time(target, sample_time)
    rolling_estimate = solve_at_index(
        scenario,
        target,
        index,
        rolling_covariance,
        scope,
        solver,
        prediction_horizon_sec,
    )

    fixed_estimate = None
    if target.fixed_velocity_covariance is not None:
        fixed_estimate = solve_at_index(
            scenario,
            target,
            index,
            target.fixed_velocity_covariance,
            scope,
            solver,
            prediction_horizon_sec,
        )

    future_truth = future_truth_for_target(
        target,
        index,
        prediction_horizon_sec,
    )
    rolling_error = (
        float(np.linalg.norm(rolling_estimate - future_truth))
        if future_truth is not None
        else None
    )
    estimate_difference = (
        float(np.linalg.norm(rolling_estimate - fixed_estimate))
        if fixed_estimate is not None
        else None
    )

    squared_errors: list[float] = []
    for evaluation_index in range(index + 1):
        truth = future_truth_for_target(
            target,
            evaluation_index,
            prediction_horizon_sec,
        )
        if truth is None:
            continue
        evaluation_time = float(target.samples.time_sec[evaluation_index])
        covariance = scenario.covariance_at_time(target, evaluation_time)
        estimate = solve_at_index(
            scenario,
            target,
            evaluation_index,
            covariance,
            scope,
            solver,
            prediction_horizon_sec,
        )
        error = estimate - truth
        squared_errors.append(float(np.dot(error, error)))

    if squared_errors:
        mse = float(np.mean(squared_errors))
        rmse = float(np.sqrt(mse))
    else:
        mse = None
        rmse = None

    return SolverSnapshot(
        solver_name=solver.name,
        rolling_estimate=rolling_estimate,
        fixed_estimate=fixed_estimate,
        future_truth=future_truth,
        rolling_error_m=rolling_error,
        estimate_difference_m=estimate_difference,
        mse_m2=mse,
        rmse_m=rmse,
        evaluated_samples=len(squared_errors),
    )
