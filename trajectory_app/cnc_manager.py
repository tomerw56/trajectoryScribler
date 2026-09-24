from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter_ns
from typing import Callable, Sequence
from uuid import uuid4

import numpy as np

from .cnc_solver import (
    CnCBirdSnapshot,
    CnCCameramanSnapshot,
    CnCSolver,
    CnCSolverRequest,
    CnCSolverResult,
    CnCTerrainSnapshot,
)
from .covariance_policy import CovarianceScope
from .logging_config import get_logger
from .models import CameramanState, TargetState, TerrainConfig


logger = get_logger(__name__)


@dataclass(frozen=True)
class CnCSolveExecution:
    """One timed invocation of CnCSolver.solve()."""

    solution_id: str
    solver_name: str
    request: CnCSolverRequest
    result: CnCSolverResult
    solve_elapsed_sec: float

    @property
    def exceeded_solution_time(self) -> bool:
        return self.solve_elapsed_sec > self.request.solution_time_sec


class CnCManager:
    """Build a scenario snapshot, call the scenario solver, and time solve()."""

    def __init__(
        self,
        solver: CnCSolver,
        *,
        clock_ns: Callable[[], int] = perf_counter_ns,
    ) -> None:
        self.solver = solver
        self._clock_ns = clock_ns
        self.last_execution: CnCSolveExecution | None = None

    @staticmethod
    def _target_index_at_time(target: TargetState, time_sec: float) -> int:
        if target.samples is None or len(target.samples.time_sec) == 0:
            raise ValueError(
                f"Target {target.name!r} has no motion samples for CnC solve"
            )
        index = int(
            np.searchsorted(
                target.samples.time_sec,
                float(time_sec),
                side="right",
            )
            - 1
        )
        return int(np.clip(index, 0, len(target.samples.time_sec) - 1))

    @staticmethod
    def _rolling_covariance_at_time(
        target: TargetState,
        time_sec: float,
    ) -> np.ndarray:
        if target.covariance is None or len(target.covariance.time_sec) == 0:
            return np.zeros((3, 3), dtype=float)

        index = int(
            np.searchsorted(
                target.covariance.time_sec,
                float(time_sec),
                side="right",
            )
            - 1
        )
        index = int(np.clip(index, 0, len(target.covariance.time_sec) - 1))
        return np.asarray(target.covariance.matrices[index], dtype=float).copy()

    @staticmethod
    def _readonly_array(value, *, shape=None) -> np.ndarray:
        array = np.asarray(value, dtype=float).copy()
        if shape is not None:
            array = array.reshape(shape)
        array.setflags(write=False)
        return array

    def build_request(
        self,
        *,
        scenario_name: str,
        scenario_time_sec: float,
        prediction_time_sec: float,
        solver_dt_sec: float,
        solution_time_sec: float,
        covariance_scope: CovarianceScope,
        targets: Sequence[TargetState],
        cameramen: Sequence[CameramanState],
        terrain: np.ndarray,
        terrain_config: TerrainConfig,
    ) -> CnCSolverRequest:
        prediction_time = float(prediction_time_sec)
        solver_dt = float(solver_dt_sec)
        solution_time = float(solution_time_sec)

        if not np.isfinite(prediction_time) or prediction_time <= 0.0:
            raise ValueError("prediction_time_sec must be finite and > 0")
        if not np.isfinite(solver_dt) or solver_dt <= 0.0:
            raise ValueError("solver_dt_sec must be finite and > 0")
        if not np.isfinite(solution_time) or solution_time <= 0.0:
            raise ValueError("solution_time_sec must be finite and > 0")

        birds: list[CnCBirdSnapshot] = []
        for target in targets:
            if target.samples is None or len(target.samples.time_sec) == 0:
                continue

            index = self._target_index_at_time(target, scenario_time_sec)
            samples = target.samples
            sample_time = float(samples.time_sec[index])
            fixed_covariance = (
                None
                if target.fixed_velocity_covariance is None
                else self._readonly_array(
                    target.fixed_velocity_covariance,
                    shape=(3, 3),
                )
            )

            birds.append(
                CnCBirdSnapshot(
                    target_id=target.id,
                    target_name=target.name,
                    sample_index=index,
                    sample_time_sec=sample_time,
                    position_xyz=self._readonly_array(
                        samples.position[index],
                        shape=(3,),
                    ),
                    velocity_xyz_mps=self._readonly_array(
                        samples.velocity[index],
                        shape=(3,),
                    ),
                    rolling_velocity_covariance=self._readonly_array(
                        self._rolling_covariance_at_time(target, sample_time),
                        shape=(3, 3),
                    ),
                    fixed_velocity_covariance=fixed_covariance,
                )
            )

        camera_snapshots: list[CnCCameramanSnapshot] = []
        for cameraman in cameramen:
            position = (
                None
                if cameraman.position_xyz is None
                else self._readonly_array(cameraman.position_xyz, shape=(3,))
            )
            camera_snapshots.append(
                CnCCameramanSnapshot(
                    cameraman_id=cameraman.id,
                    cameraman_name=cameraman.name,
                    position_xyz=position,
                    view_radius_m=float(cameraman.view_radius_m),
                    view_start_angle_deg=float(cameraman.view_start_angle_deg),
                    view_end_angle_deg=float(cameraman.view_end_angle_deg),
                )
            )

        terrain_copy = self._readonly_array(terrain)
        terrain_snapshot = CnCTerrainSnapshot(
            heightmap=terrain_copy,
            x_min=float(terrain_config.x_min),
            x_max=float(terrain_config.x_max),
            y_min=float(terrain_config.y_min),
            y_max=float(terrain_config.y_max),
            x_cell_size=float(terrain_config.x_cell_size),
            y_cell_size=float(terrain_config.y_cell_size),
        )

        return CnCSolverRequest(
            scenario_name=str(scenario_name),
            scenario_time_sec=float(scenario_time_sec),
            prediction_time_sec=prediction_time,
            solver_dt_sec=solver_dt,
            solution_time_sec=solution_time,
            covariance_scope=covariance_scope,
            birds=tuple(birds),
            cameramen=tuple(camera_snapshots),
            terrain=terrain_snapshot,
        )

    @staticmethod
    def _validate_observation_probability_grid(
        request: CnCSolverRequest,
        result: CnCSolverResult,
    ) -> None:
        """Require exactly one probability for each camera/bird/time tuple."""

        expected_offsets = tuple(float(value) for value in request.prediction_offsets_sec)
        expected = {
            (
                cameraman.cameraman_id,
                bird.target_id,
                offset,
            )
            for cameraman in request.cameramen
            for bird in request.birds
            for offset in expected_offsets
        }

        actual_keys: list[tuple[str, str, float]] = []
        for item in result.observation_probabilities:
            actual_keys.append(
                (
                    item.cameraman_id,
                    item.target_id,
                    float(item.prediction_offset_sec),
                )
            )

        actual = set(actual_keys)
        if len(actual_keys) != len(actual):
            raise ValueError(
                "CnC solver returned duplicate cameraman/bird/prediction-time "
                "probability records"
            )

        missing = expected - actual
        extra = actual - expected
        if missing or extra:
            raise ValueError(
                "CnC solver probability grid does not match the request: "
                f"missing={len(missing)} extra={len(extra)}"
            )

    def solve(
        self,
        *,
        scenario_name: str,
        scenario_time_sec: float,
        prediction_time_sec: float,
        solver_dt_sec: float,
        solution_time_sec: float,
        covariance_scope: CovarianceScope,
        targets: Sequence[TargetState],
        cameramen: Sequence[CameramanState],
        terrain: np.ndarray,
        terrain_config: TerrainConfig,
    ) -> CnCSolveExecution:
        request = self.build_request(
            scenario_name=scenario_name,
            scenario_time_sec=scenario_time_sec,
            prediction_time_sec=prediction_time_sec,
            solver_dt_sec=solver_dt_sec,
            solution_time_sec=solution_time_sec,
            covariance_scope=covariance_scope,
            targets=targets,
            cameramen=cameramen,
            terrain=terrain,
            terrain_config=terrain_config,
        )

        logger.info(
            "CnC solve start: solver=%s scenario=%s t=%.3f birds=%d "
            "cameramen=%d prediction=%.3fs solver_dt=%.3fs "
            "solution_budget=%.3fs",
            self.solver.name,
            request.scenario_name,
            request.scenario_time_sec,
            len(request.birds),
            len(request.cameramen),
            request.prediction_time_sec,
            request.solver_dt_sec,
            request.solution_time_sec,
        )

        start_ns = int(self._clock_ns())
        result = self.solver.solve(request)
        end_ns = int(self._clock_ns())
        elapsed_sec = max(0.0, (end_ns - start_ns) / 1_000_000_000.0)

        self._validate_observation_probability_grid(request, result)

        execution = CnCSolveExecution(
            solution_id=uuid4().hex,
            solver_name=self.solver.name,
            request=request,
            result=result,
            solve_elapsed_sec=elapsed_sec,
        )
        self.last_execution = execution

        logger.info(
            "CnC solve end: solver=%s status=%s elapsed=%.6fs "
            "budget=%.3fs exceeded=%s",
            self.solver.name,
            result.status,
            elapsed_sec,
            request.solution_time_sec,
            execution.exceeded_solution_time,
        )
        return execution
