"""First-failure classification for batches of imagined trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from crw_simulation.event_log import write_event
from crw_simulation.feasibility_calculator import FeasibilityCalculator_XY_2D
from crw_simulation.visibility_calculator import PovWorldCalc_XY_2D


@dataclass(frozen=True)
class FirstFailureResults:
	"""First failure event and direct walk groups for a trajectory batch."""

	first_failure_step: np.ndarray
	first_failure_reason: np.ndarray
	first_failure_x: np.ndarray
	first_failure_y: np.ndarray
	infeasible_walks: np.ndarray
	invisible_walks: np.ndarray
	both_failure_walks: np.ndarray
	surviving_walks: np.ndarray

	@property
	def survived(self) -> np.ndarray:
		"""Boolean mask of walks that never failed either constraint."""
		return self.first_failure_reason == "none"

	@property
	def visible_all_steps(self) -> np.ndarray:
		"""Boolean mask of walks that stayed visible for their entire length."""
		return ~np.isin(self.first_failure_reason, ["invisible", "both"])

	@property
	def feasible_all_steps(self) -> np.ndarray:
		"""Boolean mask of walks that stayed feasible for their entire length."""
		return ~np.isin(self.first_failure_reason, ["infeasible", "both"])


def _failure_call_parameters(
	x: np.ndarray,
	y: np.ndarray,
	visibility_calculator: PovWorldCalc_XY_2D,
	feasibility_calculator: FeasibilityCalculator_XY_2D,
) -> dict[str, object]:
	return {
		"x_shape": list(np.shape(x)),
		"y_shape": list(np.shape(y)),
		"cameraman_id": getattr(visibility_calculator, "cameraman_id", None),
		"visibility_origin": getattr(visibility_calculator, "origin", None),
		"visibility_radius": getattr(visibility_calculator, "radius", None),
		"infeasible_areas": getattr(feasibility_calculator, "infeasible_areas", None),
	}


def _failure_result_details(results: FirstFailureResults) -> dict[str, object]:
	total = int(results.first_failure_reason.shape[0])
	counts = {
		reason: int(np.count_nonzero(results.first_failure_reason == reason))
		for reason in ("invisible", "infeasible", "both", "none")
	}
	return {
		"n_walks": total,
		"failure_counts": counts,
		"probability_visible_all_steps": float(results.visible_all_steps.mean()),
		"probability_feasible_all_steps": float(results.feasible_all_steps.mean()),
		"probability_survived": float(results.survived.mean()),
	}


def find_first_failures(
	x: np.ndarray,
	y: np.ndarray,
	visibility_calculator: PovWorldCalc_XY_2D,
	feasibility_calculator: FeasibilityCalculator_XY_2D,
) -> FirstFailureResults:
	"""Record the first infeasible or invisible point for every walk.

	Points are checked in step order starting at step zero. Once a walk fails,
	later points from that walk are ignored. If both calculators reject the
	point at the same first failing step, its reason is ``"both"``.
	"""
	parameters = _failure_call_parameters(
		x, y, visibility_calculator, feasibility_calculator,
	)
	started_at = perf_counter()
	try:
		if x.shape != y.shape:
			raise ValueError("x and y must have the same shape")
		if x.ndim != 2:
			raise ValueError("x and y must be two-dimensional arrays")

		n_walks = x.shape[0]
		first_failure_step = np.full(n_walks, -1, dtype=int)
		first_failure_reason = np.full(n_walks, "none", dtype="<U10")
		first_failure_x = np.full(n_walks, np.nan, dtype=float)
		first_failure_y = np.full(n_walks, np.nan, dtype=float)
		active_walks = np.ones(n_walks, dtype=bool)

		for step in range(x.shape[1]):
			active_indices = np.flatnonzero(active_walks)
			visible = visibility_calculator.is_visible_points(
				x[active_indices, step],
				y[active_indices, step],
			)
			feasible = feasibility_calculator.is_feasible_points(
				x[active_indices, step],
				y[active_indices, step],
			)
			failed_visibility = ~visible
			failed_feasibility = ~feasible
			failed = failed_visibility | failed_feasibility
			failed_indices = active_indices[failed]
			first_failure_step[failed_indices] = step
			first_failure_x[failed_indices] = x[failed_indices, step]
			first_failure_y[failed_indices] = y[failed_indices, step]
			first_failure_reason[
				failed_indices[failed_visibility[failed] & failed_feasibility[failed]]
			] = "both"
			first_failure_reason[
				failed_indices[~failed_visibility[failed] & failed_feasibility[failed]]
			] = "infeasible"
			first_failure_reason[
				failed_indices[failed_visibility[failed] & ~failed_feasibility[failed]]
			] = "invisible"
			active_walks[failed_indices] = False

		results = FirstFailureResults(
			first_failure_step=first_failure_step,
			first_failure_reason=first_failure_reason,
			first_failure_x=first_failure_x,
			first_failure_y=first_failure_y,
			infeasible_walks=np.flatnonzero(first_failure_reason == "infeasible"),
			invisible_walks=np.flatnonzero(first_failure_reason == "invisible"),
			both_failure_walks=np.flatnonzero(first_failure_reason == "both"),
			surviving_walks=np.flatnonzero(first_failure_reason == "none"),
		)
	except Exception as error:
		write_event(
			"find_first_failures.failed",
			"find_first_failures",
			parameters,
			details={"elapsed_ms": (perf_counter() - started_at) * 1000.0},
			error=error,
		)
		raise

	details = _failure_result_details(results)
	details["n_trajectory_points"] = int(x.shape[1])
	details["elapsed_ms"] = (perf_counter() - started_at) * 1000.0
	write_event(
		"find_first_failures.completed",
		"find_first_failures",
		parameters,
		details=details,
	)
	return results
