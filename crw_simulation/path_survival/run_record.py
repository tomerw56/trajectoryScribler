"""Persist a survival-simulation run's outcomes (summary + per-walk results)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from crw_simulation.common_geometry import Point
from crw_simulation.feasibility_calculator import FeasibilityCalculator_XY_2D
from crw_simulation.visibility_calculator import PovWorldCalc_XY_2D

from .first_failure import FirstFailureResults


def _json_safe(value: Any) -> Any:
	"""Convert model/geometry attributes into plain JSON-serializable values."""
	if isinstance(value, Point):
		return {"coordinates": dict(value.coordinates), "point_type": value.point_type.name}
	if isinstance(value, (list, tuple)):
		return [_json_safe(item) for item in value]
	if isinstance(value, dict):
		return {key: _json_safe(item) for key, item in value.items()}
	return value


def imaginator_params(imaginator: object) -> dict[str, Any]:
	"""Serialize a ``PathImaginator``'s constructor attributes into a JSON-safe dict."""
	return {name: _json_safe(value) for name, value in vars(imaginator).items()}


def first_failure_statistics(first_failure_results: FirstFailureResults) -> dict[str, Any]:
	"""Build serializable failure counts, probabilities, and constraint statistics."""
	total = int(first_failure_results.first_failure_reason.shape[0])
	reason_counts = {
		reason: int(np.count_nonzero(first_failure_results.first_failure_reason == reason))
		for reason in ("invisible", "infeasible", "both", "none")
	}
	return {
		"total_trajectories": total,
		"failed_trajectories": total - reason_counts["none"],
		"survived_trajectories": reason_counts["none"],
		"failure_reasons": reason_counts,
		"failure_reason_probabilities": {
			reason: count / total for reason, count in reason_counts.items()
		},
		"probability_visible_all_steps": float(first_failure_results.visible_all_steps.mean()),
		"probability_feasible_all_steps": float(first_failure_results.feasible_all_steps.mean()),
		"probability_survived": float(first_failure_results.survived.mean()),
	}


def build_run_summary(
	*,
	model_name: str,
	imaginator: object,
	n_walks: int,
	n_steps: int,
	visibility_calculator: PovWorldCalc_XY_2D,
	feasibility_calculator: FeasibilityCalculator_XY_2D,
	first_failure_results: FirstFailureResults,
) -> dict[str, Any]:
	"""Build the JSON-serializable summary of one survival-simulation run."""
	if not feasibility_calculator.infeasible_areas:
		raise ValueError("feasibility_calculator must contain at least one infeasible area")

	return {
		"model_name": model_name,
		"model_params": imaginator_params(imaginator),
		"n_walks": n_walks,
		"n_steps": n_steps,
		"circle_radius": float(visibility_calculator.radius),
		"forbidden_polygons": [
			[
				{"x": float(point.x), "y": float(point.y)}
				for point in polygon.points
			]
			for polygon in feasibility_calculator.infeasible_areas
		],
		"probability_visible_all_steps": float(first_failure_results.visible_all_steps.mean()),
		"probability_feasible_all_steps": float(first_failure_results.feasible_all_steps.mean()),
		"probability_survived": float(first_failure_results.survived.mean()),
	}


def save_run_summary(summary: dict[str, Any], output_dir: Path) -> Path:
	"""Save the run summary as ``output_dir/run_summary.json``."""
	output_path = output_dir / "run_summary.json"
	output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
	return output_path


def save_walk_outcomes(first_failure_results: FirstFailureResults, output_dir: Path) -> Path:
	"""Save per-walk outcomes as ``output_dir/walk_outcomes.csv``, one row per walk."""
	output_path = output_dir / "walk_outcomes.csv"
	n_walks = first_failure_results.first_failure_reason.shape[0]
	with output_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow([
			"walk_index",
			"first_failure_step",
			"first_failure_reason",
			"first_failure_x",
			"first_failure_y",
			"survived",
		])
		for walk_index in range(n_walks):
			writer.writerow([
				walk_index,
				int(first_failure_results.first_failure_step[walk_index]),
				str(first_failure_results.first_failure_reason[walk_index]),
				float(first_failure_results.first_failure_x[walk_index]),
				float(first_failure_results.first_failure_y[walk_index]),
				bool(first_failure_results.survived[walk_index]),
			])
	return output_path


def save_walk_paths(x: np.ndarray, y: np.ndarray, output_dir: Path) -> list[Path]:
	"""Save each walk's full ``(step, x, y)`` trajectory as its own CSV file.

	Files are named ``walk_<index>_path.csv`` (zero-padded), pairing with the
	``walk_<index>.html`` visualization of the same walk. Returns the list of
	saved paths, in walk-index order.
	"""
	if x.shape != y.shape:
		raise ValueError("x and y must have the same shape")
	if x.ndim != 2:
		raise ValueError("x and y must be two-dimensional arrays")

	n_walks, n_columns = x.shape
	index_width = len(str(n_walks - 1))

	saved_paths = []
	for walk_index in range(n_walks):
		output_path = output_dir / f"walk_{walk_index:0{index_width}d}_path.csv"
		with output_path.open("w", newline="", encoding="utf-8") as handle:
			writer = csv.writer(handle)
			writer.writerow(["step", "x", "y"])
			for step in range(n_columns):
				writer.writerow([step, float(x[walk_index, step]), float(y[walk_index, step])])
		saved_paths.append(output_path)

	return saved_paths
