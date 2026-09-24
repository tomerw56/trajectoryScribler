"""Shared contract for path-imagination algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

import numpy as np

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.event_log import write_event

from crw_simulation.path_imagination.initial_data import (
	PositionDirectionInitialData,
	PositionDirectionTurningRateInitialData,
	PositionInitialData,
)


def validate_xy_2d_origin(origin: Point) -> None:
	"""Raise unless ``origin`` is a 2D XY point every model can share."""
	if not isinstance(origin, Point):
		raise TypeError("origin must be a Point")
	if origin.point_type is not PointType.XY_2D:
		raise ValueError("origin must use PointType.XY_2D")
	if origin.horizontal_coordinate_names != frozenset({"x", "y"}):
		raise ValueError("origin requires x and y coordinates")


def validate_walk_counts(n_walks: int, n_steps: int) -> None:
	"""Raise unless ``n_walks``/``n_steps`` are large enough to simulate."""
	if n_walks < 1:
		raise ValueError("n_walks must be at least 1")
	if n_steps < 1:
		raise ValueError("n_steps must be at least 1")


def initial_xy_arrays(
	origin: Point, n_walks: int, n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
	"""Allocate ``(n_walks, n_steps + 1)`` trajectory arrays seeded at ``origin``."""
	x = np.full((n_walks, n_steps + 1), origin.x)
	y = np.full((n_walks, n_steps + 1), origin.y)
	return x, y


class PathImaginator(ABC):
	"""Imagines candidate future trajectories from initial conditions.

	Subclasses take their own motion-model parameters (step length, turning
	behavior, ...) at construction time and implement ``_imagine``.
	``imagine`` validates the arguments shared by every model and returns
	``(x, y)`` arrays of shape ``(n_walks, n_steps + 1)``, where column ``0``
	is ``origin``.
	"""

	def __init__(self, seed: int | None = None) -> None:
		self.seed = seed

	def imagine(
		self,
		initial_data: PositionInitialData | PositionDirectionInitialData | PositionDirectionTurningRateInitialData,
		n_walks: int,
		n_steps: int,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Validate the shared arguments and delegate to the concrete model."""
		parameters = {
			"imaginator_params": dict(vars(self)),
			"initial_data_type": type(initial_data).__name__,
			"initial_data": dict(vars(initial_data))
			if hasattr(initial_data, "__dict__")
			else repr(initial_data),
			"n_walks": n_walks,
			"n_steps": n_steps,
		}
		started_at = perf_counter()
		try:
			validate_walk_counts(n_walks, n_steps)
			validate_xy_2d_origin(initial_data.position)
			x, y = self._imagine(initial_data, n_walks, n_steps)
		except Exception as error:
			write_event(
				"imaginator.imagine.failed",
				f"{type(self).__name__}.imagine",
				parameters,
				details={"elapsed_ms": (perf_counter() - started_at) * 1000.0},
				error=error,
			)
			raise

		write_event(
			"imaginator.imagine.completed",
			f"{type(self).__name__}.imagine",
			parameters,
			details={
				"result_shapes": {"x": list(x.shape), "y": list(y.shape)},
				"elapsed_ms": (perf_counter() - started_at) * 1000.0,
			},
		)
		return x, y

	@property
	@abstractmethod
	def mean_step_length(self) -> float:
		"""Approximate distance covered per step, for cross-model comparison."""

	@abstractmethod
	def _imagine(
		self,
		initial_data: PositionInitialData | PositionDirectionInitialData | PositionDirectionTurningRateInitialData,
		n_walks: int,
		n_steps: int,
	) -> tuple[np.ndarray, np.ndarray]:
		"""Generate the model-specific ``(x, y)`` trajectories."""

