"""Correlated random-walk path imagination."""

from __future__ import annotations

import numpy as np

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.event_log import log_construction

from .base import PathImaginator, initial_xy_arrays, validate_walk_counts, validate_xy_2d_origin
from .initial_data import PositionDirectionInitialData


def simulate_crw_2d_xy(
	n_walks: int,
	n_steps: int,
	step_length: float,
	angular_std: float,
	origin: Point,
	initial_direction: float,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Simulate and retain every step of correlated random-walk trajectories.

	The returned arrays have shape ``(n_walks, n_steps + 1)``. For each
	trajectory, column ``0`` is ``origin`` and column ``1`` is the
	first step in ``initial_direction``. Every later step reuses the previous
	heading and adds an independent Gaussian turning angle with
	standard deviation ``angular_std``::

	``initial_direction`` is an angle in radians measured counterclockwise from
	the positive x-axis. Therefore, ``0`` points right, ``pi / 2`` points up,
	``pi`` points left, and ``3 * pi / 2`` points down. Negative angles and
	angles larger than ``2 * pi`` are also accepted and follow the usual
	periodic trigonometric interpretation.
	    heading[1] = initial_direction
	    x[1] = origin.x + step_length * cos(initial_direction)
	    y[1] = origin.y + step_length * sin(initial_direction)

	    heading[t] = heading[t - 1] + Normal(0, angular_std)
	    x[t] = x[t - 1] + step_length * cos(heading[t])
	    y[t] = y[t - 1] + step_length * sin(heading[t])

	The complete trajectory is retained so callers can inspect the state after
	each step or apply calculator-based constraints to every position. Pass a
	``seed`` for a reproducible walkthrough or test; leave it as ``None`` for
	new random trajectories.

	Examples
	--------
	Inspect one trajectory after every step::
	    origin = Point({"x": 10.0, "y": 20.0}, PointType.XY_2D)
	    x, y = simulate_crw_2d_xy(
	        1, 3, 1.0, 0.3, origin, np.pi / 2, seed=7
	    )
	    for step in range(x.shape[1]):
	        print(step, x[0, step], y[0, step])

	The first two rows are ``origin`` and the first step in
	``initial_direction``. The remaining rows show the progressively turned
	path after each subsequent step.
	"""

	if not isinstance(initial_direction, (int, float)):
		raise TypeError("initial_direction must be a real number in radians")
	validate_xy_2d_origin(origin)
	validate_walk_counts(n_walks, n_steps)
	if step_length <= 0:
		raise ValueError("step_length must be positive")
	if angular_std < 0:
		raise ValueError("angular_std cannot be negative")

	rng = np.random.default_rng(seed)

	x, y = initial_xy_arrays(origin, n_walks, n_steps)

	heading = np.full(n_walks, initial_direction)
	x[:, 1] = origin.x + step_length * np.cos(initial_direction)
	y[:, 1] = origin.y + step_length * np.sin(initial_direction)

	for step in range(2, n_steps + 1):
		heading += rng.normal(0, angular_std, n_walks)
		x[:, step] = x[:, step - 1] + step_length * np.cos(heading)
		y[:, step] = y[:, step - 1] + step_length * np.sin(heading)

	return x, y


def print_simulation_steps() -> None:
	"""Print a small seeded trajectory to explain the simulation step by step."""
	x, y = simulate_crw_2d_xy(
		n_walks=1,
		n_steps=4,
		step_length=1.0,
		angular_std=0.3,
		origin=Point({"x": 10.0, "y": 20.0}, PointType.XY_2D),
		initial_direction=np.pi / 2,
		seed=7,
	)

	for step in range(x.shape[1]):
		print(f"step {step}: x={x[0, step]:.4f}, y={y[0, step]:.4f}")


class CorrelatedRandomWalkImaginator(PathImaginator):
	"""``PathImaginator`` wrapping the correlated random walk model."""

	def __init__(
		self,
		step_length: float,
		angular_std: float,
		seed: int | None = None,
	) -> None:
		parameters = {
			"step_length": step_length,
			"angular_std": angular_std,
			"seed": seed,
		}
		with log_construction("imaginator", type(self).__name__, parameters):
			super().__init__(seed)
			if step_length <= 0:
				raise ValueError("step_length must be positive")
			if angular_std < 0:
				raise ValueError("angular_std cannot be negative")
			self.step_length = step_length
			self.angular_std = angular_std

	@property
	def mean_step_length(self) -> float:
		return self.step_length

	def _imagine(self, initial_data: PositionDirectionInitialData, n_walks: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
		if not isinstance(initial_data, PositionDirectionInitialData):
			raise TypeError("initial_data must be PositionDirectionInitialData")
		return simulate_crw_2d_xy(
			n_walks=n_walks,
			n_steps=n_steps,
			step_length=self.step_length,
			angular_std=self.angular_std,
			origin=initial_data.position,
			initial_direction=initial_data.direction,
			seed=self.seed,
		)

