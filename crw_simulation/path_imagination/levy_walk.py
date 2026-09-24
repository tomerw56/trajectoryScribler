"""Levy walk path imagination.

Unlike the correlated random walk, headings are independent and uniform on
every step; only the step-length distribution has memory-free heavy tails.
"""

from __future__ import annotations

import numpy as np

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.event_log import log_construction

from .base import PathImaginator, initial_xy_arrays, validate_walk_counts, validate_xy_2d_origin
from .initial_data import PositionInitialData


def simulate_levy_walk_2d_xy(
	n_walks: int,
	n_steps: int,
	origin: Point,
	alpha: float,
	step_length_min: float = 1.0,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Simulate 2D Levy walk trajectories with power-law step lengths.

	Step lengths are drawn from a Pareto Type I distribution with shape
	``alpha - 1`` and scale ``step_length_min``, so ``P(L > l) = (step_length_min / l) ** (alpha - 1)``
	for ``l >= step_length_min``. Headings are independently uniform on
	``[0, 2*pi)`` for every step, unlike the correlated random walk's
	Gaussian-turning heading. The returned arrays have shape
	``(n_walks, n_steps + 1)``, with column ``0`` equal to ``origin``.
	"""

	validate_xy_2d_origin(origin)
	validate_walk_counts(n_walks, n_steps)
	if alpha <= 1:
		raise ValueError("alpha must be greater than 1")
	if step_length_min <= 0:
		raise ValueError("step_length_min must be positive")

	rng = np.random.default_rng(seed)
	shape = alpha - 1

	x, y = initial_xy_arrays(origin, n_walks, n_steps)

	for step in range(1, n_steps + 1):
		directions = rng.uniform(0, 2 * np.pi, n_walks)
		# 1 - rng.random() keeps the quantile in (0, 1], avoiding a division by zero.
		step_lengths = step_length_min * (1 - rng.random(n_walks)) ** (-1 / shape)
		x[:, step] = x[:, step - 1] + step_lengths * np.cos(directions)
		y[:, step] = y[:, step - 1] + step_lengths * np.sin(directions)

	return x, y


class LevyWalkImaginator(PathImaginator):
	"""``PathImaginator`` wrapping the Levy walk model."""

	def __init__(
		self,
		alpha: float,
		step_length_min: float = 1.0,
		seed: int | None = None,
	) -> None:
		parameters = {
			"alpha": alpha,
			"step_length_min": step_length_min,
			"seed": seed,
		}
		with log_construction("imaginator", type(self).__name__, parameters):
			super().__init__(seed)
			if alpha <= 1:
				raise ValueError("alpha must be greater than 1")
			if step_length_min <= 0:
				raise ValueError("step_length_min must be positive")
			self.alpha = alpha
			self.step_length_min = step_length_min

	@property
	def mean_step_length(self) -> float:
		# The distribution's true mean is infinite for alpha <= 2; use the scale as an approximation.
		return self.step_length_min

	def _imagine(self, initial_data: PositionInitialData, n_walks: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
		if not isinstance(initial_data, PositionInitialData):
			raise TypeError("initial_data must be PositionInitialData")
		return simulate_levy_walk_2d_xy(
			n_walks=n_walks,
			n_steps=n_steps,
			origin=initial_data.position,
			alpha=self.alpha,
			step_length_min=self.step_length_min,
			seed=self.seed,
		)
