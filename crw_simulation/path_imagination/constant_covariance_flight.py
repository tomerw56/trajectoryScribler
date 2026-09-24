"""Flight model with constant Gaussian increment covariance."""

from __future__ import annotations

import numpy as np

from crw_simulation.common_geometry import Point
from crw_simulation.event_log import log_construction

from .base import PathImaginator, initial_xy_arrays, validate_walk_counts, validate_xy_2d_origin
from crw_simulation.path_imagination.initial_data import PositionInitialData


def _validated_mean_step(mean_step: tuple[float, float] | np.ndarray) -> tuple[float, float]:
	values = np.asarray(mean_step, dtype=float)
	if values.shape != (2,):
		raise ValueError("mean_step must contain exactly two values")
	if not np.all(np.isfinite(values)):
		raise ValueError("mean_step must contain only finite values")
	return float(values[0]), float(values[1])


def _validated_covariance(covariance: np.ndarray | list[list[float]]) -> list[list[float]]:
	values = np.asarray(covariance, dtype=float)
	if values.shape != (2, 2):
		raise ValueError("covariance must have shape (2, 2)")
	if not np.all(np.isfinite(values)):
		raise ValueError("covariance must contain only finite values")
	if not np.allclose(values, values.T):
		raise ValueError("covariance must be symmetric")
	if np.any(np.linalg.eigvalsh(values) < -1e-12):
		raise ValueError("covariance must be positive semidefinite")
	return values.tolist()


def simulate_constant_covariance_flight_2d_xy(
	n_walks: int,
	n_steps: int,
	origin: Point,
	mean_step: tuple[float, float] | np.ndarray,
	covariance: np.ndarray | list[list[float]],
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Simulate independent Gaussian flight increments with fixed covariance."""
	validate_xy_2d_origin(origin)
	validate_walk_counts(n_walks, n_steps)
	validated_mean = _validated_mean_step(mean_step)
	validated_covariance = np.asarray(_validated_covariance(covariance), dtype=float)

	rng = np.random.default_rng(seed)
	increments = rng.multivariate_normal(
		mean=validated_mean,
		cov=validated_covariance,
		size=(n_walks, n_steps),
	)

	x, y = initial_xy_arrays(origin, n_walks, n_steps)
	x[:, 1:] = origin.x + np.cumsum(increments[:, :, 0], axis=1)
	y[:, 1:] = origin.y + np.cumsum(increments[:, :, 1], axis=1)
	return x, y


class ConstantCovarianceFlightImaginator(PathImaginator):
	"""``PathImaginator`` wrapping the constant-covariance flight model."""

	def __init__(
		self,
		mean_step: tuple[float, float] | np.ndarray,
		covariance: np.ndarray | list[list[float]],
		seed: int | None = None,
	) -> None:
		parameters = {
			"mean_step": mean_step,
			"covariance": covariance,
			"seed": seed,
		}
		with log_construction("imaginator", type(self).__name__, parameters):
			super().__init__(seed)
			self.mean_step = _validated_mean_step(mean_step)
			self.covariance = _validated_covariance(covariance)
			parameters["mean_step"] = self.mean_step
			parameters["covariance"] = self.covariance

	@property
	def mean_step_length(self) -> float:
		return float(np.linalg.norm(self.mean_step))

	def _imagine(self, initial_data: PositionInitialData, n_walks: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
		if not isinstance(initial_data, PositionInitialData):
			raise TypeError("initial_data must be PositionInitialData")
		return simulate_constant_covariance_flight_2d_xy(
			n_walks=n_walks,
			n_steps=n_steps,
			origin=initial_data.position,
			mean_step=self.mean_step,
			covariance=self.covariance,
			seed=self.seed,
		)
