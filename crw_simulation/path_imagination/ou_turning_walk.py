"""Constant-speed walk with an Ornstein-Uhlenbeck turning rate.

Speed is held exactly constant; only the heading's turning rate wanders,
mean-reverting toward zero. Each step advances along the exact circular arc
swept by holding that step's turning rate fixed, so the trajectory's speed
does not depend on the time-step size (unlike a naive Euler update).
"""

from __future__ import annotations

import numpy as np

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.event_log import log_construction

from .base import PathImaginator, initial_xy_arrays, validate_walk_counts, validate_xy_2d_origin
from .initial_data import PositionDirectionTurningRateInitialData


def simulate_ou_turning_walk_2d_xy(
	n_walks: int,
	n_steps: int,
	origin: Point,
	speed: float,
	turning_rate_std: float,
	turning_rate_persistence: float,
	time_step: float,
	initial_direction: float = 0.0,
	initial_turning_rate: float = 0.0,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Simulate constant-speed trajectories whose turning rate is an OU process.

	The turning rate ``omega`` mean-reverts to zero between steps::

	    a = exp(-time_step / turning_rate_persistence)
	    omega[k+1] = a * omega[k] + turning_rate_std * sqrt(1 - a**2) * Z_k

	``turning_rate_persistence`` sets how long turns persist and
	``turning_rate_std`` sets their typical magnitude. Position then advances
	exactly along the circular arc swept by holding ``omega[k+1]`` fixed for
	``time_step``, so every step has arc length ``speed * time_step``
	regardless of ``time_step``::

	    theta[k+1] = theta[k] + omega[k+1] * time_step
	    (x, y) follow the arc of radius speed / omega[k+1] from theta[k] to
	    theta[k+1], or a straight line when omega[k+1] == 0.

	Because chord length is never larger than arc length, the straight-line
	distance between consecutive sampled points can be smaller than
	``speed * time_step``; only the straight-line case (``omega == 0``)
	reaches that bound exactly. The returned arrays have shape
	``(n_walks, n_steps + 1)``, with column ``0`` equal to ``origin``.
	"""

	validate_xy_2d_origin(origin)
	validate_walk_counts(n_walks, n_steps)
	if speed <= 0:
		raise ValueError("speed must be positive")
	if turning_rate_std < 0:
		raise ValueError("turning_rate_std cannot be negative")
	if turning_rate_persistence <= 0:
		raise ValueError("turning_rate_persistence must be positive")
	if time_step <= 0:
		raise ValueError("time_step must be positive")

	rng = np.random.default_rng(seed)
	a = np.exp(-time_step / turning_rate_persistence)
	diffusion = turning_rate_std * np.sqrt(1 - a ** 2)

	x, y = initial_xy_arrays(origin, n_walks, n_steps)
	theta = np.full(n_walks, initial_direction, dtype=float)
	omega = np.full(n_walks, initial_turning_rate, dtype=float)

	for step in range(1, n_steps + 1):
		omega = a * omega + diffusion * rng.normal(size=n_walks)
		next_theta = theta + omega * time_step

		# The arc formula has a removable singularity at omega == 0; use the straight-line limit there.
		omega_safe = np.where(omega == 0, 1.0, omega)
		radius = speed / omega_safe
		arc_dx = radius * (np.sin(next_theta) - np.sin(theta))
		arc_dy = radius * (np.cos(theta) - np.cos(next_theta))
		straight_dx = speed * time_step * np.cos(theta)
		straight_dy = speed * time_step * np.sin(theta)

		x[:, step] = x[:, step - 1] + np.where(omega == 0, straight_dx, arc_dx)
		y[:, step] = y[:, step - 1] + np.where(omega == 0, straight_dy, arc_dy)
		theta = next_theta

	return x, y


class OUTurningWalkImaginator(PathImaginator):
	"""``PathImaginator`` wrapping the OU-turning-rate constant-speed model."""

	def __init__(
		self,
		speed: float,
		turning_rate_std: float,
		turning_rate_persistence: float,
		time_step: float,
		seed: int | None = None,
	) -> None:
		parameters = {
			"speed": speed,
			"turning_rate_std": turning_rate_std,
			"turning_rate_persistence": turning_rate_persistence,
			"time_step": time_step,
			"seed": seed,
		}
		with log_construction("imaginator", type(self).__name__, parameters):
			super().__init__(seed)
			if speed <= 0:
				raise ValueError("speed must be positive")
			if turning_rate_std < 0:
				raise ValueError("turning_rate_std cannot be negative")
			if turning_rate_persistence <= 0:
				raise ValueError("turning_rate_persistence must be positive")
			if time_step <= 0:
				raise ValueError("time_step must be positive")
			self.speed = speed
			self.turning_rate_std = turning_rate_std
			self.turning_rate_persistence = turning_rate_persistence
			self.time_step = time_step

	@property
	def mean_step_length(self) -> float:
		return self.speed * self.time_step

	def _imagine(self, initial_data: PositionDirectionTurningRateInitialData, n_walks: int, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
		if not isinstance(initial_data, PositionDirectionTurningRateInitialData):
			raise TypeError("initial_data must be PositionDirectionTurningRateInitialData")
		return simulate_ou_turning_walk_2d_xy(
			n_walks=n_walks,
			n_steps=n_steps,
			origin=initial_data.position,
			speed=self.speed,
			turning_rate_std=self.turning_rate_std,
			turning_rate_persistence=self.turning_rate_persistence,
			time_step=self.time_step,
			initial_direction=initial_data.direction,
			initial_turning_rate=initial_data.turning_rate,
			seed=self.seed,
		)
