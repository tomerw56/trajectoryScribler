"""Constant-speed flight with independent left, straight, and right maneuvers."""

from __future__ import annotations

import math

import numpy as np

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.event_log import log_construction

from .base import PathImaginator, initial_xy_arrays, validate_walk_counts, validate_xy_2d_origin
from .initial_data import PositionDirectionInitialData


def simulate_three_choice_turn_flight_2d_xy(
	n_walks: int,
	n_steps: int,
	origin: Point,
	speed: float,
	turning_radius: float,
	time_step: float,
	initial_direction: float,
	seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
	"""Simulate constant-speed paths with equally likely left, straight, and right turns."""
	validate_xy_2d_origin(origin)
	validate_walk_counts(n_walks, n_steps)
	if speed <= 0:
		raise ValueError("speed must be positive")
	if turning_radius <= 0:
		raise ValueError("turning_radius must be positive")
	if time_step <= 0:
		raise ValueError("time_step must be positive")

	rng = np.random.default_rng(seed)
	x, y = initial_xy_arrays(origin, n_walks, n_steps)
	heading = np.full(n_walks, initial_direction, dtype=float)
	turn_angle = speed * time_step / turning_radius

	for step in range(1, n_steps + 1):
		maneuver = rng.choice(np.array([-1, 0, 1]), size=n_walks)
		next_heading = heading + maneuver * turn_angle
		is_straight = maneuver == 0
		safe_maneuver = np.where(is_straight, 1, maneuver)
		arc_x = turning_radius * (np.sin(next_heading) - np.sin(heading)) / safe_maneuver
		arc_y = turning_radius * (np.cos(heading) - np.cos(next_heading)) / safe_maneuver
		straight_x = speed * time_step * np.cos(heading)
		straight_y = speed * time_step * np.sin(heading)

		x[:, step] = x[:, step - 1] + np.where(is_straight, straight_x, arc_x)
		y[:, step] = y[:, step - 1] + np.where(is_straight, straight_y, arc_y)
		heading = next_heading

	return x, y


class ThreeChoiceTurnFlightImaginator(PathImaginator):
	"""``PathImaginator`` wrapping the three-choice turn-flight model."""

	def __init__(
		self,
		speed: float,
		turning_radius: float,
		time_step: float,
		seed: int | None = None,
	) -> None:
		parameters = {
			"speed": speed,
			"turning_radius": turning_radius,
			"time_step": time_step,
			"seed": seed,
		}
		with log_construction("imaginator", type(self).__name__, parameters):
			super().__init__(seed)
			if speed <= 0:
				raise ValueError("speed must be positive")
			if turning_radius <= 0:
				raise ValueError("turning_radius must be positive")
			if time_step <= 0:
				raise ValueError("time_step must be positive")
			self.speed = speed
			self.turning_radius = turning_radius
			self.time_step = time_step

	@property
	def mean_step_length(self) -> float:
		return self.speed * self.time_step

	def _imagine(
		self,
		initial_data: PositionDirectionInitialData,
		n_walks: int,
		n_steps: int,
	) -> tuple[np.ndarray, np.ndarray]:
		if not isinstance(initial_data, PositionDirectionInitialData):
			raise TypeError("initial_data must be PositionDirectionInitialData")
		return simulate_three_choice_turn_flight_2d_xy(
			n_walks=n_walks,
			n_steps=n_steps,
			origin=initial_data.position,
			speed=self.speed,
			turning_radius=self.turning_radius,
			time_step=self.time_step,
			initial_direction=initial_data.direction,
			seed=self.seed,
		)

	def point_at_time(
		self,
		x: np.ndarray,
		y: np.ndarray,
		initial_direction: float,
		t: float,
	) -> Point:
		"""Return the exact point on a single rollout at time ``t``.

		``x``/``y`` are the 1D endpoint arrays for one walk (e.g. one row
		sliced from ``imagine()``'s output), and ``initial_direction`` is the
		heading used to generate that rollout. ``t`` may fall strictly
		between two sampled steps; the in-between point is reconstructed
		from this model's exact arc/straight-line motion.
		"""
		x = np.asarray(x, dtype=float)
		y = np.asarray(y, dtype=float)
		if x.shape != y.shape:
			raise ValueError("x and y must have the same shape")
		if x.ndim != 1:
			raise ValueError("x and y must be one-dimensional")
		if x.shape[0] < 2:
			raise ValueError("x and y must have at least two points")

		n_steps = x.shape[0] - 1
		duration = n_steps * self.time_step
		if t < 0 or t > duration:
			raise ValueError(f"t must be between 0 and {duration}")

		scaled = t / self.time_step
		nearest_step = round(scaled)
		if math.isclose(scaled, nearest_step, abs_tol=1e-9):
			return Point(
				{"x": float(x[nearest_step]), "y": float(y[nearest_step])}, PointType.XY_2D,
			)

		step = int(math.floor(scaled))
		fraction = scaled - step
		elapsed = fraction * self.time_step
		turn_angle = self.speed * self.time_step / self.turning_radius

		heading = float(initial_direction)
		for i in range(step):
			maneuver = self._classify_maneuver(heading, x[i], y[i], x[i + 1], y[i + 1])
			heading += maneuver * turn_angle

		maneuver = self._classify_maneuver(
			heading, x[step], y[step], x[step + 1], y[step + 1],
		)
		if maneuver == 0:
			px = x[step] + self.speed * elapsed * math.cos(heading)
			py = y[step] + self.speed * elapsed * math.sin(heading)
		else:
			angle_covered = maneuver * (self.speed * elapsed / self.turning_radius)
			next_heading = heading + angle_covered
			px = x[step] + self.turning_radius * (math.sin(next_heading) - math.sin(heading)) / maneuver
			py = y[step] + self.turning_radius * (math.cos(heading) - math.cos(next_heading)) / maneuver

		return Point({"x": float(px), "y": float(py)}, PointType.XY_2D)

	def _classify_maneuver(
		self, heading: float, x0: float, y0: float, x1: float, y1: float,
	) -> int:
		"""Infer which maneuver (-1 right, 0 straight, +1 left) produced this step's chord."""
		turn_angle = self.speed * self.time_step / self.turning_radius
		straight_chord = self.speed * self.time_step
		turn_chord = 2.0 * self.turning_radius * math.sin(turn_angle / 2.0)
		dx, dy = x1 - x0, y1 - y0
		chord_length = math.hypot(dx, dy)
		if math.isclose(chord_length, straight_chord, rel_tol=1e-9, abs_tol=1e-9):
			return 0
		if math.isclose(chord_length, turn_chord, rel_tol=1e-9, abs_tol=1e-9):
			cross = math.cos(heading) * dy - math.sin(heading) * dx
			return 1 if cross > 0 else -1
		raise ValueError(
			"step chord length does not match this model's straight or turn "
			"distance; x/y may not be a rollout from this imaginator"
		)
