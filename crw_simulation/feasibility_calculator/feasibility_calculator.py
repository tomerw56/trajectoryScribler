from abc import ABC, abstractmethod

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely import contains_xy

from crw_simulation.common_geometry.horizontal_face import HorizontalPolygonFace

from crw_simulation.common_geometry.point import Point, PointType

from crw_simulation.event_log import log_construction


from .horizontal_face_feasibility import is_point_on_horizontal_face



class FeasibilityCalculator(ABC):

	@property
	@abstractmethod
	def dimension(self) -> int:
		raise NotImplementedError

	@property
	@abstractmethod
	def point_type(self) -> PointType:
		raise NotImplementedError

	@abstractmethod
	def is_feasible_point(self, point: Point) -> bool:
		raise NotImplementedError



@dataclass(frozen=True, init=False)
class FeasibilityCalculator_XY_2D(FeasibilityCalculator):

	def __init__(self, infeasible_areas: Sequence[HorizontalPolygonFace]) -> None:
		parameters = {"infeasible_areas": infeasible_areas}
		with log_construction(
			"feasibility_calculator",
			type(self).__name__,
			parameters,
		):
			if not isinstance(infeasible_areas, Sequence):
				raise TypeError("infeasible_areas must be a sequence")
			areas = tuple(infeasible_areas)
			parameters["infeasible_areas"] = areas
			if not all(
				isinstance(area, HorizontalPolygonFace)
				and area.dimension == 2
				and area.point_type is PointType.XY_2D
				for area in areas
			):
				raise ValueError(
					"infeasible_areas must contain only 2D XY horizontal polygons"
				)
			object.__setattr__(self, "_infeasible_areas", areas)

	@property
	def infeasible_areas(self) -> tuple[HorizontalPolygonFace, ...]:
		return self._infeasible_areas

	@property
	def dimension(self) -> int:
		return 2

	@property
	def point_type(self) -> PointType:
		return PointType.XY_2D

	def is_feasible_point(self, point: Point) -> bool:
		if not isinstance(point, Point):
			raise TypeError("point must be a Point")

		if point.dimension != self.dimension:
			raise ValueError("point dimension must match calculator dimension")

		if point.point_type is not self.point_type:
			raise ValueError("point type must match calculator point type")

		return not any(
			is_point_on_horizontal_face(area, point)
			for area in self._infeasible_areas
		)

	def is_feasible(self, point: Point) -> bool:
		"""Return feasibility using the shorter calculator method name."""
		return self.is_feasible_point(point)

	def is_feasible_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""Return feasibility for matching arrays of 2D XY coordinates."""
		x_values = np.asarray(x, dtype=float)
		y_values = np.asarray(y, dtype=float)
		if x_values.shape != y_values.shape:
			raise ValueError("x and y must have the same shape")
		feasible = np.ones(x_values.shape, dtype=bool)
		for area in self._infeasible_areas:
			feasible &= ~contains_xy(area.shapely_2d_xy_geometry, x_values, y_values)
		return feasible

