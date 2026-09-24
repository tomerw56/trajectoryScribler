from abc import ABC, abstractmethod

from collections.abc import Sequence
from dataclasses import dataclass

from math import atan2, cos, hypot, isclose, sin, sqrt

from numbers import Real

import numpy as np

from shapely import contains_xy
from shapely.geometry import GeometryCollection, MultiPoint

from shapely.geometry import Point as ShapelyPoint

from shapely.geometry.base import BaseGeometry


from crw_simulation.common_geometry.horizontal_face import HorizontalInverseCircleFace, HorizontalPolygonFace

from crw_simulation.common_geometry.point import Point, PointType

from crw_simulation.common_geometry.pov import Pov

from crw_simulation.event_log import log_construction



class PovWorldCalc(ABC):

	@property
	@abstractmethod
	def dimension(self) -> int:
		raise NotImplementedError

	@property
	@abstractmethod
	def point_type(self) -> PointType:
		raise NotImplementedError

	@abstractmethod
	def is_visible_point(self, point: Point) -> bool:
		raise NotImplementedError



def _extend_ray(origin: Point, target: Point, distance: Real) -> tuple[float, float]:

	dx = target.x - origin.x

	dy = target.y - origin.y

	length = hypot(dx, dy)

	if isclose(length, 0.0):

		raise ValueError("an infeasible polygon cannot contain the world origin")

	return (origin.x + distance * dx / length, origin.y + distance * dy / length)


def _segment_circle_intersections(
	start: Point,
	end: Point,
	origin: Point,
	radius: Real,
) -> list[tuple[float, float]]:
	"""Return all intersections of a polygon side segment and the circle."""
	dx = end.x - start.x
	dy = end.y - start.y
	fx = start.x - origin.x
	fy = start.y - origin.y
	a = dx * dx + dy * dy
	if isclose(a, 0.0):
		return []
	b = 2.0 * (fx * dx + fy * dy)
	c = fx * fx + fy * fy - float(radius) ** 2
	discriminant = b * b - 4.0 * a * c
	if discriminant < 0.0:
		return []

	square_root = sqrt(max(discriminant, 0.0))
	parameters = ((-b - square_root) / (2.0 * a), (-b + square_root) / (2.0 * a))
	intersections = []
	for parameter in parameters:
		if -1e-12 <= parameter <= 1.0 + 1e-12:
			clamped = min(1.0, max(0.0, parameter))
			intersections.append((start.x + clamped * dx, start.y + clamped * dy))
	return intersections


def _tangent_intersection(
	first: tuple[float, float],
	second: tuple[float, float],
	origin: Point,
	radius: Real,
) -> tuple[float, float] | None:
	"""Return the intersection of two circle tangents, if it is finite."""
	first_normal = (first[0] - origin.x, first[1] - origin.y)
	second_normal = (second[0] - origin.x, second[1] - origin.y)
	determinant = first_normal[0] * second_normal[1] - first_normal[1] * second_normal[0]
	if isclose(determinant, 0.0, abs_tol=1e-12):
		return None

	first_rhs = first_normal[0] * first[0] + first_normal[1] * first[1]
	second_rhs = second_normal[0] * second[0] + second_normal[1] * second[1]
	x = (first_rhs * second_normal[1] - first_normal[1] * second_rhs) / determinant
	y = (first_normal[0] * second_rhs - first_rhs * second_normal[0]) / determinant
	return x, y


def _build_obstacle_shadow(area: HorizontalPolygonFace, origin: Point, radius: Real) -> BaseGeometry:
	vertices = list(area.points)
	radius_squared = float(radius) ** 2
	inside_vertices = [
		(point.x, point.y)
		for point in vertices
		if (point.x - origin.x) ** 2 + (point.y - origin.y) ** 2 <= radius_squared + 1e-12
	]

	intersection_points = []
	for start, end in zip(vertices, vertices[1:] + vertices[:1]):
		intersection_points.extend(_segment_circle_intersections(start, end, origin, radius))

	def unique(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
		result = []
		for point in points:
			if not any(hypot(point[0] - other[0], point[1] - other[1]) <= 1e-10 for other in result):
				result.append(point)
		return result

	intersection_points = unique(intersection_points)
	relevant_points = unique(inside_vertices + intersection_points)
	if not relevant_points:
		return GeometryCollection()

	angles = [
		atan2(point[1] - origin.y, point[0] - origin.x) % (2.0 * np.pi)
		for point in relevant_points
	]
	ordered = sorted(zip(angles, relevant_points), key=lambda item: item[0])
	if len(ordered) == 1:
		extreme_points = [ordered[0][1]]
	else:
		gaps = [
			((ordered[(index + 1) % len(ordered)][0] - ordered[index][0]) % (2.0 * np.pi), index)
			for index in range(len(ordered))
		]
		_, largest_gap_index = max(gaps)
		start_index = (largest_gap_index + 1) % len(ordered)
		end_index = largest_gap_index
		extreme_points = [ordered[start_index][1], ordered[end_index][1]]

	shadow_points = list(relevant_points)
	perimeter_extremes = []
	for point in extreme_points:
		distance = hypot(point[0] - origin.x, point[1] - origin.y)
		if isclose(distance, float(radius), rel_tol=1e-10, abs_tol=1e-10):
			perimeter_point = point
		else:
			angle = atan2(point[1] - origin.y, point[0] - origin.x)
			perimeter_point = (
				origin.x + float(radius) * cos(angle),
				origin.y + float(radius) * sin(angle),
			)
			shadow_points.append(perimeter_point)
		perimeter_extremes.append(perimeter_point)

	if len(perimeter_extremes) == 2:
		tangent_point = _tangent_intersection(
			perimeter_extremes[0], perimeter_extremes[1], origin, radius,
		)
		if tangent_point is not None:
			shadow_points.append(tangent_point)

	circle_geometry = ShapelyPoint(origin.x, origin.y).buffer(float(radius), resolution=256)
	clipped_area = area.shapely_2d_xy_geometry.intersection(circle_geometry)
	return MultiPoint(shadow_points).convex_hull.union(clipped_area)


@dataclass(frozen=True, init=False)
class PovWorldCalc_XY_2D(PovWorldCalc):

	def __init__(
		self,
		origin: Point,
		radius: Real,
		infeasible_areas: Sequence[HorizontalPolygonFace],
		cameraman_id: str = "default",
	) -> None:
		parameters = {
			"origin": origin,
			"radius": radius,
			"infeasible_areas": infeasible_areas,
			"cameraman_id": cameraman_id,
		}
		with log_construction(
			"visibility_calculator",
			type(self).__name__,
			parameters,
		):
			if not isinstance(origin, Point):
				raise TypeError("origin must be a Point")
			if origin.point_type is not PointType.XY_2D:
				raise ValueError("PovWorldCalc_XY_2D origin must use PointType.XY_2D")
			if origin.horizontal_coordinate_names != frozenset({"x", "y"}):
				raise ValueError("PovWorldCalc_XY_2D origin requires x and y coordinates")
			if not isinstance(radius, Real):
				raise TypeError("radius must be a real number")
			if radius <= 0:
				raise ValueError("radius must be positive")
			if not isinstance(cameraman_id, str) or not cameraman_id:
				raise ValueError("cameraman_id must be a non-empty string")
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
				raise ValueError("infeasible_areas must contain only 2D XY horizontal polygons")
			object.__setattr__(self, "_point_of_view", Pov(origin))
			object.__setattr__(self, "_origin", origin)
			object.__setattr__(self, "_radius", radius)
			object.__setattr__(self, "_cameraman_id", cameraman_id)
			object.__setattr__(self, "_infeasible_areas", areas)
			object.__setattr__(self, "_far_area", HorizontalInverseCircleFace(origin, radius))
			object.__setattr__(
				self,
				"_nonvisibility_polygons",
				tuple(_build_obstacle_shadow(area, origin, radius) for area in areas),
			)

	@property
	def origin(self) -> Point:
		return self._origin


	@property
	def radius(self) -> Real:
		return self._radius


	@property
	def cameraman_id(self) -> str:
		return self._cameraman_id


	@property
	def infeasible_areas(self) -> tuple[HorizontalPolygonFace, ...]:
		return self._infeasible_areas


	@property
	def dimension(self) -> int:
		return 2


	@property
	def point_type(self) -> PointType:
		return PointType.XY_2D


	def is_visible_point(self, point: Point) -> bool:

		if not isinstance(point, Point):
			raise TypeError("point must be a Point")

		if point.dimension != self.dimension:
			raise ValueError("point dimension must match face dimension")

		if point.point_type is not self.point_type:
			raise ValueError("point type must match face point type")

		point_geometry = ShapelyPoint(point.x, point.y)

		if not self._far_area.shapely_2d_xy_geometry.covers(point_geometry):
			return False

		return not any(
            polygon.covers(point_geometry) 
            for polygon in self._nonvisibility_polygons
        )

	def is_visible_points(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
		"""Return visibility for matching arrays of XY coordinates."""
		x_values = np.asarray(x, dtype=float)
		y_values = np.asarray(y, dtype=float)
		if x_values.shape != y_values.shape:
			raise ValueError("x and y must have the same shape")
		visible = (
			(x_values - self.origin.x) ** 2
			+ (y_values - self.origin.y) ** 2
			<= self.radius**2
		)
		for polygon in self._nonvisibility_polygons:
			visible &= ~contains_xy(polygon, x_values, y_values)
		return visible

