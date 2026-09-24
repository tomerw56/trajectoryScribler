from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose
from numbers import Real

from shapely.geometry import Point as ShapelyPoint
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry.base import BaseGeometry

from .point import Point, PointType


class HorizontalFace(ABC):
	@property
	@abstractmethod
	def dimension(self) -> int:
		raise NotImplementedError

	@property
	@abstractmethod
	def point_type(self) -> PointType:
		raise NotImplementedError

	@property
	@abstractmethod
	def shapely_2d_xy_geometry(self) -> BaseGeometry:
		raise NotImplementedError


@dataclass(frozen=True)
class HorizontalPolygonFace(HorizontalFace):
	points: tuple[Point, ...]

	@property
	def dimension(self) -> int:
		return self.points[0].dimension

	@property
	def point_type(self) -> PointType:
		return self.points[0].point_type

	@property
	def shapely_2d_xy_geometry(self) -> ShapelyPolygon:
		return ShapelyPolygon((point.x, point.y) for point in self.points)

	def __post_init__(self) -> None:
		self._validate()
		object.__setattr__(self, "points", tuple(self.points))

	def _validate(self) -> None:
		if not isinstance(self.points, (tuple, list)):
			raise TypeError("points must be a sequence")
		if len(self.points) < 3:
			raise ValueError("a polygon must have at least three points")
		if not all(isinstance(point, Point) for point in self.points):
			raise TypeError("polygon points must be Point instances")
		first_point = self.points[0]
		if first_point.point_type not in (PointType.XY_2D, PointType.XYZ_3D):
			raise ValueError("horizontal faces require XY_2D or XYZ_3D points")
		expected_names = frozenset({"x", "y"} if first_point.dimension == 2 else {"x", "y", "z"})
		if first_point.horizontal_coordinate_names != expected_names:
			raise ValueError("horizontal face points require x and y coordinates")
		if not all(
			point.dimension == first_point.dimension
			and point.point_type is first_point.point_type
			and point.horizontal_coordinate_names == expected_names
			and (first_point.dimension == 2 or isclose(point.z, first_point.z))
			for point in self.points[1:]
		):
			raise ValueError("all horizontal face points must have the same dimension, point type, coordinates, and z value")
		if not self.shapely_2d_xy_geometry.is_valid:
			raise ValueError("horizontal polygon face must be a valid polygon")


HorizontalPolygon = HorizontalPolygonFace


@dataclass(frozen=True)
class HorizontalInverseCircleFace(HorizontalFace):
	center: Point
	radius: Real

	@property
	def dimension(self) -> int:
		return self.center.dimension

	@property
	def point_type(self) -> PointType:
		return self.center.point_type

	@property
	def shapely_2d_xy_geometry(self) -> ShapelyPolygon:
		return ShapelyPoint(self.center.x, self.center.y).buffer(self.radius)

	def __post_init__(self) -> None:
		if not isinstance(self.center, Point):
			raise TypeError("center must be a Point")
		if self.center.point_type not in (PointType.XY_2D, PointType.XYZ_3D):
			raise ValueError("horizontal faces require an XY_2D or XYZ_3D center")
		expected_names = frozenset({"x", "y"} if self.center.dimension == 2 else {"x", "y", "z"})
		if self.center.horizontal_coordinate_names != expected_names:
			raise ValueError("horizontal face centers require x and y coordinates")
		if not isinstance(self.radius, Real):
			raise TypeError("radius must be a real number")
		if self.radius <= 0:
			raise ValueError("radius must be positive")


