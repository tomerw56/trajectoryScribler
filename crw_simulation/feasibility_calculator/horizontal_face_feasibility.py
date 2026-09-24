from math import isclose

from shapely.geometry import Point as ShapelyPoint

from crw_simulation.common_geometry.horizontal_face import (
	HorizontalFace,
	HorizontalInverseCircleFace,
	HorizontalPolygonFace,
)
from crw_simulation.common_geometry.point import Point


def _validate_face_point(point: Point, face: HorizontalFace) -> None:
	if not isinstance(point, Point):
		raise TypeError("point must be a Point")
	if point.dimension != face.dimension:
		raise ValueError("point dimension must match face dimension")
	if point.point_type is not face.point_type:
		raise ValueError("point type must match face point type")


def is_point_on_horizontal_face(face: HorizontalFace, point: Point) -> bool:
	"""Return whether a point belongs to a horizontal face."""

	if not isinstance(face, HorizontalFace):
		raise TypeError("face must be a HorizontalFace")
	_validate_face_point(point, face)
	if face.dimension == 3:
		face_z = face.points[0].z if isinstance(face, HorizontalPolygonFace) else face.center.z
		if not isclose(point.z, face_z):
			return False
	point_geometry = ShapelyPoint(point.x, point.y)
	if isinstance(face, HorizontalPolygonFace):
		return face.shapely_2d_xy_geometry.covers(point_geometry)
	if isinstance(face, HorizontalInverseCircleFace):
		return not face.shapely_2d_xy_geometry.covers(point_geometry)
	raise TypeError(f"unsupported horizontal face type: {type(face).__name__}")
