"""Convert external input mappings into CRW domain objects."""

from __future__ import annotations

from typing import Any, List

from crw_simulation.common_geometry import HorizontalPolygonFace, Point, PointType
from crw_simulation.feasibility_calculator import FeasibilityCalculator_XY_2D
from crw_simulation.visibility_calculator import PovWorldCalc_XY_2D

from .input_models import CalculatorInput, PointsInput


def point_type_from_input(value: str) -> PointType:
	"""Convert a coordinate-system name into a :class:`PointType`."""
	try:
		return PointType[value]
	except KeyError as error:
		raise ValueError(f"unsupported coordinate system: {value}") from error


def point_from_input(
	data: dict[str, Any],
	point_type: PointType,
) -> Point:
	"""Build an immutable point from an input coordinate mapping."""
	return Point(
		coordinates={
			name: value
			for name, value in data.items()
			if name != "point_type"
		},
		point_type=point_type,
	)


def horizontal_polygon_face_from_input(
	data: dict[str, Any],
	point_type: PointType,
) -> HorizontalPolygonFace:
	"""Build a horizontal polygon face from serialized point data."""
	points = tuple(
		point_from_input(point_data, point_type)
		for point_data in data["points"]
	)
	return HorizontalPolygonFace(points)


def build_calculators_2D_XY(
	data: dict[str, Any] | CalculatorInput,
) -> tuple[List[PovWorldCalc_XY_2D], FeasibilityCalculator_XY_2D]:
	"""Build visibility and feasibility calculators from XY input data."""
	model = CalculatorInput.model_validate(data)
	coordinate_system_type = point_type_from_input(model.coordinate_system)
	if coordinate_system_type is not PointType.XY_2D:
		raise ValueError("this adapter currently supports only XY_2D")

	cameramen = model.cameramen
	cameraman_ids = [cameraman.id for cameraman in cameramen]
	if len(cameraman_ids) != len(set(cameraman_ids)):
		raise ValueError("cameraman ids must be unique")

	infeasible_areas = tuple(
		horizontal_polygon_face_from_input(
			area_data.model_dump(),
			coordinate_system_type,
		)
		for area_data in model.infeasible_horizontal_polygon_faces
	)
	visibility_calculators = [
        PovWorldCalc_XY_2D(
            origin=point_from_input(
				cameraman.position.model_dump(),
				coordinate_system_type,
			),
            radius=cameraman.radius,
            infeasible_areas=infeasible_areas,
			cameraman_id=cameraman.id,
        )
        for cameraman in cameramen
    ]
	feasibility_calculator = FeasibilityCalculator_XY_2D(
		infeasible_areas=infeasible_areas,
	)
	return visibility_calculators, feasibility_calculator


def build_points_2D_XY(data: dict[str, Any] | PointsInput) -> list[Point]:
	"""Build XY points to check from serialized input data."""
	model = PointsInput.model_validate(data)
	coordinate_system_type = point_type_from_input(model.coordinate_system)
	if coordinate_system_type is not PointType.XY_2D:
		raise ValueError("this adapter currently supports only XY_2D")
	return [
		point_from_input(point_data.model_dump(), coordinate_system_type)
		for point_data in model.points_to_check
	]
