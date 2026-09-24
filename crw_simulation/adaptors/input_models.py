"""Pydantic models for validating calculator input at the external boundary."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PointInput(BaseModel):
	"""Serialized two-dimensional XY point."""

	model_config = ConfigDict(extra="forbid")

	x: float
	y: float


class CameramanInput(BaseModel):
	"""Serialized cameraman position and visibility radius."""

	model_config = ConfigDict(extra="forbid")

	id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_.-]+$")
	position: PointInput
	radius: float = Field(gt=0)


class PolygonFaceInput(BaseModel):
	"""Serialized polygonal infeasible face."""

	model_config = ConfigDict(extra="forbid")

	points: list[PointInput] = Field(min_length=3)


class CalculatorInput(BaseModel):
	"""Validated input used to construct calculator instances."""

	model_config = ConfigDict(extra="forbid")

	coordinate_system: Literal["XY_2D"]
	cameramen: list[CameramanInput] = Field(min_length=1)
	infeasible_horizontal_polygon_faces: list[PolygonFaceInput]


class PointsInput(BaseModel):
	"""Validated input containing points to evaluate."""

	model_config = ConfigDict(extra="forbid")

	coordinate_system: Literal["XY_2D"]
	points_to_check: list[PointInput]
