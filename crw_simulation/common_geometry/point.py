from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Real
from types import MappingProxyType


class PointType(Enum):
	XY_2D = "2dxy"
	XYZ_3D = "3dxyz"
	POLAR_2D = "2dpolar"


@dataclass(frozen=True)
class Point:
	coordinates: Mapping[str, Real]
	point_type: PointType

	@property
	def dimension(self) -> int:
		return len(self.coordinates)

	@property
	def horizontal_coordinate_names(self) -> frozenset[str]:
		return frozenset(self.coordinates)

	def __post_init__(self) -> None:
		self._validate()
		object.__setattr__(self, "coordinates", MappingProxyType(dict(self.coordinates)))

	def _validate(self) -> None:
		if not isinstance(self.coordinates, Mapping):
			raise TypeError("coordinates must be a mapping")
		if not self.coordinates:
			raise ValueError("coordinates must not be empty")
		for name, value in self.coordinates.items():
			if not isinstance(name, str):
				raise TypeError("coordinate names must be strings")
			if not name:
				raise ValueError("coordinate names must not be empty")
			if not isinstance(value, Real):
				raise TypeError("coordinate values must be real numbers")
		if not isinstance(self.point_type, PointType):
			raise TypeError("point_type must be a PointType")

	def __getattr__(self, name: str) -> Real:
		coordinates = self.__dict__.get("coordinates")
		if coordinates is not None and name in coordinates:
			return coordinates[name]
		raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")
