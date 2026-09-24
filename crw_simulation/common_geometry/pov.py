from dataclasses import dataclass

from .point import Point


@dataclass(frozen=True)
class Pov:
	point: Point

	def __post_init__(self) -> None:
		if not isinstance(self.point, Point):
			raise TypeError("point must be a Point")
