"""Initial states supplied to path-imagination models."""

from __future__ import annotations

from dataclasses import dataclass

from crw_simulation.common_geometry import Point


@dataclass(frozen=True)
class PositionInitialData:
	position: Point


@dataclass(frozen=True)
class PositionDirectionInitialData:
	position: Point
	direction: float


@dataclass(frozen=True)
class PositionDirectionTurningRateInitialData:
	position: Point
	direction: float
	turning_rate: float = 0.0
