"""Matplotlib overlays for polygonal obstacles and visibility shadows."""

from __future__ import annotations

from collections.abc import Iterable
from math import sqrt

from matplotlib.patches import Polygon
from matplotlib.patches import Wedge
from shapely.geometry import GeometryCollection, MultiPolygon
from shapely.geometry.base import BaseGeometry


def _polygon_geometries(geometry: BaseGeometry) -> Iterable[BaseGeometry]:
	if geometry.is_empty:
		return
	if isinstance(geometry, (MultiPolygon, GeometryCollection)):
		for part in geometry.geoms:
			yield from _polygon_geometries(part)
		return
	if geometry.geom_type == "Polygon":
		yield geometry


def add_geometry_overlays(
	ax,
	infeasible_areas: Iterable[BaseGeometry],
	nonvisibility_polygons: Iterable[BaseGeometry],
) -> None:
	"""Draw forbidden faces and visibility shadows with explicit legend labels."""
	for shadow_index, geometry in enumerate(nonvisibility_polygons):
		for polygon in _polygon_geometries(geometry):
			ax.add_patch(Polygon(
				list(polygon.exterior.coords),
				closed=True,
				facecolor="red",
				edgecolor="cyan",
				linewidth=1.5,
				alpha=0.35,
				label="Visibility shadow" if shadow_index == 0 else "_nolegend_",
			))

	for area_index, geometry in enumerate(infeasible_areas):
		for polygon in _polygon_geometries(geometry):
			ax.add_patch(Polygon(
				list(polygon.exterior.coords),
				closed=True,
				facecolor="black",
				edgecolor="cyan",
				linewidth=2,
				alpha=0.8,
				label="Forbidden polygon" if area_index == 0 else "_nolegend_",
			))


def add_outside_visibility_overlay(ax, radius: float, outer_radius: float) -> None:
	"""Shade the region outside the visibility circle in red."""
	if outer_radius <= radius:
		return
	# Extend beyond the square frame's corners so no out-of-radius corner remains unshaded.
	outer_radius *= sqrt(2.0)
	ax.add_patch(Wedge(
		(0.0, 0.0),
		outer_radius,
		0.0,
		360.0,
		width=outer_radius - radius,
		facecolor="red",
		edgecolor="none",
		alpha=0.2,
		label="Outside visibility radius",
	))

