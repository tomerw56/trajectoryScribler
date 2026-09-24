"""Heatmap visualization for surviving trajectory endpoints."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .geometry_overlays import add_geometry_overlays, add_outside_visibility_overlay


def plot_survival_heatmap(
	x: np.ndarray,
	y: np.ndarray,
	survived: np.ndarray,
	circle_radius: float,
	infeasible_areas,
	nonvisibility_polygons,
	bins: int = 150,
	model_name: str = "Trajectories",
	output_dir: Path | str = Path("."),
) -> Path:
	"""Plot the endpoint density of trajectories that survived both constraints."""
	fig, ax = plt.subplots(figsize=(10, 8))
	outer_radius = circle_radius * 1.2
	histogram = ax.hist2d(
		x[survived, -1],
		y[survived, -1],
		bins=bins,
		density=True,
		cmap="hot",
	)

	colorbar_axes = fig.add_axes([0.63, 0.28, 0.018, 0.44])
	fig.colorbar(histogram[3], cax=colorbar_axes, label="Probability density")
	ax.set_xlim(-outer_radius, outer_radius)
	ax.set_ylim(-outer_radius, outer_radius)
	add_outside_visibility_overlay(ax, circle_radius, outer_radius)
	ax.add_patch(Circle(
		(0, 0),
		circle_radius,
		fill=False,
		color="cyan",
		lw=2,
		label=f"Circle boundary (R={circle_radius})",
	))

	add_geometry_overlays(ax, infeasible_areas, nonvisibility_polygons)

	ax.scatter([0], [0], color="white", s=120, marker="*", edgecolors="black", label="Origin")
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_title(f"Endpoint Density of Surviving {model_name}")
	ax.set_aspect("equal")
	ax.legend(loc="upper left", bbox_to_anchor=(1.32, 1), borderaxespad=0)
	ax.grid(alpha=0.3)
	fig.subplots_adjust(left=0.08, right=0.6, bottom=0.1, top=0.94)
	output_path = Path(output_dir) / "survival_heatmap.png"
	fig.savefig(output_path)
	plt.show()
	return output_path
