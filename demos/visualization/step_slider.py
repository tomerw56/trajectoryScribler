"""Step-by-step slider visualization of trajectory density over time."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle

from .geometry_overlays import add_geometry_overlays, add_outside_visibility_overlay


ARTIFACTS_ROOT = Path(__file__).resolve().parent.parent / "artifacts"


def new_run_output_dir(output_root: Path | None = None) -> Path:
	"""Create and return a fresh ``<output_root>/<UTC timestamp>`` directory.

	Pass the result to multiple ``save_step_slider_*`` calls to collect all of
	a single run's artifacts under one timestamped directory.
	"""
	timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
	output_dir = (output_root or ARTIFACTS_ROOT) / timestamp
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir


def save_step_slider_heatmap(
	x: np.ndarray,
	y: np.ndarray,
	circle_radius: float,
	infeasible_areas,
	nonvisibility_polygons,
	model_name: str = "Trajectories",
	bins: int = 100,
	output_dir: Path | None = None,
) -> Path:
	"""Save a self-contained HTML scrubber stepping through every walk step's density.

	Density is binned per step (rather than plotting every walker) so the
	artifact stays a manageable size even for large walk counts. Returns the
	path to the saved HTML file, under
	``output_dir/step_slider.html`` (a fresh timestamped directory under
	``src/artifacts`` is created when ``output_dir`` is not given).
	"""
	if x.shape != y.shape:
		raise ValueError("x and y must have the same shape")
	if x.ndim != 2:
		raise ValueError("x and y must be two-dimensional arrays")

	n_steps = x.shape[1] - 1
	data_x_min, data_x_max = float(np.min(x)), float(np.max(x))
	data_y_min, data_y_max = float(np.min(y)), float(np.max(y))
	hist_range = [[data_x_min, data_x_max], [data_y_min, data_y_max]]

	histograms = [
		np.histogram2d(x[:, step], y[:, step], bins=bins, range=hist_range, density=True)[0].T
		for step in range(n_steps + 1)
	]

	fig, ax = plt.subplots(figsize=(8, 6.5))
	outer_radius = circle_radius * 1.2
	image = ax.imshow(
		histograms[0],
		origin="lower",
		extent=(data_x_min, data_x_max, data_y_min, data_y_max),
		cmap="hot",
		aspect="equal",
	)
	colorbar_axes = fig.add_axes([0.63, 0.28, 0.018, 0.44])
	fig.colorbar(image, cax=colorbar_axes, label="Probability density")
	ax.set_xlim(-outer_radius, outer_radius)
	ax.set_ylim(-outer_radius, outer_radius)
	add_outside_visibility_overlay(ax, circle_radius, outer_radius)
	ax.add_patch(Circle(
		(0, 0), circle_radius, fill=False, color="cyan", lw=2,
		label=f"Circle boundary (R={circle_radius})",
	))
	add_geometry_overlays(ax, infeasible_areas, nonvisibility_polygons)
	ax.scatter([0], [0], color="white", s=120, marker="*", edgecolors="black", label="Origin")
	ax.legend(loc="upper left", bbox_to_anchor=(1.32, 1), borderaxespad=0)
	fig.subplots_adjust(right=0.6)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	title = ax.set_title(f"{model_name} density - step 0/{n_steps}")

	def update(step: int):
		image.set_data(histograms[step])
		# Rescale color limits per step since density is concentrated differently at each step.
		image.set_clim(vmin=0, vmax=histograms[step].max() or 1.0)
		title.set_text(f"{model_name} density - step {step}/{n_steps}")
		return image, title

	animation = FuncAnimation(fig, update, frames=n_steps + 1, interval=200, blit=False)
	html = animation.to_jshtml()
	plt.close(fig)

	output_dir = output_dir or new_run_output_dir()
	output_path = output_dir / "step_slider.html"
	output_path.write_text(html, encoding="utf-8")
	return output_path


def save_step_slider_paths(
	x: np.ndarray,
	y: np.ndarray,
	circle_radius: float,
	infeasible_areas,
	nonvisibility_polygons,
	model_name: str = "Trajectories",
	model_params: dict[str, Any] | None = None,
	failure_statistics: dict[str, Any] | None = None,
	output_dir: Path | None = None,
	line_width: float | None = None,
	line_alpha: float | None = None,
) -> Path:
	"""Save a self-contained HTML scrubber showing every walk's own traced path.

	Unlike ``save_step_slider_heatmap`` (which bins all walks into a density
	image), this draws one polyline per walk. The slider reveals each walk's
	path from step 0 up to the selected step, for every walk simultaneously.

	Rendering cost scales with ``n_walks * n_steps`` line segments per frame,
	so this is far more expensive than the density-based slider for large
	batches (e.g. millions of walks can take a long time and a lot of memory
	to render). When ``line_width``/``line_alpha`` are left as ``None``, they
	are scaled from ``n_walks`` so a handful of walks stay clearly visible
	while very large batches still avoid a solid smear. Returns the path to
	the saved HTML file, under ``output_dir/step_slider_paths.html`` (a fresh
	timestamped directory under ``src/artifacts`` is created when
	``output_dir`` is not given).
	"""
	if x.shape != y.shape:
		raise ValueError("x and y must have the same shape")
	if x.ndim != 2:
		raise ValueError("x and y must be two-dimensional arrays")

	n_walks = x.shape[0]
	n_steps = x.shape[1] - 1
	if line_width is None:
		line_width = float(np.clip(150.0 / n_walks, 0.3, 2.0))
	if line_alpha is None:
		line_alpha = float(np.clip(15.0 / n_walks, 0.05, 1.0))
	data_x_min, data_x_max = float(np.min(x)), float(np.max(x))
	data_y_min, data_y_max = float(np.min(y)), float(np.max(y))

	def segments_up_to(step: int) -> np.ndarray:
		if step == 0:
			return np.empty((0, 2, 2))
		starts = np.stack([x[:, :step], y[:, :step]], axis=-1)
		ends = np.stack([x[:, 1:step + 1], y[:, 1:step + 1]], axis=-1)
		return np.stack([starts, ends], axis=2).reshape(-1, 2, 2)

	fig, ax = plt.subplots(figsize=(8, 6.5))
	outer_radius = circle_radius * 1.2
	line_collection = LineCollection(
		segments_up_to(0), colors="orange", linewidths=line_width, alpha=line_alpha,
	)
	ax.add_collection(line_collection)
	outer_radius = max(outer_radius, abs(data_x_min), abs(data_x_max), abs(data_y_min), abs(data_y_max))
	ax.set_xlim(-outer_radius, outer_radius)
	ax.set_ylim(-outer_radius, outer_radius)
	add_outside_visibility_overlay(ax, circle_radius, outer_radius)
	ax.add_patch(Circle(
		(0, 0), circle_radius, fill=False, color="cyan", lw=2,
		label=f"Circle boundary (R={circle_radius})",
	))
	add_geometry_overlays(ax, infeasible_areas, nonvisibility_polygons)
	ax.scatter([0], [0], color="white", s=120, marker="*", edgecolors="black", label="Origin")
	ax.legend(loc="upper left", bbox_to_anchor=(1.32, 1), borderaxespad=0)
	fig.subplots_adjust(right=0.6)
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.set_aspect("equal")
	parameters_text = json.dumps(model_params or {}, sort_keys=True, separators=(", ", ": "))
	statistics_text = json.dumps(failure_statistics or {}, sort_keys=True, separators=(", ", ": "))
	title = ax.set_title(f"{model_name} paths - step 0/{n_steps}")
	fig.text(0.02, 0.015, f"Model parameters: {parameters_text}", fontsize=8)
	fig.text(0.02, 0.035, f"Trajectory statistics: {statistics_text}", fontsize=8)

	def update(step: int):
		line_collection.set_segments(segments_up_to(step))
		title.set_text(f"{model_name} paths - step {step}/{n_steps}")
		return line_collection, title

	animation = FuncAnimation(fig, update, frames=n_steps + 1, interval=200, blit=False)
	html = animation.to_jshtml()
	plt.close(fig)

	output_dir = output_dir or new_run_output_dir()
	output_path = output_dir / "step_slider_paths.html"
	output_path.write_text(html, encoding="utf-8")
	return output_path


def save_step_slider_paths_per_walk(
	x: np.ndarray,
	y: np.ndarray,
	circle_radius: float,
	infeasible_areas,
	nonvisibility_polygons,
	model_name: str = "Trajectories",
	model_params: dict[str, Any] | None = None,
	failure_statistics: dict[str, Any] | None = None,
	output_dir: Path | None = None,
	line_width: float = 1.5,
) -> list[Path]:
	"""Save one HTML slider per walk, each showing only that walk's own path.

	Files are named ``walk_<index>.html`` (zero-padded) in ``output_dir`` (a
	fresh timestamped directory under ``src/artifacts`` is created when
	``output_dir`` is not given). Returns the list of saved paths, one per
	walk, in walk-index order.
	"""
	if x.shape != y.shape:
		raise ValueError("x and y must have the same shape")
	if x.ndim != 2:
		raise ValueError("x and y must be two-dimensional arrays")

	n_walks, n_columns = x.shape
	n_steps = n_columns - 1
	output_dir = output_dir or new_run_output_dir()
	index_width = len(str(n_walks - 1))

	saved_paths = []
	for walk_index in range(n_walks):
		walk_x = x[walk_index]
		walk_y = y[walk_index]

		fig, ax = plt.subplots(figsize=(6, 5))
		outer_radius = circle_radius * 1.2
		line, = ax.plot([], [], color="orange", lw=line_width)
		current_point, = ax.plot([], [], "o", color="red", markersize=6)
		outer_radius = max(
			outer_radius,
			abs(float(walk_x.min())) - 1,
			abs(float(walk_x.max())) + 1,
			abs(float(walk_y.min())) - 1,
			abs(float(walk_y.max())) + 1,
		)
		ax.set_xlim(-outer_radius, outer_radius)
		ax.set_ylim(-outer_radius, outer_radius)
		add_outside_visibility_overlay(ax, circle_radius, outer_radius)
		ax.add_patch(Circle(
			(0, 0), circle_radius, fill=False, color="cyan", lw=2,
			label=f"Circle boundary (R={circle_radius})",
		))
		add_geometry_overlays(ax, infeasible_areas, nonvisibility_polygons)
		ax.scatter([0], [0], color="white", s=120, marker="*", edgecolors="black", label="Origin")
		ax.legend(loc="upper left", bbox_to_anchor=(1.32, 1), borderaxespad=0)
		fig.subplots_adjust(right=0.6)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
		ax.set_aspect("equal")
		title = ax.set_title(f"{model_name} - walk {walk_index} - step 0/{n_steps}")
		parameters_text = json.dumps(model_params or {}, sort_keys=True, separators=(", ", ": "))
		statistics_text = json.dumps(failure_statistics or {}, sort_keys=True, separators=(", ", ": "))
		fig.text(0.02, 0.015, f"Model parameters: {parameters_text}", fontsize=8)
		fig.text(0.02, 0.035, f"Trajectory statistics: {statistics_text}", fontsize=8)

		def update(step: int, walk_x=walk_x, walk_y=walk_y, walk_index=walk_index):
			line.set_data(walk_x[: step + 1], walk_y[: step + 1])
			current_point.set_data(walk_x[step: step + 1], walk_y[step: step + 1])
			title.set_text(f"{model_name} - walk {walk_index} - step {step}/{n_steps}")
			return line, current_point, title

		animation = FuncAnimation(fig, update, frames=n_steps + 1, interval=200, blit=False)
		html = animation.to_jshtml()
		plt.close(fig)

		output_path = output_dir / f"walk_{walk_index:0{index_width}d}.html"
		output_path.write_text(html, encoding="utf-8")
		saved_paths.append(output_path)

	return saved_paths
