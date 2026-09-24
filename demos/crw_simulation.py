"""Two-dimensional endpoint simulation and analysis.

Model
-----
The first of ``N`` steps has heading pi/2 and ends at (0, L).  Each later
heading is the previous heading plus an independent N(0, sigma**2) turning
angle.  The simulation is vectorized in memory-bounded batches, so the default
100,000 trajectories can be run without storing every complete path.

The code also evaluates exact finite-step formulas for the endpoint mean,
mean squared displacement (MSD), and covariance.  There is no simple closed
form for the complete finite-step endpoint density.  A moment-matched Gaussian
is therefore provided as a long-walk/diffusion approximation and compared with
the Monte Carlo density and radial distribution.

Example
-------
    python crw_simulation.py --steps 100 --step-length 1 --sigma 0.2 \
        --trajectories 100000 --output crw_endpoint_distribution.png

    python crw_simulation.py --model three-choice-turn-flight --steps 100 \
        --speed 2 --turning-radius 5 --time-step 0.1 --trajectories 100000

Dependencies: NumPy and Matplotlib.
"""

from __future__ import annotations


import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Force the package next to this launcher to be the first import location.
project_root_text = str(PROJECT_ROOT)
if not sys.path or Path(sys.path[0] or ".").resolve() != PROJECT_ROOT:
    sys.path.insert(0, project_root_text)


import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.path_imagination import simulate_three_choice_turn_flight_2d_xy
from tests import crw_test_utils
FloatArray = NDArray[np.float64]



def plot_three_choice_turn_flight_analysis(
    endpoints: FloatArray,
    parameters: ThreeChoiceTurnFlightParameters,
    bins: int = 180,
) -> tuple[Figure, tuple[Axes, Axes]]:
    """Plot empirical endpoint density and radius distributions for three-choice flight."""
    statistics = crw_test_utils.compute_endpoint_statistics(endpoints)
    x_centers, y_centers, density, plot_range = crw_test_utils.estimate_endpoint_density(
        endpoints, bins=bins,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, (heatmap_axis, radial_axis) = plt.subplots(
        1,
        2,
        figsize=(13.6, 6.0),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (1.1, 1.0)},
    )

    image = heatmap_axis.pcolormesh(
        x_centers,
        y_centers,
        density,
        shading="auto",
        cmap="magma",
        rasterized=True,
    )
    colorbar = figure.colorbar(image, ax=heatmap_axis, pad=0.02)
    colorbar.set_label("Endpoint probability density")
    heatmap_axis.scatter(
        *statistics.mean,
        marker="x",
        s=80,
        linewidths=2.0,
        color="white",
        label="Monte Carlo mean",
        zorder=4,
    )
    heatmap_axis.scatter(0.0, 0.0, s=22, color="lime", label="Origin", zorder=4)
    heatmap_axis.set(
        xlabel="Final x position",
        ylabel="Final y position",
        xlim=tuple(plot_range[0]),
        ylim=tuple(plot_range[1]),
        title="Monte Carlo endpoint density",
    )
    heatmap_axis.set_aspect("equal", adjustable="box")
    heatmap_legend = heatmap_axis.legend(
        loc="best",
        fontsize=8,
        framealpha=0.78,
        facecolor="black",
        edgecolor="white",
    )
    for label in heatmap_legend.get_texts():
        label.set_color("white")

    endpoint_radii = np.linalg.norm(endpoints, axis=1)
    max_radius = max(float(endpoint_radii.max() * 1.02), 1.0)
    radial_centers, radial_density = crw_test_utils.radial_distribution(
        endpoints,
        bins=max(50, bins // 2),
        max_radius=max_radius,
    )
    radial_axis.plot(
        radial_centers,
        radial_density,
        color="#8C1D40",
        linewidth=2.1,
        label="Monte Carlo",
    )
    radial_axis.fill_between(
        radial_centers,
        radial_density,
        color="#8C1D40",
        alpha=0.16,
    )
    radial_axis.set(
        xlabel="Endpoint radius r",
        ylabel="Radial probability density p(r)",
        title="Radial endpoint distribution",
        xlim=(0.0, max_radius),
    )
    radial_axis.set_ylim(bottom=0.0)
    radial_axis.legend(loc="best")

    figure.suptitle(
        "Three-Choice Turn Flight after exactly "
        f"N={parameters.n_steps} steps\n"
        f"speed={parameters.speed:g}, turning_radius={parameters.turning_radius:g}, "
        f"time_step={parameters.time_step:g}, "
        f"initial_direction={parameters.initial_direction:g} rad, "
        f"trajectories={parameters.n_trajectories:,}",
        fontsize=14,
        fontweight="bold",
    )
    return figure, (heatmap_axis, radial_axis)



def main() -> None:
    parser = crw_test_utils.build_argument_parser()
    args = parser.parse_args()
    if args.model == "crw":
        parameters = crw_test_utils.CRWParameters(
            n_steps=args.steps,
            step_length=args.step_length,
            angular_std=args.sigma,
            n_trajectories=args.trajectories,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        endpoints = crw_test_utils.simulate_crw_endpoints(parameters)
        statistics = crw_test_utils.compute_endpoint_statistics(endpoints)
        theory = crw_test_utils.analytical_endpoint_moments(
            parameters.n_steps,
            parameters.step_length,
            parameters.angular_std,
        )
        crw_test_utils.print_analysis_report(parameters, statistics, theory)
        figure, _ = crw_test_utils.plot_endpoint_analysis(
            endpoints,
            parameters,
            bins=args.bins,
            overlay_contours=not args.no_contours,
        )
    else:
        if args.time_step is None:
            parser.error("--time-step is required when --model is three-choice-turn-flight")
        parameters = crw_test_utils.ThreeChoiceTurnFlightParameters(
            n_steps=args.steps,
            speed=args.speed,
            turning_radius=args.turning_radius,
            time_step=args.time_step,
            initial_direction=args.initial_direction,
            n_trajectories=args.trajectories,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        endpoints = crw_test_utils.simulate_three_choice_turn_flight_endpoints(parameters)
        statistics = crw_test_utils.compute_endpoint_statistics(endpoints)
        crw_test_utils.print_three_choice_turn_flight_report(parameters, statistics)
        figure, _ = plot_three_choice_turn_flight_analysis(
            endpoints,
            parameters,
            bins=args.bins,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    print(f"\nSaved visualization to: {args.output.resolve()}")
    if not args.no_show:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()
