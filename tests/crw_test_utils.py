
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from crw_simulation.common_geometry import Point, PointType
from crw_simulation.path_imagination import simulate_three_choice_turn_flight_2d_xy

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class CRWParameters:
    """Parameters defining an ensemble of two-dimensional CRWs."""

    n_steps: int = 100
    step_length: float = 1.0
    angular_std: float = 0.2
    n_trajectories: int = 100_000
    seed: int | None = 7
    batch_size: int = 20_000

    def validate(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.step_length <= 0:
            raise ValueError("step_length must be positive")
        if self.angular_std < 0:
            raise ValueError("angular_std cannot be negative")
        if self.n_trajectories < 1:
            raise ValueError("n_trajectories must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class ThreeChoiceTurnFlightParameters:
    """Parameters defining an ensemble of three-choice turn-flight paths."""

    n_steps: int = 100
    speed: float = 1.0
    turning_radius: float = 1.0
    time_step: float | None = None
    initial_direction: float = np.pi / 2
    n_trajectories: int = 100_000
    seed: int | None = 7
    batch_size: int = 20_000

    def validate(self) -> None:
        if self.n_steps < 1:
            raise ValueError("n_steps must be at least 1")
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if self.turning_radius <= 0:
            raise ValueError("turning_radius must be positive")
        if self.time_step is None:
            raise ValueError("time_step is required")
        if self.time_step <= 0:
            raise ValueError("time_step must be positive")
        if self.n_trajectories < 1:
            raise ValueError("n_trajectories must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


@dataclass(frozen=True)
class EndpointStatistics:
    """Summary statistics for a collection of endpoints."""

    mean: FloatArray
    covariance: FloatArray
    msd: float
    mean_radius: float
    std_radius: float
    radial_quantiles: FloatArray


@dataclass(frozen=True)
class AnalyticalMoments:
    """Exact finite-step endpoint moments for the CRW model."""

    mean: FloatArray
    covariance: FloatArray
    msd: float
    directional_correlation: float


def simulate_crw_endpoints(parameters: CRWParameters) -> FloatArray:
    """Simulate final positions using vectorized, memory-bounded batches.

    Returns
    -------
    ndarray, shape (n_trajectories, 2)
        The ``(x, y)`` position after exactly ``n_steps`` steps.
    """

    parameters.validate()
    assert parameters.time_step is not None
    rng = np.random.default_rng(parameters.seed)
    endpoints = np.empty((parameters.n_trajectories, 2), dtype=np.float64)

    for start in range(0, parameters.n_trajectories, parameters.batch_size):
        stop = min(start + parameters.batch_size, parameters.n_trajectories)
        batch_count = stop - start

        if parameters.n_steps == 1:
            endpoints[start:stop, 0] = 0.0
            endpoints[start:stop, 1] = parameters.step_length
            continue

        # Reuse the turn array as the subsequent heading array to limit memory.
        headings = rng.normal(
            loc=0.0,
            scale=parameters.angular_std,
            size=(batch_count, parameters.n_steps - 1),
        )
        np.cumsum(headings, axis=1, out=headings)
        headings += np.pi / 2.0

        # The deterministic first step contributes (0, L).
        endpoints[start:stop, 0] = parameters.step_length * np.cos(headings).sum(axis=1)
        endpoints[start:stop, 1] = parameters.step_length * (
            1.0 + np.sin(headings).sum(axis=1)
        )

    return endpoints


def simulate_three_choice_turn_flight_endpoints(
    parameters: ThreeChoiceTurnFlightParameters,
) -> FloatArray:
    """Simulate final positions with memory-bounded three-choice path batches."""
    parameters.validate()
    rng = np.random.default_rng(parameters.seed)
    endpoints = np.empty((parameters.n_trajectories, 2), dtype=np.float64)
    origin = Point({"x": 0.0, "y": 0.0}, PointType.XY_2D)

    for start in range(0, parameters.n_trajectories, parameters.batch_size):
        stop = min(start + parameters.batch_size, parameters.n_trajectories)
        batch_seed = int(rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        x, y = simulate_three_choice_turn_flight_2d_xy(
            n_walks=stop - start,
            n_steps=parameters.n_steps,
            origin=origin,
            speed=parameters.speed,
            turning_radius=parameters.turning_radius,
            time_step=parameters.time_step,
            initial_direction=parameters.initial_direction,
            seed=batch_seed,
        )
        endpoints[start:stop, 0] = x[:, -1]
        endpoints[start:stop, 1] = y[:, -1]

    return endpoints


def simulate_crw_trajectories(
    n_trajectories: int,
    n_steps: int,
    step_length: float,
    angular_std: float,
    seed: int | None = None,
) -> FloatArray:
    """Return complete paths for a modest number of trajectories.

    The result has shape ``(n_trajectories, n_steps + 1, 2)`` and includes the
    origin.  For large ensembles, prefer :func:`simulate_crw_endpoints`.
    """

    parameters = CRWParameters(
        n_steps=n_steps,
        step_length=step_length,
        angular_std=angular_std,
        n_trajectories=n_trajectories,
    )
    parameters.validate()

    rng = np.random.default_rng(seed)
    headings = np.empty((n_trajectories, n_steps), dtype=np.float64)
    headings[:, 0] = np.pi / 2.0
    if n_steps > 1:
        turns = rng.normal(0.0, angular_std, size=(n_trajectories, n_steps - 1))
        headings[:, 1:] = np.pi / 2.0 + np.cumsum(turns, axis=1)

    increments = step_length * np.stack((np.cos(headings), np.sin(headings)), axis=-1)
    paths = np.empty((n_trajectories, n_steps + 1, 2), dtype=np.float64)
    paths[:, 0, :] = 0.0
    paths[:, 1:, :] = np.cumsum(increments, axis=1)
    return paths


def compute_endpoint_statistics(endpoints: FloatArray) -> EndpointStatistics:
    """Compute empirical endpoint moments and radial summary statistics."""

    endpoints = np.asarray(endpoints, dtype=np.float64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2 or endpoints.shape[0] < 1:
        raise ValueError("endpoints must have shape (n, 2), with n >= 1")

    radii = np.linalg.norm(endpoints, axis=1)
    covariance = (
        np.cov(endpoints, rowvar=False, ddof=1)
        if endpoints.shape[0] > 1
        else np.zeros((2, 2), dtype=np.float64)
    )
    return EndpointStatistics(
        mean=endpoints.mean(axis=0),
        covariance=np.asarray(covariance, dtype=np.float64),
        msd=float(np.mean(np.einsum("ij,ij->i", endpoints, endpoints))),
        mean_radius=float(radii.mean()),
        std_radius=float(radii.std(ddof=1)) if radii.size > 1 else 0.0,
        radial_quantiles=np.quantile(radii, [0.05, 0.50, 0.95]),
    )


def _geometric_sum(ratio: float, count: int) -> float:
    """Return sum(ratio**k, k=0..count-1), including stable near-one cases."""

    if count <= 0:
        return 0.0
    if ratio == 0.0:
        return 1.0
    if ratio == 1.0:
        return float(count)
    log_ratio = np.log(ratio)
    return float(np.expm1(count * log_ratio) / np.expm1(log_ratio))


def analytical_endpoint_moments(
    n_steps: int,
    step_length: float,
    angular_std: float,
) -> AnalyticalMoments:
    """Compute exact mean, MSD, and covariance after ``n_steps``.

    If ``rho = exp(-sigma**2/2)``, headings ``k`` steps apart satisfy
    ``E[cos(theta[t+k] - theta[t])] = rho**k``.  This exponential decay is the
    directional correlation created by the Gaussian turning-angle process.
    """

    parameters = CRWParameters(
        n_steps=n_steps,
        step_length=step_length,
        angular_std=angular_std,
        n_trajectories=1,
    )
    parameters.validate()

    rho = float(np.exp(-0.5 * angular_std**2))
    mean_y = step_length * _geometric_sum(rho, n_steps)
    mean = np.array([0.0, mean_y], dtype=np.float64)

    if rho == 1.0:
        return AnalyticalMoments(
            mean=mean,
            covariance=np.zeros((2, 2), dtype=np.float64),
            msd=float((n_steps * step_length) ** 2),
            directional_correlation=rho,
        )

    # D = sum_j sum_k E[cos(theta_j - theta_k)] = E[|R/L|^2].
    lags = np.arange(1, n_steps, dtype=np.float64)
    d_sum = float(n_steps + 2.0 * np.dot(n_steps - lags, rho**lags))

    # J = sum_j sum_k E[cos(W_j + W_k)], where theta_j = pi/2 + W_j.
    # This retains the finite-time anisotropy induced by the fixed first heading.
    indices = np.arange(n_steps, dtype=np.float64)
    diagonal = float(np.sum(rho ** (4.0 * indices)))
    off_diagonal = 0.0
    for j in range(n_steps - 1):
        off_diagonal += rho ** (4 * j + 1) * _geometric_sum(rho, n_steps - j - 1)
    j_sum = diagonal + 2.0 * off_diagonal

    raw_x2 = 0.5 * step_length**2 * (d_sum - j_sum)
    raw_y2 = 0.5 * step_length**2 * (d_sum + j_sum)
    variance_x = max(raw_x2, 0.0)
    variance_y = max(raw_y2 - mean_y**2, 0.0)
    covariance = np.array([[variance_x, 0.0], [0.0, variance_y]], dtype=np.float64)

    return AnalyticalMoments(
        mean=mean,
        covariance=covariance,
        msd=float(step_length**2 * d_sum),
        directional_correlation=rho,
    )


def estimate_endpoint_density(
    endpoints: FloatArray,
    bins: int = 180,
    plot_range: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Estimate the normalized two-dimensional endpoint density."""

    if bins < 10:
        raise ValueError("bins must be at least 10")
    endpoints = np.asarray(endpoints, dtype=np.float64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 2:
        raise ValueError("endpoints must have shape (n, 2)")

    if plot_range is None:
        minima = endpoints.min(axis=0)
        maxima = endpoints.max(axis=0)
        spans = np.maximum(maxima - minima, 1e-9)
        padding = 0.04 * spans
        plot_range = (
            (float(minima[0] - padding[0]), float(maxima[0] + padding[0])),
            (float(minima[1] - padding[1]), float(maxima[1] + padding[1])),
        )

    density, x_edges, y_edges = np.histogram2d(
        endpoints[:, 0],
        endpoints[:, 1],
        bins=bins,
        range=plot_range,
        density=True,
    )
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, density.T, np.array(plot_range, dtype=np.float64)


def gaussian_approximation_density(
    x: FloatArray,
    y: FloatArray,
    moments: AnalyticalMoments,
) -> FloatArray:
    """Evaluate the moment-matched bivariate Gaussian density.

    This is a diffusion/long-walk approximation, not an exact finite-step
    spatial density.  A degenerate ballistic distribution has no ordinary 2D
    density and raises ``ValueError``.
    """

    covariance = moments.covariance
    determinant = float(np.linalg.det(covariance))
    scale = max(float(np.trace(covariance)), 1.0)
    if determinant <= np.finfo(float).eps * scale**2:
        raise ValueError("Gaussian approximation is degenerate in the ballistic limit")

    inverse = np.linalg.inv(covariance)
    dx = np.asarray(x, dtype=np.float64) - moments.mean[0]
    dy = np.asarray(y, dtype=np.float64) - moments.mean[1]
    exponent = -0.5 * (
        inverse[0, 0] * dx**2
        + 2.0 * inverse[0, 1] * dx * dy
        + inverse[1, 1] * dy**2
    )
    return np.exp(exponent) / (2.0 * np.pi * np.sqrt(determinant))


def radial_distribution(
    endpoints: FloatArray,
    bins: int = 90,
    max_radius: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Estimate the radial probability density of endpoint distance from origin."""

    radii = np.linalg.norm(np.asarray(endpoints, dtype=np.float64), axis=1)
    if max_radius is None:
        max_radius = float(radii.max() * 1.02)
    density, edges = np.histogram(radii, bins=bins, range=(0.0, max_radius), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, density


def gaussian_radial_density(
    radii: FloatArray,
    moments: AnalyticalMoments,
    angular_points: int = 720,
) -> FloatArray:
    """Numerically integrate the Gaussian approximation over polar angle."""

    radii = np.asarray(radii, dtype=np.float64)
    phi = np.linspace(0.0, 2.0 * np.pi, angular_points, endpoint=False)
    x = radii[:, None] * np.cos(phi)[None, :]
    y = radii[:, None] * np.sin(phi)[None, :]
    spatial_density = gaussian_approximation_density(x, y, moments)
    return radii * (2.0 * np.pi) * spatial_density.mean(axis=1)


def compare_simulation_and_theory(
    statistics: EndpointStatistics,
    theory: AnalyticalMoments,
) -> dict[str, Any]:
    """Return compact absolute and relative Monte Carlo moment errors."""

    mean_error = float(np.linalg.norm(statistics.mean - theory.mean))
    msd_absolute_error = abs(statistics.msd - theory.msd)
    msd_relative_error = msd_absolute_error / theory.msd if theory.msd > 0.0 else 0.0
    return {
        "mean_vector_error": mean_error,
        "msd_absolute_error": msd_absolute_error,
        "msd_relative_error": msd_relative_error,
        "covariance_frobenius_error": float(
            np.linalg.norm(statistics.covariance - theory.covariance)
        ),
    }


def print_analysis_report(
    parameters: CRWParameters,
    statistics: EndpointStatistics,
    theory: AnalyticalMoments,
) -> None:
    """Print a readable numerical comparison of simulation and theory."""

    errors = compare_simulation_and_theory(statistics, theory)
    persistence_steps = (
        np.inf
        if theory.directional_correlation == 1.0
        else 1.0 / (1.0 - theory.directional_correlation)
    )
    print("\n2D CORRELATED RANDOM WALK ANALYSIS")
    print("-" * 43)
    print(
        f"N={parameters.n_steps}, L={parameters.step_length:g}, "
        f"sigma={parameters.angular_std:g} rad, M={parameters.n_trajectories:,}"
    )
    print(f"One-step directional correlation rho : {theory.directional_correlation:.8f}")
    print(f"Persistence scale 1/(1-rho)          : {persistence_steps:.3f} steps")
    print(f"Monte Carlo mean [x, y]              : {statistics.mean}")
    print(f"Exact mean [x, y]                    : {theory.mean}")
    print(f"Monte Carlo MSD E[x^2+y^2]           : {statistics.msd:.8f}")
    print(f"Exact MSD                            : {theory.msd:.8f}")
    print(f"Relative MSD error                   : {errors['msd_relative_error']:.3%}")
    print(f"Mean endpoint radius                 : {statistics.mean_radius:.8f}")
    print(f"Radius quantiles [5%, 50%, 95%]      : {statistics.radial_quantiles}")
    print("Monte Carlo covariance:")
    print(statistics.covariance)
    print("Exact covariance:")
    print(theory.covariance)


def print_three_choice_turn_flight_report(
    parameters: ThreeChoiceTurnFlightParameters,
    statistics: EndpointStatistics,
) -> None:
    """Print empirical endpoint statistics for the three-choice flight model."""
    print("\nTHREE-CHOICE TURN FLIGHT ANALYSIS")
    print("-" * 43)
    print(
        f"N={parameters.n_steps}, speed={parameters.speed:g}, "
        f"turning_radius={parameters.turning_radius:g}, "
        f"time_step={parameters.time_step:g}, "
        f"initial_direction={parameters.initial_direction:g} rad, "
        f"M={parameters.n_trajectories:,}"
    )
    print(f"Monte Carlo mean [x, y]              : {statistics.mean}")
    print(f"Monte Carlo MSD E[x^2+y^2]           : {statistics.msd:.8f}")
    print(f"Mean endpoint radius                 : {statistics.mean_radius:.8f}")
    print(f"Radius quantiles [5%, 50%, 95%]      : {statistics.radial_quantiles}")
    print("Monte Carlo covariance:")
    print(statistics.covariance)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate and visualize two-dimensional endpoint distributions."
    )
    parser.add_argument(
        "--model",
        choices=("crw", "three-choice-turn-flight"),
        default="crw",
        help="motion model (default: crw)",
    )
    parser.add_argument("--steps", type=int, default=100, help="number of steps (default: 100)")
    parser.add_argument(
        "--step-length", type=float, default=1.0, help="constant step length (default: 1)"
    )
    parser.add_argument(
        "--sigma", type=float, default=0.2, help="turning-angle standard deviation in radians"
    )
    parser.add_argument(
        "--speed", type=float, default=1.0, help="three-choice flight speed (default: 1)"
    )
    parser.add_argument(
        "--turning-radius", type=float, default=1.0,
        help="three-choice flight turning radius (default: 1)",
    )
    parser.add_argument(
        "--time-step", type=float, default=None,
        help="three-choice flight step duration (required for three-choice model)",
    )
    parser.add_argument(
        "--initial-direction", type=float, default=np.pi / 2,
        help="three-choice initial direction in radians (default: pi/2)",
    )
    parser.add_argument(
        "--trajectories", type=int, default=100_000, help="number of Monte Carlo trajectories"
    )
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--batch-size", type=int, default=20_000, help="trajectories simulated per batch"
    )
    parser.add_argument("--bins", type=int, default=180, help="heatmap bins per axis")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("crw_endpoint_distribution.png"),
        help="output figure path",
    )
    parser.add_argument("--no-contours", action="store_true", help="hide Gaussian contours")
    parser.add_argument("--no-show", action="store_true", help="save without opening a plot window")
    return parser
