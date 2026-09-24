"""Live invisible-only-fraction queries for repeatedly re-imagined paths."""

from __future__ import annotations

import math

from crw_simulation.feasibility_calculator import FeasibilityCalculator_XY_2D
from crw_simulation.path_imagination import PathImaginator
from crw_simulation.visibility_calculator import PovWorldCalc_XY_2D

from .first_failure import find_first_failures


class InvisibilityFractionSession:
    """Re-imagine each bird's future from its current state and score it.

    All configuration except each call's bird states is fixed for the
    lifetime of the session: the imaginator (model + parameters), the
    Monte Carlo sample count, the imagination horizon, the cameramen, and
    the infeasible polygons.
    """

    def __init__(
        self,
        *,
        imaginator: PathImaginator,
        n_walks: int,
        horizon: float,
        visibility_calculators: list[PovWorldCalc_XY_2D],
        feasibility_calculator: FeasibilityCalculator_XY_2D,
        time_step: float | None = None,
    ) -> None:
        if n_walks < 1:
            raise ValueError("n_walks must be at least 1")
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if not visibility_calculators:
            raise ValueError("visibility_calculators must not be empty")

        resolved_time_step = (
            time_step if time_step is not None else getattr(imaginator, "time_step", None)
        )
        if resolved_time_step is None:
            raise ValueError(
                "time_step must be provided explicitly for imaginators that "
                "have no time_step attribute of their own"
            )
        if resolved_time_step <= 0:
            raise ValueError("time_step must be positive")

        self._imaginator = imaginator
        self._n_walks = n_walks
        self._n_steps = math.ceil(horizon / resolved_time_step)
        self._visibility_calculators = list(visibility_calculators)
        self._feasibility_calculator = feasibility_calculator

    def invisible_fraction(
        self, bird_states: dict[str, object]
    ) -> dict[str, dict[str, float]]:
        """Re-imagine each bird's future and score its invisible-only fraction.

        Returns ``{bird_id: {cameraman_id: fraction}}``, where ``fraction``
        is the share of this call's Monte Carlo batch whose first failure
        reason was ``"invisible"`` strictly (excluding ``"both"`` and
        ``"infeasible"``).
        """
        results: dict[str, dict[str, float]] = {}
        for bird_id, initial_data in bird_states.items():
            try:
                x, y = self._imaginator.imagine(
                    initial_data, self._n_walks, self._n_steps
                )
            except TypeError as error:
                raise ValueError(
                    f"bird {bird_id!r} state does not match the session's "
                    f"configured imaginator: {error}"
                ) from error

            per_cameraman: dict[str, float] = {}
            for visibility_calculator in self._visibility_calculators:
                first_failure_results = find_first_failures(
                    x, y, visibility_calculator, self._feasibility_calculator
                )
                per_cameraman[visibility_calculator.cameraman_id] = (
                    first_failure_results.invisible_walks.size / self._n_walks
                )
            results[bird_id] = per_cameraman
        return results
