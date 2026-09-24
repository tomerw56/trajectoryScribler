"""Minimal example showing how to replace the bundled solver."""

import numpy as np

from trajectory_app.main_window import run_app
from trajectory_app.solver import SolverRequest, SolverResult


class MySolver:
    name = "Example custom solver"

    def solve(self, request: SolverRequest) -> SolverResult:
        # Replace this with the real covariance consumer.
        # This baseline simply says the estimated mean is the supplied
        # current position.
        estimate = np.asarray(request.current_position, dtype=float).copy()
        return SolverResult(
            estimated_position=estimate,
            metadata={"source": "example_custom_solver"},
        )


if __name__ == "__main__":
    run_app(solver=MySolver())
