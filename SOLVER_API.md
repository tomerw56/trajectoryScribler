# Solver API — estimated current position

The simulator owns the **ground-truth** trajectory and motion samples.  A separate object named `solver` consumes the current state and the velocity covariance and returns the position it believes the target occupies.

## Contract

Implement:

```python
class MySolver:
    name = "My solver"

    def solve(self, request: SolverRequest) -> SolverResult:
        ...
        return SolverResult(estimated_position=np.array([x, y, z]))
```

The request provides:

- target id/name
- sample index and time
- current ground-truth position (for this simulation/API demo)
- current 3D velocity
- latest causal rolling velocity covariance (`3 x 3`)
- terrain altitude at the current XY
- target `MotionConfig`
- target `TrajectoryConfig`
- previous solver estimate, when available

The GUI calls only `solver.solve(request)` and draws the returned position.

To inject your implementation:

```python
from trajectory_app.main_window import run_app
from my_solver import MySolver

run_app(solver=MySolver())
```

## Visualization

- **Filled colored circle** — actual simulated position.
- **Hollow colored diamond** — solver-estimated position.
- **Dotted white line** — current estimation offset.

The right-hand current-sample panel also shows the estimated XYZ and scalar offset from ground truth.

## Bundled demo solver

`PrincipalSigmaDemoSolver` exists only so the API is visibly testable immediately.  Covariance by itself does not shift a mean position, so the demo does **not** claim to calculate the statistical mean.  It draws one deterministic 1-sigma hypothesis along the dominant eigenvector of the velocity covariance.

Because the covariance has units `(m/s)^2`, the demo converts velocity uncertainty to a positional offset over a short horizon:

```text
offset = dominant_axis * sqrt(dominant_variance) * horizon
```

By default the horizon is the larger of motion `dt` and covariance sample rate.  Replace this demo solver with the real covariance consumer when ready.
