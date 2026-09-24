# CnC Manager and scenario solver contract

The CnC layer is the scenario-level solver boundary. It is deliberately
separate from the existing per-target `PrincipalSigmaDemoSolver`.

## Runtime flow

```text
Dash current time
     |
     v
CnCManager
  - capture every bird at current scenario time
  - capture every cameraman
  - copy terrain + bounds
  - attach covariance scope
  - attach prediction_time_sec
  - attach solver_dt_sec
  - attach solution_time_sec
     |
     v
CnCSolver.solve(request)
     |
     +---- timed with perf_counter_ns()
     |
     v
CnCSolveExecution
  - typed observation probabilities
  - measured solve_elapsed_sec
  - exceeded_solution_time
```

Only the call to `solver.solve(request)` is included in
`solve_elapsed_sec`. Snapshot construction and result rendering are outside
that timing.

## Prediction time grid

The solver receives both:

```text
prediction_time_sec
solver_dt_sec
```

The request exposes `prediction_offsets_sec`.

For example:

```text
prediction_time_sec = 15
solver_dt_sec       = 5

prediction offsets:
0, 5, 10, 15
```

`t+0` is always included.

If the prediction horizon is not exactly divisible by `solver_dt_sec`, the
exact horizon is also included:

```text
prediction_time_sec = 12
solver_dt_sec       = 5

prediction offsets:
0, 5, 10, 12
```

This makes the final requested prediction horizon explicit.

## Bird data

Each `CnCBirdSnapshot` contains:

- target id and name;
- current sample index and sample time;
- current XYZ position;
- current XYZ velocity;
- current rolling 3x3 velocity covariance;
- fixed/model 3x3 velocity covariance, when available.

The full covariance matrices are supplied. `covariance_scope` is also supplied
on the request so a solver can decide how it wants to consume them.

## Cameraman data

Each `CnCCameramanSnapshot` contains:

- id and name;
- XYZ position, or `None` when not placed;
- view radius;
- start angle;
- end angle.

## Terrain data

`CnCTerrainSnapshot` contains a read-only copy of the height matrix plus X/Y
bounds and cell sizes.

## Observation-probability result

The primary solver result is now a typed list of:

```python
CnCObservationProbability(
    cameraman_id=...,
    cameraman_name=...,
    target_id=...,
    target_name=...,
    prediction_offset_sec=...,
    observation_probability=...,
)
```

The probability must be in `[0, 1]`.

A valid result contains exactly one record for every:

```text
cameraman × bird × prediction offset
```

The `CnCManager` validates the grid after `solve()` returns. Missing cells,
extra cells or duplicates are rejected instead of silently presenting a
partial matrix.

For 5 cameramen, 6 birds and four prediction offsets:

```text
5 × 6 × 4 = 120 probability records
```

## Dash CnC panel

The left **CnC Manager** panel owns:

```text
Prediction time (s)
Solver dt (s)
Future steps to display
Solution time budget (s)
Solve
```

The **Future steps to display** combobox affects presentation only.

For:

```text
prediction = 15 s
dt         = 5 s
```

the solver always computes:

```text
t0, t+5, t+10, t+15
```

while the display selector behaves as:

```text
1 -> t0
2 -> t0, t+5
3 -> t0, t+5, t+10
4 -> t0, t+5, t+10, t+15
```

Changing the display count does **not** rerun the solver. The latest typed
result is kept in a Dash store and the UI only changes which time columns are
shown.

The result is rendered as one table per cameraman:

```text
Cameraman 1
Bird                  t0      t+5     t+10
Outer mountain arc   72.1%    65.4%    54.0%
Summit crossing      18.0%    24.2%    31.8%
...
```

## Solver controls: budget vs measured runtime

`solution_time_sec` is the solver's requested time budget.

The manager measures the actual call:

```python
start = perf_counter_ns()
result = solver.solve(request)
end = perf_counter_ns()
```

and reports:

```text
solve_elapsed_sec
exceeded_solution_time
```

At this phase the manager does not forcibly terminate a solver that exceeds
the budget. A hard timeout requires a cancellable worker/process boundary.

## Current reference solver

`SnapshotInspectionSolver` is still only a contract/demo implementation.

It fills the complete cameraman × bird × time result grid using a simple,
deterministic distance-based placeholder probability. That logic is
intentionally not the real observation algorithm; it exists so the request,
result validation and UI can be exercised before the real solver is written.

A real solver only needs to implement:

```python
class MySolver:
    name = "my-solver"

    def solve(self, request: CnCSolverRequest) -> CnCSolverResult:
        ...
```

## Operator selection and acceptance

The probability matrix is also the interaction surface. There is no button
beside every candidate.

For each cameraman:

1. click a probability cell to select one bird/time candidate;
2. the selected bird row is emphasized;
3. a single action below that cameraman's matrix becomes available;
4. choose **Accept selection**;
5. accepting another candidate for the same cameraman uses **Change acceptance**;
6. selecting the already accepted cell changes the action to **Clear accepted**.

The percentage remains printed in every cell. Cell background intensity also
tracks the probability, so the table preserves both precise numeric values and
quick visual scanning.

An accepted observation is identified by:

```text
solution_id
cameraman
bird
prediction offset
observation probability
scenario time of the solve
```

Acceptance is therefore tied to one specific solver execution. Every new
**Solve** receives a fresh `solution_id` and clears the visible
selection/acceptance state, preventing stale recommendations from being
mistaken for current ones.

The top of the CnC result contains a compact **Accepted observations** summary,
one row per cameraman. Each cameraman block also keeps an accepted banner, and
the accepted percentage cell remains marked with a check and a persistent
border while all other percentages remain visible.

## Solver Metry tab

Solver execution/performance information is intentionally separated from the
CnC decision surface.

The CnC result panel now focuses on:

```text
accepted observations
probability matrices
selection / acceptance
```

The **Control → Solver Metry** tab contains execution information from the
latest solve:

```text
status
solver
scenario time
bird count
cameraman count
prediction-time count
prediction horizon
solver dt
solution-time budget
measured solve() duration
within-budget / exceeded-budget state
solution id
```

This keeps operational telemetry out of the probability/decision workflow.

## Bird context menu

Bird-name cells in each cameraman probability matrix support a native
right-click context menu.

Current actions are:

```text
Make active bird
Track bird
Select best observation
```

**Make active bird** selects that target in the Active Target controls.

**Track bird** selects the target and enables plot tracking.

**Select best observation** selects the highest probability for that
cameraman/bird among the prediction columns currently visible in the CnC
matrix. It deliberately does not accept the observation; acceptance remains an
explicit operator action using the single per-cameraman acceptance button.

A small vertical-ellipsis hint is shown beside each bird name to make the
context menu discoverable.

