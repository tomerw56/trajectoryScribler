# Trajectory Scribbler

PySide6 desktop application for drawing, configuring, and replaying multiple
3D target trajectories over generated height-map terrain.

The application separates the main simulation/view from entity configuration:

- **Main window** — terrain, active target, playback, live state, velocity,
  covariance, solver estimate, and predicted-position ellipse.
- **Target Manager** — target creation/removal, drawing, generation, color,
  speed, altitude policy, covariance settings, and prediction horizon.
- **Cameraman Manager** — static cameramen, placement, color, and view sectors.
- **Terrain Designer** — terrain generation and `.terrain` files.

## Quick start

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Development tools:

```powershell
pip install -r requirements-dev.txt
.\scripts\check.ps1
```

Apply safe formatting/fixes:

```powershell
.\scripts\format.ps1
```

Black, Ruff, Mypy, Pytest, and Coverage configuration lives in `pyproject.toml`.

## Terrain

Use **New Terrain…** to open the Terrain Designer.

A terrain defines:

- XY bounds
- grid size
- base altitude
- mountains and valleys
- feature sigma range
- noise amplitude

Save terrain as `*.terrain`.

A `*.trajectory` project stores a reference to the terrain file instead of
embedding the height map. Loading fails clearly if that referenced terrain
cannot be found.

Ready-made examples are included in `terrains/`:

- `multi_hills.terrain`
- `big_top.terrain`
- `canyon.terrain`
- `ridge_pass.terrain`

## Targets

Open **Target Manager…** from the main toolbar.

Up to 10 targets are supported. Each target independently owns:

- name and color
- raw scribble
- constant 3D speed `|V|`
- motion sample `dt`
- covariance sample rate and window
- prediction horizon
- path resolution
- altitude policy and related parameters

### Drawing lifecycle

Drawing is explicitly armed from Target Manager:

1. Select a target.
2. Press **Begin drawing / redraw**.
3. Draw with the left mouse button on the terrain.
4. Press **Generate selected trajectory** to commit it.

Beginning a redraw deletes the previous path. Stopping drawing or leaving the
drawing workflow before successful generation discards the uncommitted draft.
Playback cannot accidentally continue drawing.

See `DRAWING_STATE.md` for the exact state rules.

### Altitude modes

Each target has its own altitude mode.

**Terrain clearance**

```text
z = terrain(x, y) + clearance
```

**Cruise altitude**

The target attempts to remain at an absolute cruise altitude while always
clearing terrain. A maximum climb/descent angle causes it to begin climbing
before an obstacle instead of jumping vertically.


## Derived velocity model

After a target trajectory is generated, Target Manager shows a read-only
velocity-model summary derived from that final smoothed trajectory:

- minimum observed turn radius
- maximum implied turn rate
- maximum implied lateral acceleration
- maximum climb/descent angle
- maximum climb/descent rate

This analysis does **not** modify or constrain the trajectory and is not
saved/exported. See `DERIVED_VELOCITY_MODEL.md`.


## Covariance stability

The rolling covariance now has a trailing-window stability calculator with four
states: `INSUFFICIENT_DATA`, `STABLE`, `CHANGING`, and `UNSTABLE`. The active
covariance plot shows the current state plus drift, RMS change, and maximum
single jump. The stability window is configurable per target.

See `COVARIANCE_STABILITY.md`.

## Rolling vs fixed covariance

Each generated target now has two covariance views:

- **rolling covariance** from recent sampled velocity history
- **fixed model covariance** derived once from the target's read-only
  derived velocity model

Both are shown as prediction ellipses and are independently passed through
the same configured solver so their estimated positions can be compared.

On the canvas, the rolling ellipse is solid and target-colored; the fixed
ellipse is dashed white. The rolling solver estimate is a diamond and the
fixed-covariance estimate is a square.

See `COVARIANCE_COMPARISON.md`.

## Forward prediction envelope

Prediction visualization now starts at the current target position and
shows the envelope developing forward in time to the configured
prediction horizon. The rolling/observed envelope and the fixed model
envelope can each be toggled on or off from the right-side panel.

See `FORWARD_PREDICTION_ENVELOPE.md`.

## Playback

All generated targets share one global simulation time.

The main view provides:

- First
- Previous
- Play / Pause
- Next
- Last

Targets may have different motion sample intervals; each displayed target uses
the sample nearest the current global time.

## Velocity and covariance

Target speed is the constant magnitude of the 3D velocity:

```text
|V| = sqrt(Vx² + Vy² + Vz²)
```

Climbing and descending redistribute that fixed magnitude between horizontal
and vertical components.

The velocity plot shows:

- `|V|`
- `Vx`
- `Vy`
- `Vz`

The rolling velocity-covariance plot shows:

- `Cxx`
- `Cyy`
- `Czz`
- `trace(C)`

Both plots follow the active target.

## Predicted-position ellipse

Each target has a prediction horizon, default `2.0 s`.

The simple prediction layer assumes exact current position and constant velocity
statistics over the configured horizon:

```text
predicted_center_xy = position_xy + velocity_xy * prediction_dt
position_covariance_xy = velocity_covariance_xy * prediction_dt²
```

The canvas renders the resulting 1σ XY covariance ellipse using the target
color with an approximately 75% transparent fill.

## Solver estimate and error statistics

The GUI calls a replaceable solver API for the estimated current position.

The terrain view distinguishes:

- filled circle — simulated ground truth
- hollow diamond — solver estimate
- dotted line — estimation offset

For each unique evaluated sample the application stores the 3D squared position
error and displays:

- current position error
- MSE in `m²`
- RMSE in `m`
- number of evaluated samples

See `SOLVER_API.md` for the solver contract.

## Cameramen

Cameramen are static entities configured in **Cameraman Manager…**.

Each cameraman owns:

- name and color
- terrain-surface position
- view radius
- start azimuth
- end azimuth

Placement is click-based; Z is sampled directly from the terrain.

Azimuth convention:

- `0°` = +X
- `90°` = +Y
- sectors may wrap across `360°`

The main canvas renders the view sector as a faint translucent cone.

See `CAMERAMEN.md` for details.

## Logging

The centralized application logger writes to both stdout and:

```text
./logs/trajectory_scribbler.log
```

The log rotates at roughly 2 MB with 5 retained backups.

Inside application modules use:

```python
from .logging_config import get_logger

logger = get_logger(__name__)
```

Use `logger.exception(...)` inside exception handlers when a traceback should be
preserved.

## Project files

- `*.terrain` — standalone height-map terrain
- `*.trajectory` — targets, cameramen, and a reference to a terrain file

Runtime-specific generated trajectory samples, solver outputs, and statistics
are regenerated rather than treated as authoritative serialized data.

## Example trajectory projects

The `examples/` folder now contains **11 ready-to-load scenarios** spanning
five terrains, 4–8 birds per project, and 3–8 cameramen:

- `01_covariance_showcase_flat.trajectory` — covariance scale / stability reference.
- `02_basic_multi_target_patrol.trajectory` — simple multi-bird UI demo.
- `03_full_3d_hills_showcase.trajectory` — Full-3D hill routes.
- `04_canyon_crossings.trajectory` — canyon crossings and cruise route.
- `05_ridge_pass_watch.trajectory` — opposing traffic through a ridge pass.
- `06_big_top_orbits.trajectory` — mountain arcs, summit crossings, and fast turns.
- `07_multi_hills_mixed_flock.trajectory` — seven-bird mixed flock.
- `08_dense_camera_network_flat.trajectory` — eight birds and eight cameramen.
- `09_sparse_camera_challenge_canyon.trajectory` — six birds with only three observers.
- `10_clearance_vs_cruise_big_top.trajectory` — paired altitude-mode comparisons.
- `11_high_variance_maneuvers_flat.trajectory` — aggressive covariance/stability stress test.

`examples/README.md` describes the scenarios and
`examples/scenario_manifest.json` gives machine-readable counts, terrain,
scope, bird names, and cameraman names.

## Stability policy

The default stability policy is intentionally tolerant of small covariance
motion:

- normalization floor: `0.005 (m/s)^2`
- covariance source window: `5 s` (about 21 velocity samples at 0.25 s sampling)
- stability window: `6 s` (about 13 covariance estimates at 0.5 s cadence)
- STABLE: drift <= `0.40`, RMS <= `0.15`, max jump <= `0.35`
- a single large jump is `CHANGING`, not automatically `UNSTABLE`
- UNSTABLE requires sustained high RMS movement, repeated large jumps
  (`>=25%` of recent covariance transitions), or severe drift plus sustained motion

The app waits for the rolling covariance source window plus the requested
stability window before making a stability judgment. With the defaults this is
`5 s + 6 s = 11 s`; earlier samples are `INSUFFICIENT_DATA`.

### Covariance reliability regression

For a scenario-level temporal reliability report:

```powershell
python .\scripts\covariance_reliability_report.py `
    .\examples\06_big_top_orbits.trajectory
```

The report shows the fraction of classified samples that are STABLE, CHANGING,
UNSTABLE, and operationally usable (`STABLE + CHANGING`). This is a temporal
stability regression, not a NEES/calibration test.


## Dash trajectory player — Phase 01

The repository now also contains a Dash playback/inspection UI. It is separate from the PySide6 editor and reuses the same generated trajectory, covariance, prediction, reliability, solver, and logging backend.

On Windows:

```powershell
python -m pip install -r .\requirements.txt
.\run_dash.ps1
```

Open `http://127.0.0.1:8050`.

See `DASH_PHASE_01.md` for the layout and phase boundary.

### Experimental Dubins freehand drawing

The desktop Target Manager now has **Begin Dubins drawing / redraw**. It asks
for a minimum turn radius before drawing begins, fits the freehand intent with
piecewise Dubins paths, and then uses the normal altitude/motion/covariance
pipeline.

See `DUBINS_DRAWING.md` for the interaction and current limitations.

### Scenario-level CnC solver

The Dash left panel now contains a **CnC Manager** with prediction time,
solution-time budget and a manual **Solve** button. The manager captures all
birds, cameramen and terrain, calls a scenario-level solver, and reports the
measured `solve()` duration.

See `CNC_SOLVER.md`.

### CnC observation-probability grid

The scenario solver now receives a prediction horizon plus solver `dt` and
returns one observation probability for every cameraman × bird × prediction
time. The Dash CnC panel renders one table per cameraman and lets you choose
how many future time columns to display without rerunning the solver.

See `CNC_SOLVER.md`.

### Dash Track selected bird

The Dash playback toolbar includes a **Track** toggle. When enabled, the plot
zooms into a local window and continuously recenters on the currently selected
bird as playback advances. Selecting another active bird moves the tracked
viewport to that bird. Disable Track to return to the full scenario view.

### CnC probability selection and acceptance

Probability cells in the Dash CnC matrix are clickable. Select a percentage
and use the single per-cameraman action to accept, change or clear that
observation. Accepted choices stay highlighted and are summarized above the
matrices while all percentages remain visible.

### Solver Metry and bird context actions

Solver runtime/status information now lives under **Control → Solver Metry**,
leaving the CnC panel focused on observation decisions. Bird names in the CnC
probability tables can be right-clicked to make a bird active, track it on the
plot, or select its best currently visible observation probability.

