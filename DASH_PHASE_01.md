# Dash Phase 01 — trajectory playback and inspection

This phase adds a web/Dash playback UI without changing the established simulation math.

## Layout

```text
+-----------------------------------------------------------------------+
| trajectory project | first prev play next last | global time          |
| global time slider                                                   |
+---------------+-------------------------------------------------------+
|               |                                                       |
| CNC           | Plotly terrain / cameras / trajectories               |
| placeholder   | current target positions                              |
|               | rolling + fixed forward covariance envelopes          |
|               | covariance reliability caption                        |
+---------------+-------------------------------------------------------+
| Control                                                               |
| [ Active Target ] [ Telemetry (next) ]                                |
| active target | covariance mode | prediction visibility | sample data |
+-----------------------------------------------------------------------+
```

## Reused backend

The Dash loader uses the same core pipeline as the desktop app:

```text
.trajectory + .terrain
        |
        v
build_trajectory
        |
sample_motion
        |
sample_velocity_covariance
        |
derive_velocity_model
        |
fixed_velocity_covariance_from_model
```

No covariance, prediction, or reliability formula is reimplemented in the browser layer.

## Phase-01 capabilities

- choose a packaged `.trajectory` scenario;
- play / pause;
- first / previous / next / last;
- seek with global time slider;
- Plotly terrain heatmap;
- all target trajectories and current positions;
- cameraman locations and sectors;
- select active target;
- system covariance mode (`Prediction XY` / `Full 3D`);
- rolling prediction envelope;
- fixed-model prediction envelope;
- current covariance reliability state and metrics;
- active-target sample data in the first Control tab;
- CNC panel placeholder;
- Telemetry tab placeholder for the next phase;
- stdout + rotating-file logging using the existing logging configuration.

## Run on Windows

```powershell
python -m pip install -r .\requirements.txt
.\run_dash.ps1
```

Then open `http://127.0.0.1:8050`.

Logs continue under `./logs/trajectory_scribbler.log`.

## Debug on Windows

For a normal run:

```powershell
.\run_dash.ps1
```

For Dash debug mode:

```powershell
.\run_dash_debug.ps1
```

or:

```powershell
python .\dash_app.py --debug
```

A VS Code launch profile is included:

```text
Trajectory Dash: Debug
```

Dash's auto-reloader is disabled in debug mode so breakpoints remain attached
to the single process started by VS Code.

## UI sizing and contrast

The main workspace intentionally gives the Plotly terrain view most of the
screen: `72vh` with a `650px` minimum height on desktop. The Control dock
remains below the map and the CNC column is kept narrow.

All Dash dropdowns use an explicit dark-theme style for selected values,
menus, placeholders, and input text. The time slider/ruler also has explicit
light mark labels and a high-contrast track/handle.

## Plot visibility fixes

The terrain heatmap no longer shows a color scale bar and the Plotly X/Y grid
is disabled. Terrain height remains available in hover text.

Rolling and fixed covariance prediction envelopes keep their original 1-sigma
geometry, but are now drawn with stronger final outlines, translucent final
fills, and clearer prediction centerlines. This improves visibility without
changing the covariance mathematics.

Dropdown selected-value text has explicit ID-level white-text overrides to
avoid React-Select theme/specificity differences across Dash versions.

## Runtime forward-prediction horizon

The Active Target control panel includes a numeric up/down:

```text
Forward prediction (s)
```

It ranges from `0.25` to `30.0` seconds in `0.25 s` increments. This is a
runtime override only; it does not change the saved project.

The selected horizon is used consistently by the rolling/fixed covariance
prediction envelopes, solver future estimates, future-truth comparison,
MSE/RMSE evaluation and the Active Target sample display.

