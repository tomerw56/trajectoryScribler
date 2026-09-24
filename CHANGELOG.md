# v6.6 — covariance stability

- Added `CovarianceStabilityCalculator` with four enum states: `INSUFFICIENT_DATA`, `STABLE`, `CHANGING`, `UNSTABLE`.
- Added trailing-window drift, RMS change, and maximum-jump metrics using normalized Frobenius matrix distance.
- Added `evaluate_history(..., last_n_seconds=N)` for downstream/history consumers.
- Added per-target covariance stability window in Target Manager.
- Added live stability caption on the covariance plot, including explicit `do not trust` text for `UNSTABLE`.
- Added tests for all four states, window isolation, and history retrieval.

# Changelog

## Cleanup baseline

- Removed duplicate Generate Selected / Generate All actions from the main toolbar.
- Target generation remains available from Target Manager.
- Removed duplicate Target Manager / Cameraman Manager buttons from the side panel.
- Removed unused `project_path` runtime state.
- Removed unused imports and the unused `IdentitySolver`.
- Removed non-runtime reference/stub files from the distributable project.
- Rewrote README to describe the current application instead of historical UI versions.

Earlier development history is represented by the repository/archive history rather
than embedded version notes in the main README.

## Dash Phase 01 — playback and inspection

- Added a separate Dash application entry point (`dash_app.py`, `run_dash.ps1`) while preserving the PySide6 desktop application.
- Added packaged `.trajectory` scenario selection and global playback controls.
- Added Plotly terrain, camera sectors, all target paths/current positions, rolling/fixed forward covariance envelopes, solver estimate markers, and reliability caption.
- Added left-side CNC placeholder panel for the next phase.
- Added bottom `Control` tab dock. The first tab mirrors the desktop Active Target Sample data and contains active-target, covariance-scope, and prediction-visibility controls.
- Added a placeholder Telemetry tab for the next phase.
- Reused the existing simulation, covariance, stability/reliability, solver, and logging modules instead of duplicating algorithms in the Dash layer.

## Phase 01.3 — dropdown theme fix

- Removed inline white text styling from Dash dropdown components.
- Moved dropdown theming entirely into CSS.
- Explicitly styles control background, selected value, placeholder, menu,
  focused option, selected option, and input text.
- Adds ID-level fallback selectors for the three current dropdowns.
- Adds regression tests preventing the white-on-white selected-value bug.

## Phase 01.4 — controls visibility

- Added React-Select/Dash 3 compatible dropdown selectors using generated-class
  suffixes (`*-control`, `*-singleValue`, `*-menu`, `*-option`).
- Retained legacy Dash 2.x selectors for compatibility.
- Selected dropdown value is explicitly white on a dark control background.
- Prediction checkboxes now use a visible blue accent, light outline, and
  high-contrast labels on the dark Control panel.

## Phase 01.5 — control text contrast

- Dropdown controls now use a light surface with dark selected/input text.
- Dropdown menus remain dark with white option text.
- Slider tooltip/value box now uses dark text on a light background.
- Avoids inherited white text on light control surfaces.

## Phase 01.6 — friendly light theme

- Replaced the accumulated dark-theme CSS with a clean light UI.
- Dropdowns are white with dark selected values, placeholders, inputs and menus.
- Prediction checkboxes use a standard blue accent with dark labels.
- Timeline, controls and tabs use light surfaces with dark text.
- Plotly uses a white canvas and dark labels.
- Terrain colorbar and X/Y grid remain disabled.
- Playback, covariance, prediction and reliability behavior is unchanged.

## Phase 01.7 — plot layout

- Removed the Plotly equal-X/Y aspect lock from the dashboard map.
- Terrain/trajectory drawing now stretches to fill the available plot area.
- Legend moved from the top horizontal strip to a vertical column on the right.
- Added dedicated right margin so the legend no longer overlaps title/reliability.
- Kept terrain colorbar/grid disabled and covariance/prediction rendering unchanged.

## Phase 01.8 — runtime forward prediction

- Added a numeric up/down `Forward prediction (s)` control.
- Range: 0.25–30.0 s, step: 0.25 s.
- Scenario load initializes it from the first target's configured prediction horizon.
- Changing it is runtime-only and does not modify the `.trajectory` file.
- The override drives rolling/fixed covariance envelopes, solver estimates,
  future-truth error evaluation, RMSE/MSE and active-sample display.

## v6.12 — experimental Dubins freehand drawing

- Added original freehand vs Dubins drawing modes.
- Dubins drawing prompts for minimum turn radius before the drawing is armed.
- Added pure-Python shortest Dubins solver for LSL/LSR/RSL/RSR/RLR/LRL.
- Freehand intent is converted to automatic anchor poses and piecewise Dubins legs.
- Existing altitude, motion, covariance and solver pipeline is reused unchanged.
- Active Dubins targets show anchor markers and heading arrows.
- Fit mean/max error is exposed in Target Manager.
- Project format version 7 persists generation mode and radius.
- Dash runtime can reload and regenerate version-7 Dubins projects.

## v6.13 — CnC Manager and scenario solver

- Added scenario-level `CnCSolver` protocol and immutable request snapshots.
- CnC request contains all current birds, rolling/fixed covariances,
  cameramen, terrain, prediction time and solution-time budget.
- Added `CnCManager`, which times only the `solve()` call with a monotonic
  high-resolution clock.
- Added measured solve duration and budget-overrun reporting.
- Moved prediction time into the left Dash CnC panel.
- Added solution-time numeric control and manual Solve button.
- Added non-optimizing `SnapshotInspectionSolver` to exercise the interface.

## v6.14 — cameraman/bird/time observation probabilities

- Added `solver_dt_sec` to the scenario solver request.
- Prediction offsets always include t0, regular dt jumps and the exact horizon.
- Added typed `CnCObservationProbability` results.
- CnCManager validates a complete cameraman × bird × prediction-time grid.
- Added solver-dt control to the Dash CnC panel.
- Added Future steps to display combobox.
- Display count changes only the latest result presentation; it does not rerun solve.
- CnC result UI now renders one probability matrix per cameraman.
- Reference solver fills the matrix with intentionally placeholder probability logic.

## v6.15 — selected-bird tracking

- Added a Track toggle to the Dash playback controls.
- Tracking zooms the plot to roughly one third of the terrain extent.
- The currently selected bird is kept exactly at the center of the viewport.
- Playback continuously follows the selected bird while tracking is enabled.
- Changing Active Target immediately tracks the newly selected bird.
- Tracking uses a changing Plotly `uirevision` so manual zoom state cannot
  prevent the programmed follow behavior.
- Turning tracking off restores the normal full-scenario view.

## v6.16 — CnC probability selection and acceptance

- Probability cells are now the selection surface; no action button is added
  beside each option.
- Added one Accept/Change/Clear action per cameraman.
- Added per-cameraman selected-row emphasis and selected-cell outline.
- Accepted cells remain marked while all probability percentages stay visible.
- Added probability-intensity cell shading for quick scanning.
- Added compact Accepted observations summary across all cameramen.
- Added per-solve `solution_id` and reset acceptance on each new solve.

## v6.17 — Solver Metry and bird context menu

- Moved solver/status/performance metadata out of the CnC result panel.
- Added Control → Solver Metry tab for timing, budget and scenario/solver metrics.
- CnC result panel now focuses on accepted observations and probability matrices.
- Added right-click context menu to bird-name cells in each cameraman matrix.
- Context actions: Make active bird, Track bird, Select best observation.
- Select best observation uses the highest probability among currently displayed
  prediction columns and does not auto-accept.
- Refactored Track button rendering so context-menu tracking updates the toolbar.

