# Dubins freehand drawing — experimental phase

This phase adds a second XY trajectory generator while keeping the established
altitude, motion sampling, covariance, solver and Dash pipelines unchanged.

## User flow

Open **Target Manager** and choose:

```text
Begin Dubins drawing / redraw
```

Before the canvas is armed, the application asks for:

```text
Dubins minimum turn radius (m)
```

If the dialog is cancelled, the existing target is untouched.

After accepting the radius:

1. the existing target path is cleared;
2. draw freely on the terrain as before;
3. the dashed line is the raw freehand intent;
4. click **Generate selected trajectory**;
5. the application fits a piecewise Dubins XY trajectory;
6. the normal altitude and simulation pipeline runs.

The original **Begin freehand drawing / redraw** mode is still available.

## How the fit works

The mouse samples are treated as intent, not hard waypoints:

```text
raw scribble
    |
distance resampling + light smoothing
    |
automatic anchor poses
(x, y, heading)
    |
shortest Dubins path between consecutive poses
    |
sampled feasible XY path
    |
existing altitude policy
    |
XYZ trajectory -> motion -> covariance -> solver
```

The automatic anchor spacing is currently approximately:

```text
2.5 * minimum_turn_radius
```

with a lower bound based on the configured path resolution.

Heading at each anchor is estimated from a local tangent of the smoothed
scribble. Consecutive Dubins legs therefore share both position and heading at
their common anchor.

## Visual diagnostics

For the active generated Dubins target, the desktop canvas shows:

- dashed target-colored line: original freehand scribble;
- solid target-colored line: generated Dubins trajectory;
- white-centered anchor markers;
- short target-colored heading arrows;
- anchor indexes.

Target Manager also reports the mean and maximum distance from the smoothed
reference scribble to the generated Dubins path.

## Terrain boundaries

A Dubins turn is not clipped to the terrain rectangle because clipping would
destroy the requested curvature bound. If the fitted path leaves the terrain,
generation fails with a message asking you to reduce the minimum radius or
redraw farther from the boundary.

## Persistence

Trajectory project format version 7 stores:

```json
{
  "generation_mode": "dubins",
  "dubins_min_turn_radius_m": 10.0
}
```

Older projects load as `freehand`. Runtime anchor poses and fit diagnostics are
recalculated and are not stored.

## Current limitations

This is intentionally a first experiment:

- classical planar forward-only Dubins geometry;
- XY only; existing altitude policy creates Z afterward;
- anchor locations/headings are automatic;
- no global optimization of intermediate headings yet;
- a large requested radius can intentionally depart substantially from a tight
  hand-drawn corner.

These limitations make the raw-vs-generated visualization important while we
decide how aggressive the fitter should be.
