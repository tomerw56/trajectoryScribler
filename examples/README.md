# Example trajectory projects

The `examples/` folder contains ready-to-load `.trajectory` projects.

## 01 — covariance showcase, flat terrain

Six targets and five cameramen. This is the easiest project for comparing
small, moderate, and very large XY velocity covariance because terrain Z
does not contribute.

| Target | Duration (s) | Max XY trace | Mean XY trace | End stability |
|---|---:|---:|---:|---|
| Straight — tiny covariance | 56.7 | 0.0000 | 0.0000 | STABLE |
| Near-straight — small covariance | 51.4 | 0.0011 | 0.0002 | STABLE |
| Gentle arc — moderate covariance | 44.9 | 1.4531 | 0.2724 | STABLE |
| Tight arc — sustained changing | 17.1 | 5.3255 | 4.1493 | CHANGING |
| S-curve — large covariance | 40.6 | 2.1171 | 0.5740 | UNSTABLE |
| Zigzag — very large covariance | 37.5 | 30.9860 | 15.4622 | UNSTABLE |

The end-state is only a snapshot; during playback a target may transition
between STABLE, CHANGING, and UNSTABLE as the recent 6-second covariance
history changes.

## 02 — basic multi-target patrol

A simpler four-target / three-cameraman project for ordinary UI testing.

## 03 — full 3D hills showcase

Four targets and four cameramen on `multi_hills.terrain`, with system
covariance scope preset to `FULL_3D`. Use this to see the difference
between XY-only interpretation and complete XYZ velocity covariance.

## 04 — canyon crossings

Five birds and five cameramen on `canyon.terrain`. Mixed straight, diagonal,
S-curve, hard-turn, and cruise-altitude routes. Preset to Full 3D.

## 05 — ridge-pass watch

Six birds and five cameramen on `ridge_pass.terrain`, including opposing
pass crossings, arcs, sharp turns, and a high cruise route. Preset to Full 3D.

## 06 — big-top orbits

Six birds and five cameramen around `big_top.terrain`: inner/outer arcs,
summit crossing, S-route, fast turns, and a high cruise orbit.

## 07 — multi-hills mixed flock

Seven birds and six cameramen on `multi_hills.terrain`; useful for ordinary
Prediction-XY playback with low, moderate, and aggressive trajectories.

## 08 — dense camera network

Eight birds and eight cameramen on flat terrain. Use this to stress the UI
with many simultaneous paths, sectors, estimates, and covariance envelopes.

## 09 — sparse camera challenge

Six birds but only three cameramen on canyon terrain. The geometry is
intentionally sparse and is useful for observer-layout experiments later.

## 10 — clearance vs cruise

Three route pairs over `big_top.terrain`. Each pair uses the same XY route
once in terrain-clearance mode and once in cruise-altitude mode. Full 3D.

## 11 — high-variance maneuvers

Seven birds and six cameramen on flat terrain. Includes a straight baseline
plus sawtooths, box turns, a tight arc, and aggressive S maneuvers.

## Regeneration

From the repository root:

```powershell
python .\scripts\generate_example_projects.py
```

The generator is deterministic.
