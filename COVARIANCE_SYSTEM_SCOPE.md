# System-wide covariance scope

The application now has one global covariance interpretation selected from the
right-side UI:

- `Prediction XY` — default
- `Full 3D`

The same selection is used by:

1. covariance stability classification,
2. rolling and fixed covariance solver requests,
3. covariance values shown in the active-target panel,
4. the covariance plot and active trace,
5. rolling and fixed forward prediction envelopes.

## Prediction XY

Only the `[Vx, Vy]` covariance block is active. All Z variance and XZ/YZ
cross-covariance terms are zeroed before downstream consumers receive the
matrix.

The covariance plot hides `Czz` and `trace(active)` is `Cxx + Cyy`.

## Full 3D

The complete symmetric `[Vx, Vy, Vz]` covariance is active. The solver and
stability calculator use all three dimensions. The covariance plot shows
`Czz` and the full trace.

The terrain canvas is still a 2D XY view. Therefore, in Full 3D mode, the
drawn ellipse is the mathematically correct XY marginal of the selected 3D
covariance and is labeled `3D→XY marginal`.

## Persistence

The system scope is stored in `.trajectory` project format version 6. Older
projects load as `Prediction XY`.
