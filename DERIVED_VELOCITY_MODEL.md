# Derived Velocity Model

The derived velocity model is a **read-only runtime analysis** of a target's
generated trajectory.

It is not a constraint, is not exported as a standalone model, and is not
serialized into `.trajectory` projects.

## Pipeline

```text
raw scribble
    ↓
normal path resampling / smoothing
    ↓
altitude policy
    ↓
final generated 3D trajectory
    ↓
derived velocity-model analysis
```

The analysis never changes the trajectory.

## Reported values

For the generated target trajectory the application reports:

- configured constant 3D speed
- minimum observed XY turn radius
- maximum implied turn rate
- maximum implied lateral acceleration
- maximum climb angle
- maximum descent angle
- maximum climb rate
- maximum descent rate

For XY curvature:

```text
curvature κ ≈ Δheading / Δs
turn radius R = 1 / |κ|
turn rate ω = speed * |κ|
lateral acceleration a = speed² * |κ|
```

The curvature analysis uses a coarser, additionally smoothed **analysis copy**
of the final XY trajectory. This prevents tiny point-to-point numerical
wiggles from dominating the reported extrema. The displayed/generated
trajectory itself is not modified by this analysis smoothing.

Vertical values are calculated from the actual final 3D trajectory. For a
segment climb/descent angle `γ`:

```text
vertical rate = speed * sin(γ)
```

## Lifetime

The value lives at:

```python
target.derived_velocity_model
```

It is recalculated whenever the generated target is resampled (for example
after changing target speed), and cleared when the trajectory geometry becomes
invalid or is redrawn.
