# Rolling vs Fixed Velocity Covariance

This version compares two velocity-covariance sources for each generated target.

## 1. Rolling / observed covariance

This is the existing covariance sampled from the recent velocity history:

```text
recent Vx,Vy,Vz samples
        ↓
rolling covariance window
        ↓
Σv,rolling(t)
```

It changes with time and therefore reflects what this particular generated
trajectory has recently been doing.

## 2. Fixed model covariance

After the normal smoothed trajectory is generated, the application derives the
read-only velocity model. That model is converted once into a fixed all-purpose
velocity covariance:

```text
derived velocity model
        ↓
fixed modeling rule
        ↓
Σv,fixed
```

The convention is:

```text
reference interval = 1.0 s
derived motion envelope = 3σ
horizontal covariance = isotropic
```

With target speed `v`, maximum derived turn rate `ω`, and the larger of the
derived climb/descent rates:

```text
σxy = v * sin(min(ω * 1s, 90°)) / 3
σz  = max(climb_rate, descent_rate) / 3

Σv,fixed = diag(σxy², σxy², σz²)
```

This is a deliberate modeling convention. A trajectory-derived motion envelope
does not mathematically imply a unique probability distribution.

The fixed covariance is runtime-only and is not serialized.

## Ellipses

Both covariance sources are propagated through the existing future-position
ellipse rule at the same prediction horizon:

```text
Σp = Δt² Σv
center = p + v Δt
```

Canvas convention:

```text
solid target-colored ellipse = rolling covariance
dashed white ellipse         = fixed model covariance
```

## Estimated positions

The configured solver is run twice at the same target sample:

```text
same position
same velocity
same time
same solver
       │
       ├─ rolling covariance → rolling estimate (diamond)
       │
       └─ fixed covariance   → fixed estimate (square)
```

The active-target panel shows the 3D distance between those two estimates. A
gold dotted segment joins their XY markers on the canvas.

For the bundled `PrincipalSigmaDemoSolver`, the two estimates are deterministic
1σ hypotheses. They are useful for visual comparison; covariance by itself
does not imply a biased mean estimate.
