# Covariance stability

This feature answers a runtime question that is intentionally separate from
statistical calibration:

> Has the rolling velocity covariance changed so much over the last N seconds
> that a downstream decision should hesitate to trust it?

## Four states

`CovarianceStabilityState` contains exactly four states:

- `INSUFFICIENT_DATA` — not enough trailing history to judge.
- `STABLE` — recent changes are small.
- `CHANGING` — covariance is moving; use with caution.
- `UNSTABLE` — covariance changed drastically; do not trust it for a decision.

## Metrics

For covariance matrices A and B, the calculator uses normalized Frobenius
change:

```text
D(A,B) = ||A-B||F / max(||A||F, ||B||F, epsilon)
```

Over the requested trailing window it reports:

- `drift`: change from the window baseline to now.
- `rms_change`: RMS of consecutive covariance changes.
- `max_single_jump`: largest single consecutive change.

The thresholds are operational policy thresholds, not confidence levels. They
are kept in `CovarianceStabilityConfig` so they can later be tuned from
recorded scenarios.

## API

```python
calculator.evaluate(samples, end_time_sec=t, window_sec=5.0)
```

To ask for the stability history over the last N seconds:

```python
calculator.evaluate_history(
    samples,
    last_n_seconds=20.0,
    end_time_sec=t,
)
```

The application exposes the same behavior through
`MainWindow.get_covariance_stability(...)` and
`MainWindow.get_covariance_stability_history(...)`.

## Plot

The covariance plot carries a live caption for the active target. It includes
the four-state classification and the three metrics. `UNSTABLE` explicitly
says `do not trust` so a human and a later software consumer share the same
semantics.

## Plot title

The current stability state is displayed in a fixed title strip above the
rolling-covariance plot. The title does not move with plot pan/zoom. It shows:

- active target and current time
- four-state stability classification
- trailing window length
- drift
- RMS covariance change
- maximum single covariance jump

## Stability scope

Stability is consumer-specific.

The default scope is:

```python
CovarianceStabilityScope.PREDICTION_XY
```

This compares only:

```text
[[Cxx, Cxy],
 [Cyx, Cyy]]
```

because that is the covariance consumed by the current forward XY prediction
ellipse. Changes confined to Z no longer make the prediction covariance appear
unstable.

Full 3D behavior remains available explicitly:

```python
calculator.evaluate(
    covariance_samples,
    scope=CovarianceStabilityScope.FULL_3D,
)
```

## Near-zero deadband

Before relative normalization, covariance differences with Frobenius norm less
than `absolute_change_deadband` are treated as zero. The default is `1e-6`.
This prevents numerical noise around a zero covariance from producing a large
relative stability score.

## Default policy tuning

The defaults favor a larger covariance sample pool and persistent-change
detection:

```text
velocity sample cadence      = 0.25 s
covariance source window     = 5 s   (~21 velocity samples)
covariance output cadence    = 0.5 s
stability window             = 6 s   (~13 covariance estimates)

normalization_floor          = 0.005 (m/s)^2
absolute_change_deadband     = 1e-6

STABLE:
  drift       <= 0.40
  RMS change  <= 0.15
  max jump    <= 0.35

large-change threshold       = 0.80
UNSTABLE large-change share  = 25%
sustained-high RMS threshold = 0.35
severe drift threshold       = 1.50
```

A single large covariance transition does **not** automatically make the
window `UNSTABLE`; it becomes `CHANGING` unless the large changes repeat,
the RMS movement is high, or large drift is accompanied by sustained motion.

The result exposes `large_change_fraction` so downstream code can see how
persistent the instability was.

## Warm-up

The rolling covariance estimate has its own source window. Stability is not
classified until that source window has filled and a complete stability window
has subsequently been observed.

With the application defaults:

```text
covariance source window = 5 s
stability window         = 6 s
first classified point   ≈ 11 s
```

Earlier points return `INSUFFICIENT_DATA`, not `UNSTABLE`.

## Operational reliability regression

Temporal stability and statistical calibration are different questions.

The repository now includes a scenario-level **operational reliability**
regression. It defines:

```text
usable = STABLE or CHANGING
unusable = UNSTABLE
```

This matches the runtime API where `CHANGING` is explicitly usable with
caution. It does **not** claim that the covariance is statistically calibrated;
NEES/coverage would be a separate calibration test.

The regression uses covariance **roughness** in addition to drift/RMS/jumps.
Roughness is the RMS normalized second difference of the covariance sequence.
It prevents a smooth rotating covariance ellipse from being treated the same
way as erratic covariance thrashing.

Current default:

```text
unstable_min_roughness = 0.35
```

Run the report on the Big Top scenario:

```powershell
python .\scripts\covariance_reliability_report.py `
    .\examples\06_big_top_orbits.trajectory
```

The automated regression protects the intentionally smooth Big Top routes from
becoming spuriously unreliable while also checking that deliberately erratic
routes in scenario 11 remain substantially less usable.

