# Forward Prediction Envelope

This version draws prediction development from the **current target position**
toward the configured prediction horizon rather than showing only the final
ellipse at `t + dt`.

For a horizon `dt`, the application samples intermediate times such as:

```text
0.2 dt, 0.4 dt, 0.6 dt, 0.8 dt, 1.0 dt
```

For each sample time `τ`:

```text
future center = p0 + v0 τ
Σp(τ) = τ² Σv
```

The resulting ellipses are drawn as an envelope that starts at the current
location and grows into the future.

Display toggles:
- rolling / actual covariance envelope
- fixed velocity-model covariance envelope

The demo solver is also aligned with this interpretation:

```text
estimated future position = p0 + v0 dt + covariance-based offset
```
