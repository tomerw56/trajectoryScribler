# Runtime import diagnostic

Use the root-level launcher on Windows:

```powershell
.\run_app.ps1
```

It forces the folder containing `app.py` to the front of `PYTHONPATH`, then
prints the exact `core.py` and `main_window.py` files Python loaded.

For a diagnostic without opening the UI:

```powershell
.\scripts\diagnose_runtime.ps1
```

Expected output includes:

```text
fixed_velocity_covariance_from_model: True
_prediction_visibility_changed: True
```

If either loaded path points outside the newly extracted project folder, an
older installed/cached copy is being imported instead of this source tree.
