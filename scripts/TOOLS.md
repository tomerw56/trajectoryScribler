# Development tools

Install once:

```powershell
python -m pip install -r requirements-dev.txt
```

## Check everything

```powershell
.\scripts\check.ps1
```

This runs, in order:

1. `compileall` syntax check
2. Black in check/diff mode
3. Ruff lint (no modifications)
4. Mypy with explicit targets (`trajectory_app`, `tests`, `app.py`, `example_custom_solver.py`)
5. Pytest

Optional:

```powershell
.\scripts\check.ps1 -SkipTests
.\scripts\check.ps1 -Coverage
```

`-Coverage` uses `pytest-cov` and also writes `coverage.xml`.

## Apply safe fixes

```powershell
.\scripts\format.ps1
```

The fix order is intentional:

1. Ruff applies safe lint/import fixes (`--fix --show-fixes`).
2. Black formats the resulting code.
3. Ruff runs again without `--fix` and fails if non-auto-fixable lint remains.

Ruff classifies some transformations as unsafe. They are never applied by default. To opt in explicitly:

```powershell
.\scripts\format.ps1 -UnsafeFixes
```

## What does not auto-fix

- Mypy is a type checker and has no general automatic fix mode.
- Pytest reports behavioral failures; it does not rewrite code.
- Black formats code when run normally; `--check` is used only in the check script.

The shell scripts provide the same behavior:

```bash
./scripts/check.sh
./scripts/check.sh --coverage
./scripts/format.sh
./scripts/format.sh --unsafe-fixes
```


## One-command safe fix + check

PowerShell:

```powershell
.\scripts\check.ps1 -Fix
```

Bash/WSL:

```bash
./scripts/check.sh --fix
```

This runs Ruff safe fixes and Black first, then performs the full non-mutating
compile/lint/type/test checks. Ruff unsafe fixes remain opt-in through the
format script.
