# Testing Strategy

## Tier 0: import/layout tests
Runs anywhere.

## Tier 1: CPU config/smoke tests
Runs on the local VM with no GPU.

## Tier 2: JAX CPU train-step tests
Later phase.

## Tier 3: TPU smoke tests
Kaggle/Colab/free TPU when available. Should be tiny and fast.

## Tier 4: large TPU scale tests
Only after grant or paid TPU access.

## Phase 0 scope

Phase 0 only requires real Tier 0 tests:
- `tests/test_imports.py`
- `tests/test_project_layout.py`

Those tests should pass even if JAX or TPU libraries are unavailable.

## Phase 2.5 CI scope

P2.5 stabilizes the local and CI validation path without adding model or
trainer logic. The blessed workflow is Strategy A:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

`scripts/validate_local.py` mirrors CI and runs, in order:

- `python -m compileall src scripts tests`
- `python scripts/print_env.py`
- `python scripts/smoke_cpu.py`
- `python scripts/smoke_tpu.py`
- `python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml`
- `python scripts/inspect_targets.py artifacts/teacher_targets/fake_export`
- `python -m pytest`
- `python -m ruff check .`
- `python -m ruff format --check .`

The editable install is intentional. Tests and scripts should import the
installed package rather than depending on `PYTHONPATH=src` or local path
mutation. The fake export step writes only gitignored `artifacts/` content.
