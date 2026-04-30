# Testing Strategy

## Tier 0: import/layout tests
Runs anywhere.

## Tier 1: CPU config/smoke tests
Runs on the local VM with no GPU.

## Tier 2: JAX CPU train-step tests
Runs local JAX student forward, loss, gradient, and smoke-train coverage on CPU.

## Tier 3: TPU smoke tests
Kaggle/Colab/free TPU when available. Should be tiny and fast.

## Tier 4: large TPU scale tests
Only after grant or paid TPU access.

## Phase 0 scope

Phase 0 only requires real Tier 0 tests:
- `tests/test_imports.py`
- `tests/test_project_layout.py`

Those tests should pass even if JAX or TPU libraries are unavailable.

## Phase 4 validation scope

Phase 4 keeps the Strategy A editable-install workflow and extends validation
with JAX student runtime coverage. The `rwkv7_reference` model is covered as an
XLA-friendly recurrent reference implementation, not a final optimized kernel.
Coverage includes CPU forward tests, JIT execution tests, attention-mask
behavior, deterministic initialization/application checks, and gradient coverage
through the smoke training path.

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
- `python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2`
- `python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2`
- `python -m pytest`
- `python -m ruff check .`
- `python -m ruff format --check .`

The editable install is intentional. Tests and scripts should import the
installed package rather than depending on `PYTHONPATH=src` or local path
mutation. The fake export and smoke training steps write only gitignored
`artifacts/` content.
