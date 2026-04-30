# CI

QRWKV-XLA CI runs on `ubuntu-latest` for Python 3.11 and 3.12. Each job uses
Strategy A: install the project once with an editable install, then run scripts
and tests against the installed package.

## Validation Steps

CI runs these commands in order:

```bash
python -m compileall src scripts tests
python scripts/print_env.py
python scripts/smoke_cpu.py
python scripts/smoke_tpu.py
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/inspect_targets.py artifacts/teacher_targets/fake_export
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

The TPU smoke script is intentionally graceful when JAX or TPU devices are not
available. It verifies that the TPU config path loads without making local CPU
development depend on TPU hardware.

## Local Mirror

Create or activate a local virtual environment, then install the package in
editable mode with development tools:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

`scripts/validate_local.py` does not install dependencies. It only mirrors the
CI validation commands, prints each command, and fails fast on the first error.

## Editable Install Policy

Editable install is the blessed workflow because scripts, tests, and CI should
all import `qrwkv_xla` through packaging metadata instead of relying on
`PYTHONPATH=src` or script-local path mutation. This keeps CLI behavior closer
to how the package will run once installed normally, while still allowing source
edits to take effect immediately during development.

## Generated Artifacts

Validation writes fake teacher target bundles under `artifacts/`. That directory
is gitignored and should remain local/generated state rather than committed
repository content.
