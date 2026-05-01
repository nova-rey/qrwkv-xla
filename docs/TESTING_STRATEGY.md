# Testing Strategy

## Tier 0: import/layout tests
Runs anywhere.

## Tier 1: CPU config/smoke tests
Runs on the local VM with no GPU.

## Tier 2: JAX CPU train-step tests
Runs local JAX student forward, loss, gradient, and smoke-train coverage on CPU.

## Tier 3: TPU smoke tests
Kaggle/Colab/free TPU when available. Should be tiny and fast.

Tier 3 also includes TPU-ready smoke scripts that CI runs without requiring
TPU. Real TPU environments should use `--require-tpu`.

HF backend unit tests use mocks/stubs by default. Optional HF integration tests
are outside the default tiers. They require `.[teacher-hf]`, may require cached
or downloadable model assets, and only run when
`QRWKV_RUN_HF_INTEGRATION=1` is set.

## Tier 4: large TPU scale tests
Only after grant or paid TPU access.

## Phase 0 scope

Phase 0 only requires real Tier 0 tests:
- `tests/test_imports.py`
- `tests/test_project_layout.py`

Those tests should pass even if JAX or TPU libraries are unavailable.

## Phase 5 validation scope

Phase 5 keeps the Strategy A editable-install workflow and extends validation
with the configured distillation stage runtime. The `rwkv7_reference` model is
still covered as an XLA-friendly recurrent reference implementation, not a final
optimized kernel. Distillation coverage includes YAML config loading, weighted
loss composition, hidden-state stage training, optional logits KL validation,
metrics summaries, and the CLI stage smoke.

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

`scripts/validate_local.py` mirrors CI and runs, in order:

- `python -m compileall src scripts tests`
- `python scripts/print_env.py`
- `python scripts/xla_inspect.py`
- `python scripts/smoke_cpu.py`
- `python scripts/smoke_tpu.py`
- `python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml`
- `python scripts/inspect_targets.py artifacts/teacher_targets/fake_export`
- `python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2`
- `python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2`
- `python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --max-steps 2`
- `python scripts/tpu_distill_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2`
- `python -m pytest`
- `python -m ruff check .`
- `python -m ruff format --check .`

The editable install is intentional. Tests and scripts should import the
installed package rather than depending on `PYTHONPATH=src` or local path
mutation. The fake export and smoke training steps write only gitignored
`artifacts/` content.

The HF backend has stubbed unit/CLI tests in default pytest, but no default test
imports torch/transformers, touches the network, assumes GPU, or runs Qwen.

Qwen policy tests are offline only. Default validation resolves
`Qwen3.latest` as an unresolved local label and runs the Qwen export CLI in
`--dry-run` mode, without loading HF models or writing a Qwen bundle.

Prompt corpus coverage now includes JSONL parsing/validation, stable hashing,
deterministic split assignment, CLI inspection/manifest/split flows, and
teacher-export prompt-source provenance tests.

## Canonical Pipeline Validation

The canonical whole safe path check is:

```bash
python scripts/validate_pipeline.py
```

It covers the default end-to-end pipeline under `.[dev]` without network, HF
model loading, real Qwen export, or TPU requirements. `scripts/validate_local.py`
runs compileall, pipeline validation, pytest, Ruff lint, and Ruff format check.
Optional HF validation uses `--include-hf`; hard TPU validation uses
`--require-tpu` and is reported separately.

## Checkpoint/Resume Coverage

P10 adds CPU-only tests for the JSON + NPZ checkpoint helper, distill
save/resume behavior, CLI flags, mismatch failures, and the validation pipeline
checkpoint smoke commands. The default pipeline writes only under
`checkpoints/`, which is gitignored.

## Run tracking coverage

Run tracking tests cover run id/path helpers, JSON-safe conversion, JSONL metric
append behavior, CLI flags, and distillation integration. The default validation
pipeline includes a one-step tracked distillation smoke under `runs/`.
