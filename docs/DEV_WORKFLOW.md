# Development Workflow

QRWKV-XLA now uses Nyx as the primary implementation agent.

## Roles

### Nova
- design
- architecture
- specs
- review guidance

### Nyx
- primary implementation agent
- repo-level decisions within the current spec
- file structure consistency
- docs updates
- ensuring tests pass
- deciding when to delegate mechanical tasks

### Codex
- sub-agent used by Nyx for mechanical edits
- targeted code generation
- test writing
- lint/format cleanup
- import/path issue cleanup

## Default workflow

The default operating model is whole-implementation specs executed by Nyx,
optionally with Codex as a sub-agent for mechanical work.

The older Prompt A / B / C split is not the default workflow anymore, though it
can still be resurrected when it is useful for a phase plan.

## Local validation

QRWKV-XLA uses Strategy A editable installs for development:

```bash
python -m pip install -e ".[dev]"
python scripts/validate_local.py
```

`scripts/validate_local.py` does not install dependencies. It mirrors the core
CI validation sequence, prints each command, and fails fast. Scripts and tests
should import `qrwkv_xla` through the editable install instead of setting
`PYTHONPATH=src` or patching `sys.path`.

Generated validation outputs live under `artifacts/`, which is gitignored.

Before handoff, Nyx should run `scripts/validate_local.py` or the equivalent
individual command list from `docs/CI.md`.

For a narrower whole-pipeline handoff check, run:

```bash
python scripts/validate_pipeline.py
```

Optional HF and hard TPU checks are reported separately and are never implied by
default handoff validation:

```bash
python scripts/validate_pipeline.py --include-hf
python scripts/validate_pipeline.py --require-tpu
```

For Phase 4 student-runtime work, the validation path must include both smoke
training commands:

```bash
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --max-steps 2
python scripts/train_student_smoke.py --targets artifacts/teacher_targets/fake_export --student-architecture rwkv7_reference --max-steps 2
```

The `rwkv7_reference` architecture is an XLA-friendly recurrent reference
implementation for correctness and integration checks. It is not the final
optimized RWKV7 kernel.

From Phase 5 onward, new training behavior should prefer the distillation stage
runner (`scripts/run_distill_stage.py`) rather than adding separate one-off
smoke training paths.

P6+ changes touching JAX hot paths should preserve CPU-only CI and should avoid
claiming TPU success unless run with `--require-tpu` on an actual TPU backend.

P7+ teacher export changes should preserve the fake exporter as the default
path. HF/PyTorch work belongs behind `.[teacher-hf]`, should lazy import those
libraries, and should keep network/model-download tests gated behind
`QRWKV_RUN_HF_INTEGRATION=1`.

P8+ Qwen policy changes should keep `Qwen3.latest` as a local label only. Dry-run
policy checks must work offline without `teacher-hf`; real Qwen export remains a
manual action with an explicit model id.

For non-smoke teacher exports, prefer `prompt_corpus` configs over inline
`prompt_texts` so target bundles record corpus IDs and hashes.

Skipped HF integration tests are not failures unless the run explicitly enabled
`QRWKV_RUN_HF_INTEGRATION=1`.

## Checkpoint discipline

Every phase/checkpoint should preserve:
- small reversible changes
- lint clean
- tests passing
- docs updated
- snapshot updated
- Bible appended, never overwritten

## Historical log policy

Only append new phase/update notes to `docs/QRWKV_BIBLE.md`. Do not rewrite
earlier history.

## Current commit themes

- `P0A: scaffold QRWKV-XLA foundation docs`
- `P0.5: normalize scaffold and add config artifact contracts`
- `P2.5: stabilize editable install validation and CI`
- `P4: add RWKV7 reference recurrent core`

## Checkpoint Artifacts

Local checkpoint experiments should write under `checkpoints/`. That directory
is ignored by git and can be deleted between runs. Use `--checkpoint-overwrite`
for repeatable local smoke commands that intentionally replace an existing
checkpoint directory.
## Tracked distillation smoke

Use this when you need local run artifacts while debugging:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --max-steps 1 --track-run --run-name dev-smoke
```

Generated runs are ignored by git under `runs/`.
## Logits KL Checks

When enabling logits KL, confirm the teacher target bundle includes logits and
that the student config has `emit_logits=true`. Real Qwen logits exports should
be treated as expensive and optional, not part of default validation.
