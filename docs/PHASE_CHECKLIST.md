# Phase Checklist

## Phase 0 — Foundation

### P0A — Docs and skeleton

- [x] Create project skeleton
- [x] Add pyproject / lint / pytest setup
- [x] Add docs directory
- [x] Add architecture doc
- [x] Add roadmap
- [x] Add decisions log
- [x] Add risk register
- [x] Add artifact format doc
- [x] Add testing strategy
- [x] Add TPU notes
- [x] Add snapshot
- [x] Add append-only Bible
- [x] Add Nyx agent entrypoint
- [x] Add placeholder configs
- [x] Add placeholder scripts
- [x] Add import/layout tests
- [x] Run tests
- [x] Ensure lint/format compliance

### P0.5 — Normalize foundation and add contracts

- [x] Normalize scaffold files into readable multiline content
- [x] Update workflow docs to Nyx-primary / Codex-subagent framing
- [x] Add typed config schema dataclasses
- [x] Add config loading and validation
- [x] Add teacher target manifest schema
- [x] Add manifest validation and round-trip helpers
- [x] Update artifact contract docs to match implemented schema
- [x] Update smoke scripts to load configs
- [x] Add config loading tests
- [x] Add target manifest tests
- [x] Run full validation stack

## P1 — Target Artifact Store Foundation

- [x] Add artifact layout helpers
- [x] Add shard read/write helpers
- [x] Add bundle read/write/inspect/validate helpers
- [x] Add fake target generation script
- [x] Add target inspection script
- [x] Add shard tests
- [x] Add bundle tests
- [x] Update artifact docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation commands

## P2 — Teacher Exporter Interface + Fake Export Pipeline

- [x] Add teacher export config dataclasses
- [x] Add teacher export config loader
- [x] Add export request/result contracts
- [x] Add TeacherExporter protocol
- [x] Add fake deterministic exporter
- [x] Add exporter registry
- [x] Add export_teacher_targets.py CLI
- [x] Update teacher_export_stub.yaml
- [x] Add fake exporter tests
- [x] Add CLI tests
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation commands

## P2.5 — Test Robustness + CI Foundation

- [x] Use editable install as the blessed local and CI workflow
- [x] Remove validation-time dependency installation from `scripts/validate_local.py`
- [x] Mirror CI locally through `scripts/validate_local.py`
- [x] Add GitHub Actions CI for Python 3.11 and 3.12
- [x] Run compileall only on `src`, `scripts`, and `tests`
- [x] Keep tests CPU-only with no JAX, PyTorch, GPU, TPU, or network requirement
- [x] Keep generated fake export outputs under gitignored `artifacts/`
- [x] Document the local validation and individual check commands
- [x] Avoid model, trainer, and heavyweight dependency changes

## P3 — JAX Student Runtime Skeleton

- [x] Add student model interface and output contract
- [x] Add tiny JAX student implementation for trainer smoke coverage
- [x] Add student factory entrypoint
- [x] Add hidden-state MSE train-on-bundle smoke path
- [x] Add `scripts/train_student_smoke.py`
- [x] Add validation commands for student smoke training

## P4 — RWKV7-Style Recurrent Reference Core

- [x] Add `rwkv7_reference` student factory architecture
- [x] Add XLA-friendly scan-based recurrent reference layer
- [x] Add matrix parameterization for reference core projections
- [x] Add attention-mask behavior for recurrent state/output handling
- [x] Add CPU forward, determinism, mask, and JIT tests
- [x] Add smoke training coverage for `rwkv7_reference`
- [x] Document that `rwkv7_reference` is a reference implementation, not a final optimized kernel
- [x] Update snapshot, roadmap, testing strategy, workflow, and README
- [x] Append Bible entry

## P5 — Distillation Stage Runtime

- [x] Add distillation config dataclasses and YAML loading
- [x] Add weighted loss registry and composition helpers
- [x] Integrate hidden-state distillation into the existing train step
- [x] Add optional logits KL loss plumbing and validation
- [x] Add one-stage distillation runner and metrics summaries
- [x] Add `scripts/run_distill_stage.py`
- [x] Add unit and CLI coverage
- [x] Update local validation and CI command sequence
- [x] Update docs, snapshot, decisions, and append-only Bible
