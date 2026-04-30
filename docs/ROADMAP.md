# QRWKV-XLA Roadmap

## Phase 0 — Foundation
Goal: docs, skeleton, architecture, configs, test tiers.

Checkpoints:
- P0A: docs + repo skeleton
- P0B: config/dataclass skeleton + placeholder CLI scripts
- P0C: CI/lint/test cleanup and docs sync

## Phase 1 — Target Artifact Store Foundation
Goal: implement the first usable teacher target artifact storage layer.

Status: complete locally for fake/CPU-only bundles.

Checkpoints:
- P1A: artifact schema docs + sample shard format
- P1B: manifest JSON + NPZ shard read/write/validation
- P1C: fake bundle generator, inspector, and bundle tests

## Phase 2 — Teacher Exporter Interface
Goal: create reusable exporter contracts and a fake exporter pipeline.

Checkpoints:
- P2A: teacher export config + request/result contracts
- P2B: deterministic fake exporter + registry + CLI
- P2C: exporter tests + docs sync

## Phase 3 — JAX Student Runtime Skeleton
Goal: real JAX training loop shape without final RWKV7 complexity.

Checkpoints:
- P3A: JAX module interfaces + training state contracts
- P3B: recurrent student shell + loss registry + CPU train step
- P3C: checkpoint/resume + smoke scripts

## Phase 4 — RWKV7-Style Recurrent Core
Goal: implement XLA-friendly RWKV7-style recurrence.

Checkpoints:
- P4A: RWKV7 reference math spec + shape contracts
- P4B: scan-based recurrent reference block implementation
- P4C: correctness tests, numerical stability notes, CPU/JIT/gradient sanity

Current checkpoint: `rwkv7_reference_core`. This checkpoint provides an
XLA-friendly recurrent reference implementation for the student path; it is not
the final optimized RWKV7 kernel.

## Phase 5 — Distillation Stages
Goal: staged RADLADS-like training.

Checkpoints:
- P5A: hidden-state distillation complete for CPU stage runtime
- P5B: logit distillation plumbing complete; student logits head deferred
- P5C: attention/mixer behavior distillation
- P5D: staged schedule runner complete for one-stage configs

Current checkpoint: `distillation_stage_runtime`. This checkpoint runs a
configured hidden-state distillation stage on fake/exported target bundles,
while keeping logits KL opt-in and clearly validated until student logits are
implemented.

## Phase 6 — TPU Smoke and XLA Discipline
Goal: prove the code runs on actual TPU environments.

Checkpoints:
- P6A: Kaggle/Colab TPU smoke script
- P6B: static-shape audit
- P6C: compile/recompile logging and known XLA pitfalls doc

## Phase 7 — Qwen Teacher Integration
Goal: export real targets from selected Qwen teacher.

Checkpoints:
- P7A: resolve Qwen3.latest model ID into metadata
- P7B: export small target shards
- P7C: train student on real exported targets

## Phase 8 — Evaluation and Export
Goal: generation sanity, teacher/student comparison, packaging.

Checkpoints:
- P8A: eval harness skeleton
- P8B: recurrent inference path
- P8C: export format and model card artifacts

## Phase 9 — TPU Scale-Up
Goal: multi-device/multi-host TPU training plan.

Checkpoints:
- P9A: sharding strategy
- P9B: multi-device training smoke
- P9C: larger teacher/student experiments
