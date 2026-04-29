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
- P3A: RWKV7 math spec + shape contracts
- P3B: scan-based recurrent block implementation
- P3C: correctness tests, numerical stability notes, CPU training sanity

## Phase 5 — Distillation Stages
Goal: staged RADLADS-like training.

Checkpoints:
- P4A: hidden-state distillation
- P4B: logit distillation
- P4C: attention/mixer behavior distillation
- P4D: staged schedule runner

## Phase 6 — TPU Smoke and XLA Discipline
Goal: prove the code runs on actual TPU environments.

Checkpoints:
- P5A: Kaggle/Colab TPU smoke script
- P5B: static-shape audit
- P5C: compile/recompile logging and known XLA pitfalls doc

## Phase 7 — Qwen Teacher Integration
Goal: export real targets from selected Qwen teacher.

Checkpoints:
- P6A: resolve Qwen3.latest model ID into metadata
- P6B: export small target shards
- P6C: train student on real exported targets

## Phase 8 — Evaluation and Export
Goal: generation sanity, teacher/student comparison, packaging.

Checkpoints:
- P7A: eval harness skeleton
- P7B: recurrent inference path
- P7C: export format and model card artifacts

## Phase 9 — TPU Scale-Up
Goal: multi-device/multi-host TPU training plan.

Checkpoints:
- P8A: sharding strategy
- P8B: multi-device training smoke
- P8C: larger teacher/student experiments
