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

## Phase 6 — XLA Discipline + TPU Smoke Readiness
Goal: harden JAX/XLA runtime discipline and provide TPU-ready smoke scripts
that degrade gracefully without TPU.

Checkpoints:
- P6A: JAX runtime inspection and backend visibility
- P6B: static-shape and JIT smoke coverage
- P6C: TPU-ready launcher scripts and docs

## Phase 7 — Hugging Face Teacher Exporter Backend
Goal: add optional HF/PyTorch teacher export using small public models before
Qwen-scale exports.

Checkpoints:
- P7A: optional HF/PyTorch backend behind `teacher-hf`
- P7B: resolve Qwen3.latest model ID into metadata
- P7C: train student on real exported targets

Current checkpoint: `hf_teacher_exporter_backend`. This checkpoint adds
the backend and artifact path without making torch/transformers base
dependencies, without adding network-required default tests, and without running
Qwen smoke by default.

## Phase 8 — Qwen Policy Prep, Evaluation, and Export
Goal: offline Qwen policy prep, generation sanity, teacher/student comparison,
packaging.

Checkpoints:
- P8A: local/offline Qwen policy resolver and dry-run export preparation
- P8B: eval harness skeleton
- P8C: recurrent inference path
- P8D: export format and model card artifacts

Current checkpoint: `qwen_policy_offline_prep`. This checkpoint keeps Qwen
export manual-only, avoids automatic model lookup, and preserves default
validation under `.[dev]` without `teacher-hf`.

## Phase 9 — TPU Scale-Up
Goal: multi-device/multi-host TPU training plan.

Checkpoints:
- P9A: canonical pipeline validation harness
- P9B: sharding strategy
- P9C: multi-device training smoke
- P9D: larger teacher/student experiments

Current checkpoint: `pipeline_validation_harness`. This checkpoint makes
`scripts/validate_pipeline.py` the canonical safe end-to-end validation command
and keeps optional HF and hard TPU paths behind explicit flags.

## Phase 12 — Prompt Corpus + Export Set Management
Goal: make teacher-export inputs reproducible through prompt corpus JSONL,
manifests, stable hashes, deterministic splits, corpus inspection tools,
corpus-based teacher export configs, and target-manifest prompt-source
provenance.

## Phase 10 — Checkpoint/Resume + Staged Continuation
Goal: add local checkpoint save/resume for staged distillation without changing
the dependency boundary.

Checkpoints:
- P10A: JSON + NPZ checkpoint helper
- P10B: distill runner save/resume and CLI flags
- P10C: validation pipeline checkpoint smoke
- P10D: staged hidden-only to later logits continuation docs

Current checkpoint: `checkpoint_resume_staged_continuation`. This checkpoint
keeps checkpointing local, CPU-safe, offline, and under `checkpoints/`.

## P11 run tracking

P11 adds opt-in local run tracking for staged distillation. The next useful
extensions are richer evaluation summaries and cross-run comparison helpers,
still using local artifacts under `runs/`.
