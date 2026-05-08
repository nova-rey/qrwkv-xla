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

## Phase 21 — Multi-Device TPU Sharding Smoke

Goal: add minimum viable data-parallel `pmap` support with replicated params,
sharded batches, gradient averaging, and skip-safe smoke tests.

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

## Phase 13 — Student LM Head + Logits KL Continuation

Goal: add student logits output and logits KL distillation so hidden-only
checkpoints can later continue toward output behavior.

## Phase 14 — Generation Smoke + Tiny Evaluation Harness

Goal: load logits-capable checkpoints and run tiny greedy generation/eval smoke
checks without external tokenizer dependencies.

## Phase 15 — Evaluation Harness + Fixed Regression Prompts

Goal: add fixed generation snapshots, sanity checks, and checkpoint comparison
tools before full benchmark/evaluation work.

## Phase 16 — Adam/AdamW Optimizer

Goal: add dependency-light SGD, Adam, and AdamW optimizer support with
optimizer-state checkpoint/resume, config/CLI wiring, metrics, docs, and tests.

## Phase 20 — Stage 1 Attention/Mixer Target Distillation

Goal: add per-layer attention/mixer output targets and train recurrent mixer
outputs against them before later hidden/logits/CE stages.

## Phase 17 — Learning Rate Scheduling

Goal: add constant and warmup-cosine learning rate schedules with resume-aware
global step behavior, config/CLI support, scheduler metadata in checkpoints and
tracked runs, tests, and validation coverage.

## Phase 18 — Gradient Clipping

Goal: add simple global gradient norm clipping after gradient computation and
before optimizer updates, with config/CLI support, metrics, checkpoint/run
metadata, docs, tests, and CPU-only validation coverage.

## Phase 19 — Stage 3 CE Fine-Tuning

Goal: add the first student-only language-model fine-tuning path. Stage 3 reads
prompt corpus text, tokenizes with `SmokeTokenizer`, builds simple static
next-token batches, requires `emit_logits=true`, and trains with masked CE while
reusing the existing optimizer, schedule, clipping, checkpoint, and tracking
layers.

## Phase 22 — Real Qwen Tokenizer Integration

Goal: keep `SmokeTokenizer` as the default offline test double while adding a
tokenizer registry for optional real HF/Qwen tokenizers. Stage 3 LM configs now
accept string or mapping tokenizer forms, use tokenizer metadata for EOS/PAD and
vocab validation, and keep real-tokenizer tests opt-in.

## Phase 23 — Real Tokenized Data Pipeline

Goal: create reusable tokenized-corpus artifacts with manifest/shard validation
and route Stage 3 training through them without requiring real HF dependencies.

## Phase 25 — Tiny Real Teacher Target Export Proof

Goal: export validated teacher target bundles from fake, prompt, prompt-corpus,
or tokenized-corpus inputs, including `loss_mask` and optional HF provenance.

## Phase 26 — Tiny Real Teacher-to-Student Distillation Proof

Goal: prove the existing JAX distillation runner consumes P25 target bundles
end to end, applies `loss_mask` to token-level target losses, writes finite
metrics/checkpoints, and resumes from target-training checkpoints locally.

## Phase 41 — HF/Safetensors Student Export Smoke

Goal: export a tiny QRWKV-XLA JSON+NPZ student checkpoint to a Hugging
Face-style safetensors directory and prove helper reload output parity locally.

Current checkpoint: `hf_safetensors_export_smoke`. This checkpoint writes
`config.json`, `model.safetensors`, `qrwkv_xla_export.json`, `weight_map.json`,
and smoke reports for a tiny CPU checkpoint only. It does not add a production
HF model class, Qwen-scale export, sharding, `lm_eval`, or model quality claims.
