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

## P6 — XLA Discipline and TPU Smoke Readiness

- [x] Add JAX/XLA runtime inspection utilities
- [x] Add XLA static smoke helper
- [x] Add `xla_inspect.py`

## P21 — Multi-Device TPU Sharding Smoke

- [x] Add distributed config
- [x] Add device topology utilities
- [x] Add batch sharding utilities
- [x] Add param/state replication utilities
- [x] Add pmean reduction helpers
- [x] Add pmap distill smoke
- [x] Add pmap LM smoke
- [x] Add pmap smoke CLI scripts
- [x] Add pmap smoke configs
- [x] Add skip-safe tests
- [x] Update validation pipeline
- [x] Add DISTRIBUTED_TPU_SHARDING.md
- [x] Update TPU smoke guide
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [ ] Run validation
- [x] Add `tpu_distill_smoke.py`
- [x] Improve `smoke_tpu.py`
- [x] Add tiny TPU distill smoke config
- [x] Add XLA/TPU smoke tests
- [x] Update CI/local validation command sequence
- [x] Add `XLA_DISCIPLINE.md`
- [x] Add `TPU_SMOKE_GUIDE.md`
- [x] Clarify canonical `distill` naming and thin `distillation` aliases
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation

## P7 — Optional Hugging Face Teacher Export Backend

- [x] Add optional `teacher-hf` dependency extra
- [x] Extend teacher export config for HF fields
- [x] Add prompt loading helper
- [x] Add HFTeacherExporter
- [x] Register hf backend
- [x] Update `export_teacher_targets.py` flags
- [x] Add tiny HF export config
- [x] Add HF backend docs
- [x] Add mocked/unit tests
- [x] Add optional integration test gate
- [x] Keep fake exporter as the default validation path
- [x] Update README/docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation

## P8 — Offline Qwen Policy Prep

- [x] Add local Qwen policy dataclasses and validation
- [x] Add offline Qwen policy resolver CLI
- [x] Add unresolved Qwen policy YAML
- [x] Add Qwen dry-run and manual export configs without concrete model ids
- [x] Add teacher export CLI dry-run policy resolution
- [x] Keep real Qwen export out of default validation
- [x] Keep torch/transformers out of base and dev dependencies

## P12 — Prompt Corpus + Export Set Management

- [x] Add prompt corpus dataclasses
- [x] Add JSONL read/write/validate
- [x] Add corpus hashing
- [x] Add split helper
- [x] Add corpus inspection CLI
- [x] Add manifest generation CLI
- [x] Add split CLI
- [x] Add smoke prompt corpus
- [x] Add teacher export corpus config fields
- [x] Add prompt source metadata to target manifests
- [x] Add corpus-based export configs
- [x] Add tests
- [x] Update validation pipeline
- [x] Add PROMPT_CORPORA.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry

## P20 — Stage 1 Attention/Mixer Target Distillation

- [x] Define attention_targets shape and semantics
- [x] Update target shard validation/loading
- [x] Add fake attention target export
- [x] Add StudentOutput.mixer_outputs
- [x] Add TinyStudent mixer outputs
- [x] Add RWKV7ReferenceStudent mixer outputs
- [x] Implement attention/mixer MSE loss
- [x] Wire attention_or_mixer loss into runner
- [x] Add fake Stage 1 configs
- [x] Add optional/manual HF attention capture
- [x] Add Qwen attention manual config
- [x] Add tests
- [x] Update validation pipeline
- [x] Add ATTENTION_MIXER_DISTILLATION.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation
- [x] Run validation
- [x] Add offline resolver/config/CLI tests
- [x] Update docs and snapshot

## P9 — Canonical Pipeline Validation Harness

- [x] Add `qrwkv_xla.validation.pipeline`
- [x] Add `scripts/validate_pipeline.py`
- [x] Keep default pipeline CPU-safe, offline, and `.[dev]` only
- [x] Gate tiny HF validation behind `--include-hf`
- [x] Gate hard TPU validation behind `--require-tpu`
- [x] Refactor `scripts/validate_local.py` to call the pipeline harness
- [x] Replace duplicated CI smoke chain with the pipeline harness
- [x] Add unit and CLI help coverage
- [x] Document default, HF, and hard TPU validation modes

## Phase 10 — Checkpoint/Resume

- [x] Add local JSON + NPZ checkpoint helper.
- [x] Add distill config, runner, and CLI save/resume support.
- [x] Keep resume steps additive for each invocation.
- [x] Fail clearly on architecture and student shape config mismatch.
- [x] Add default validation pipeline checkpoint save/resume smoke.
- [x] Document why Orbax is deferred.
## P11 local run tracking

- [x] Add opt-in distillation tracking config and CLI flags.
- [x] Write `run.json`, `metrics.jsonl`, and `summary.json` under `runs/`.
- [x] Default tracked final checkpoints to `runs/<run_id>/checkpoints/final`.
- [x] Keep tracking disabled by default and offline-only.

## P13 — Student LM Head + Logits KL Continuation

- [x] Add LM head module
- [x] Add TinyStudent logits support
- [x] Add RWKV7ReferenceStudent logits support
- [x] Extend student factory/config
- [x] Wire logits KL config validation
- [x] Add fake logits export config
- [x] Add logits distill config
- [x] Add hidden-only -> logits continuation support
- [x] Add LM head/student/logits KL tests
- [x] Update validation pipeline
- [x] Add LOGITS_DISTILLATION.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
## P14 — Generation Smoke + Tiny Evaluation Harness

- [x] Add SmokeTokenizer
- [x] Add greedy generation helper
- [x] Add checkpoint-to-student loader
- [x] Add generation artifact writer
- [x] Add generation smoke harness
- [x] Add generate_from_checkpoint.py
- [x] Add eval_generation_smoke.py
- [x] Add generation_smoke.yaml
- [x] Add generation tests
- [x] Update validation pipeline
- [x] Add GENERATION_SMOKE.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation

## P15 — Evaluation Harness + Fixed Regression Prompts

- [x] Add eval config loader
- [x] Add regression prompt corpus
- [x] Add eval artifact schema
- [x] Add sanity checks
- [x] Add checkpoint evaluation harness
- [x] Add eval snapshot comparison
- [x] Add evaluate_checkpoint.py
- [x] Add compare_eval_snapshots.py
- [x] Add eval tests
- [x] Update validation pipeline
- [x] Add EVALUATION_HARNESS.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation

## P16 — Adam/AdamW Optimizer

- [x] Add optimizer config/state/factory modules
- [x] Move SGD update behind optimizer module while preserving behavior
- [x] Add Adam with bias correction
- [x] Add AdamW with decoupled weight decay
- [x] Persist optimizer config/state in checkpoints
- [x] Wire distillation config, runner, metrics, and CLI flags
- [x] Add optimizer math, checkpoint/resume, integration, and CLI tests
- [x] Add optimizer docs and update phase docs

## P17 — Learning Rate Scheduling

- [x] Add schedule config module
- [x] Add constant schedule
- [x] Add warmup+cosine schedule
- [x] Wire scheduler into distill runner
- [x] Add resume-aware scheduler step counting
- [x] Add scheduler CLI flags
- [x] Add scheduler checkpoint metadata
- [x] Add scheduler run tracking metrics
- [x] Add scheduled AdamW smoke config
- [x] Add scheduler tests
- [x] Update validation pipeline
- [x] Add LR_SCHEDULES.md
- [x] Update docs
- [x] Update snapshot
- [x] Append Bible entry
- [x] Run validation

## P18 — Gradient Clipping

- [x] Add global gradient norm and clipping utilities
- [x] Add distillation gradient config and CLI overrides
- [x] Clip after gradient computation and before optimizer updates
- [x] Report norm, scale, clipped flag, and max norm metrics
- [x] Store gradient config in checkpoints and tracked runs
- [x] Add clipped AdamW smoke config and validation coverage
- [x] Add math, integration, CLI, checkpoint, and validation tests
- [x] Add gradient clipping docs

## P19 — Stage 3 Cross-Entropy Fine-Tuning

- [x] Add masked next-token CE loss
- [x] Add student-only LM stage config, data batching, runner, and CLI
- [x] Reuse prompt corpus records and `SmokeTokenizer`
- [x] Require logits-capable students for Stage 3
- [x] Reuse optimizer, LR schedule, gradient clipping, checkpoint, and tracking layers
- [x] Add tiny CPU-safe Stage 3 smoke config
- [x] Add CE, data, runner, checkpoint resume, and CLI tests
- [x] Add validation pipeline coverage and Stage 3 docs

## P22 — Real Qwen Tokenizer Integration

- [x] Add tokenizer abstraction and registry
- [x] Preserve `SmokeTokenizer` as default offline backend
- [x] Add optional lazy HF tokenizer wrapper and treat `qwen` as `hf`
- [x] Normalize LM tokenizer config from string or mapping forms
- [x] Route LM prompt-corpus tokenization through the registry
- [x] Use tokenizer metadata for EOS, PAD, and vocab compatibility
- [x] Add offline registry/config/LM tests plus mocked HF backend tests
- [x] Add env-gated real HF tokenizer integration test skipped by default
- [x] Add real-tokenizer example config and tokenizer docs

## P23 — Real Tokenized Data Pipeline

- [x] Add tokenized corpus manifest + shard artifact format
- [x] Store Stage 3-ready `input_ids`, `labels`, `attention_mask`, and `loss_mask`
- [x] Record source provenance, tokenizer metadata, packing policy, shard hashes, and totals
- [x] Add deterministic concat-pack tokenization script
- [x] Add tokenized corpus loader validation for format, sequence length, shards, and tokenizer mismatches
- [x] Preserve raw prompt JSONL Stage 3 support
- [x] Add tokenized corpus config, runner, CLI, and validation coverage
- [x] Add focused tokenized corpus, LM integration, and CLI tests
- [x] Add data pipeline docs and append P23 Bible notes
