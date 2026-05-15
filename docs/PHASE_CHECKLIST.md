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

## P24 — RWKV7 Math Parity Audit

- [x] Audit current `rwkv7_reference` math against local RADLADS reference files
- [x] Classify current implementation as a simplified placeholder
- [x] Document equation alignment, state update semantics, masking semantics,
  batch semantics, JIT semantics, gradient sanity, parameter mapping risk, and
  numerical tolerances
- [x] Add minimal CPU-only NumPy parity harness for current JAX placeholder
- [x] Add final-state API extension without breaking existing callers
- [x] Add all-at-once vs token-by-token final-state equivalence test
- [x] Add batched vs unbatched equivalence test
- [x] Add eager vs JIT equivalence test
- [x] Add finite-gradient and tiny optimizer-step no-NaN tests
- [x] Preserve P22/P23 tokenizer and tokenized data paths

## P25 — Tiny Real Teacher Target Export Proof

- [x] Keep HF teacher export isolated behind the optional `teacher-hf` extra
- [x] Support prompt, prompt-corpus, and tokenized-corpus teacher target export
- [x] Preserve fake exporter as the offline deterministic test double
- [x] Require `input_ids`, `attention_mask`, and `loss_mask` in target shards
- [x] Record shard path, ordered SHA-256, example count, and array names in manifests
- [x] Validate target bundle shard hashes, shapes, sequence length, and arrays
- [x] Add loader validation for teacher target bundles
- [x] Record HF provenance including model/tokenizer IDs, revision, dtype,
  trust-remote-code, local-files-only, source metadata, and target flags
- [x] Add fake export, target bundle, CLI, mocked HF backend, and optional HF tests
- [x] Update artifact/export docs and append P25 Bible notes

## P26 — Tiny Real Teacher-to-Student Distillation Proof

- [x] Resolve `teacher.device: auto` to a concrete torch device before `model.to(...)`
- [x] Carry target-bundle `loss_mask` into the JAX distillation training batch
- [x] Use `loss_mask` for token-level hidden-state MSE, logits KL, and attention/mixer loss averaging
- [x] Prove tiny fake teacher target bundle training produces finite loss
- [x] Prove target-training checkpoint save and resume advance global step
- [x] Keep optional real HF smoke env-gated and skipped by default
- [x] Add focused device, loss-mask, target-loading, runner, and checkpoint tests
- [x] Document the repeatable tiny local target-training and resume command path

## P27 — RADLADS-Aligned JAX RWKV7 Reference Math

- [x] Add a separate `rwkv7_radlads_reference` backend without mutating the placeholder
- [x] Preserve the old `rwkv7_reference` smoke backend and its tests
- [x] Implement head-wise matrix recurrent state with documented `[B,H,N,N]` semantics
- [x] Add full-sequence vs token-step equivalence coverage for the new backend
- [x] Add JIT, finite-gradient, and tiny optimizer-step coverage for the new backend
- [x] Wire the new backend through distill and LM config/factory paths
- [x] Add a tiny local stub config for the new backend distill path
- [x] Prove one local distill step runs with the new backend
- [x] Document the backend as partial RADLADS-aligned reference math, not parity

## P28 — RADLADS Reference Backend Validation + Teacher Export Path Cleanup

- [x] Prove `rwkv7_radlads_reference` survives the real distill checkpoint write and resume path
- [x] Keep `rwkv7_reference` placeholder smoke semantics and coverage intact
- [x] Resolve teacher export config input paths relative to the config file
- [x] Resolve teacher export config `runtime.output_dir` relative to the config file
- [x] Preserve CLI path override behavior relative to cwd
- [x] Add focused tests for RADLADS distill resume and teacher export path behavior
- [x] Document `rwkv7_radlads_reference` as a RADLADS-shaped JAX reference backend, not full RADLADS parity.

## P29 — Tiny HF Teacher Export to RADLADS Reference Resume Smoke

- [x] Add tiny HF smoke prompt corpus fixture
- [x] Add checked-in `sshleifer/tiny-gpt2` teacher export config with CPU backend and config-relative output
- [x] Add checked-in hidden-only `rwkv7_radlads_reference` distill config for the tiny HF target bundle
- [x] Keep logits present in the HF target bundle but explicit and unconsumed by this hidden-MSE smoke
- [x] Add offline tests for tiny HF config loading, RADLADS tiny-HF dimensions, target keys, and resume behavior
- [x] Keep real HF execution opt-in/env-gated so default CI has no live network dependency
- [x] Preserve `rwkv7_radlads_reference` as a RADLADS-shaped JAX reference backend, not full RADLADS parity.

## P30 — Real HF Target Loss Hardening + Checked-In Smoke

- [x] Add checked-in logits-enabled `rwkv7_radlads_reference` tiny HF distill config
- [x] Preserve the P29 hidden-only tiny HF distill config and parsing path
- [x] Cover logits config parsing and clear failure when logits KL is enabled without teacher logits
- [x] Cover `loss_mask` handling on the logits KL path
- [x] Exercise `rwkv7_radlads_reference` logits loss with tiny fake logits in offline tests
- [x] Keep live HF export, logits distill, and resume validation opt-in behind `QRWKV_RUN_HF_INTEGRATION=1`
- [x] Preserve warning that `rwkv7_radlads_reference` is RADLADS-shaped, not full RADLADS parity

## P31 — RADLADS RWKV7 Gap Audit

- [x] Inspect current QRWKV-XLA RADLADS-shaped backend source
- [x] Inspect current QRWKV-XLA placeholder RWKV7 backend, student factory, distill config, target bundle surface, configs, and tests
- [x] Inspect local RADLADS RWKV7Qwen2 modeling, config, Triton, and CUDA references
- [x] Add `docs/RADLADS_RWKV7_GAP_AUDIT.md`
- [x] Classify gaps using explicit status and priority labels
- [x] Preserve wording: RADLADS-shaped JAX reference backend, not full RADLADS parity.
- [x] Update risk register, snapshot, checklist, and append-only Bible
- [x] Update decisions only for the supported P32-before-kernels decision
- [x] Recommend concrete P32 scope as math-completion and Qwen block compatibility work
- [x] Avoid Pallas, TPU, `pjit`, sharding, export, quality, and broad code changes

## P32 — Qwen-Compatible RADLADS Slow Reference Backend

- [x] Add selectable `rwkv7_qwen_reference` backend without deleting or mutating the old placeholder backend
- [x] Preserve existing `rwkv7_radlads_reference` behavior and tests
- [x] Implement Qwen decoder shell: input -> RMSNorm -> RWKV/time-mix/reference attention -> residual -> RMSNorm -> MLP -> residual
- [x] Add slow deterministic RoPE support with CPU tests
- [x] Add grouped KV semantics with explicit `num_heads`, `num_kv_heads`, and validated `head_size`
- [x] Add explicit state object with `wkv_matrix_state`, `shift_state`, and `next_position`
- [x] Add full-sequence vs stepwise output and state equivalence coverage
- [x] Add nested Qwen/RADLADS-oriented parameter surface
- [x] Add tiny checked-in distill config for `artifacts/teacher_targets/tiny_hf_smoke`
- [x] Document: Qwen/RADLADS-compatible slow JAX reference path, not optimized kernel parity.
- [x] Preserve boundaries: no optimized kernels, no full RADLADS numerical parity claim, no TPU/Pallas/`pjit`/sharding/export/quality work

## P33 — Tiny Qwen Reference Fixture Harness

- [x] Add deterministic `rwkv7_qwen_reference` fixture generation script
- [x] Add checked-in tiny fixture bundle under `tests/fixtures/qwen_reference/`
- [x] Record full-sequence outputs/logits/final state
- [x] Record stepwise outputs/logits/final state
- [x] Record backend/config/seed/dtype/shapes/hashes/equivalence summary in manifest
- [x] Add no-mask, interior-mask, and prefix/left-padding mask-shape cases
- [x] Lock and document current masked-token behavior without changing semantics
- [x] Add deterministic parameter-surface snapshot coverage
- [x] Add `docs/RWKV7_QWEN_REFERENCE_PARAM_SURFACE.md`
- [x] Keep default fixture generation offline and CPU-safe
- [x] Preserve boundaries: no Pallas, TPU compilation/profiling, `pjit`, sharding, large runs, HF export, `lm_eval`, WandB, full RADLADS checkpoint compatibility, or full RADLADS numerical parity claim

## P34 — Model Scale Planner and Config Generator

- [x] Add `qrwkv_xla.scale_planner` package
- [x] Add model, hardware, and training mode profile dataclasses with validation
- [x] Add built-in model profiles from tiny debug through Qwen-scale stretch estimates
- [x] Add built-in CPU, TPU, GPU, grant, and placeholder hardware profiles
- [x] Add built-in smoke/local/TPU/scale training mode profiles
- [x] Add Qwen/RADLADS reference-backend parameter estimator with documented assumptions
- [x] Add component-level memory estimator including optimizer state, activations, recurrent state, targets, checkpoint reference, and overhead reserve
- [x] Add explicit full-vocab logits target estimate and dominance warning
- [x] Add conservative yes/maybe/no/unknown fit classifier
- [x] Add `scripts/plan_model_scale.py` with readable summary and YAML/JSON output
- [x] Add auto mode for conservative batch/sequence reductions without architecture changes
- [x] Emit planning-only distill config skeletons
- [x] Add `docs/SCALE_PLANNER.md`
- [x] Add checked-in example plans under `docs/examples/scale_plans/`
- [x] Add offline tests for profiles, estimates, CLI, and auto-plan
- [x] Preserve boundaries: no Pallas, TPU execution/profiling, pjit/model sharding, large training, Qwen-scale export/training, student HF export, `lm_eval`, WandB, new model math, or full RADLADS parity claims

## P35 — Parameter Compatibility Bridge

- [ ] Not implemented in this repo state
- [ ] Keep RADLADS checkpoint import/export compatibility as future work
- [ ] Do not claim RADLADS checkpoint compatibility from P36 TPU smoke results

## P36 — Colab TPU Smoke Harness Hardening

- [x] Add checked-in hidden-only `rwkv7_qwen_reference` Colab TPU smoke config
- [x] Add manual `scripts/run_colab_tpu_smoke.py` harness
- [x] Print Python/JAX/backend/device/git summary
- [x] Fail clearly when JAX is not using a TPU backend
- [x] Run tiny JAX matmul sanity before distillation
- [x] Export deterministic fake teacher hidden targets
- [x] Run first one-step distill to stable P36 checkpoint/run paths
- [x] Run resume one-step distill from the first checkpoint
- [x] Validate checkpoint/run artifacts, finite loss metrics, and step progression `1 -> 2`
- [x] Write `artifacts/p36_colab_tpu_smoke/P36_RESULTS.md`
- [x] Write `artifacts/p36_colab_tpu_smoke/p36_results_bundle.tar.gz`
- [x] Add Colab copy/paste docs and scope limits
- [x] Add CPU-only tests for config expectations and harness validation helpers
- [x] Preserve CI as CPU-only; the TPU smoke remains manual and opt-in

## P37 — Colab TPU Logits-KL Smoke Harness

- [x] Add checked-in logits-enabled `rwkv7_qwen_reference` Colab TPU smoke config
- [x] Add checked-in fake teacher export config with `include_logits: true`
- [x] Add manual `scripts/run_colab_tpu_logits_smoke.py` harness
- [x] Reuse shared `qrwkv_xla.smoke.colab_tpu` helpers for P36 and P37
- [x] Preserve P36 hidden-only command behavior
- [x] Validate logits-bearing target manifest and shard arrays
- [x] Validate checkpoint/run artifact existence
- [x] Validate finite `loss`, `hidden_mse`, and `logits_kl`
- [x] Validate optimizer step progression `1 -> 2`
- [x] Keep non-TPU failure message Colab-friendly and unchanged
- [x] Add Colab copy/paste docs and scope limits
- [x] Add CPU-only tests for logits smoke config and validation helpers
- [x] Preserve CI as CPU-only; the TPU logits smoke remains manual and opt-in

## P38 — Real Tiny HF Teacher Targets -> TPU Distill Smoke

- [x] Add checked-in real HF export config for `sshleifer/tiny-gpt2`
- [x] Add checked-in tiny HF `rwkv7_qwen_reference` TPU distill config
- [x] Add manual `scripts/run_tiny_hf_tpu_smoke.py` harness
- [x] Reuse shared `qrwkv_xla.smoke.colab_tpu` helpers for P36, P37, and P38
- [x] Preserve P36 and P37 script behavior
- [x] Validate real-HF target manifest and shard arrays include `input_ids`, `attention_mask`, `hidden_states`, `logits`, and `loss_mask`
- [x] Validate P38 target shapes and basic dimensions
- [x] Validate finite `loss`, `hidden_mse`, and `logits_kl`
- [x] Validate optimizer/checkpoint step progression `1 -> 2`
- [x] Keep non-TPU failure message unchanged
- [x] Add Colab and Kaggle-friendly protocol docs
- [x] Add CPU-only tests for P38 config, target validation, output validation, and bundle contents
- [x] Preserve CI as CPU-only; live HF download and TPU execution remain manual and opt-in

## P39 — QRWKV-XLA Scale Planner Generated TPU Smoke

- [x] Add RoPE-valid tiny HF planner profile for `rwkv7_qwen_reference`
- [x] Add Kaggle TPU v5e hardware planning profile
- [x] Generate P39 planner outputs under `artifacts/p39_planner_tpu_smoke/`
- [x] Generate P39 distill config from the planner output
- [x] Generate P39 real tiny HF teacher export config
- [x] Add manual `scripts/run_planner_tpu_smoke.py` harness
- [x] Reuse shared TPU smoke helpers for target, metric, checkpoint, and bundle validation
- [x] Preserve P36, P37, and P38 script behavior
- [x] Validate planner fit is `yes` or `maybe` for the tiny profile
- [x] Validate real-HF target manifest and shard arrays include masks, hidden states, and logits
- [x] Validate finite `loss`, `hidden_mse`, and `logits_kl`
- [x] Validate optimizer/checkpoint step progression `1 -> 2`
- [x] Keep non-TPU failure message unchanged
- [x] Add Colab and Kaggle copy-paste docs with caveats
- [x] Add CPU-only tests for the P39 harness and generated planner artifacts
- [x] Preserve CI as CPU-only; live HF download and TPU execution remain manual and opt-in

## P40 — RADLADS Source Parity Fixture Bridge

- [x] Add canonical `radlads_source_parity.v1` fixture schema and loader/validator utilities under `src/qrwkv_xla/parity/`
- [x] Add deterministic tiny cases: `tiny_no_mask`, `tiny_attention_mask`, and `tiny_prefix_padding_or_left_padding`
- [x] Keep checked-in default fixtures honest as unsupported QRWKV current-behavior-only payloads, not invented RADLADS outputs
- [x] Add `scripts/import_radlads_source_fixtures.py` with a required canonical source fixture import path
- [x] Keep live RADLADS generation out of normal CI and document it as source-dependent/manual future work
- [x] Add `scripts/compare_radlads_source_fixtures.py` to write `parity_report.json` and `P40_PARITY_REPORT.md`
- [x] Add `scripts/map_radlads_parameter_surface.py` to write `parameter_surface_map.json` and `P40_PARAMETER_SURFACE_MAP.md`
- [x] Add tests for manifest validation, missing arrays, shape mismatches, comparison math, pass/fail/unsupported statuses, report writing, and parameter mapping
- [x] Add `docs/RADLADS_SOURCE_PARITY_BRIDGE.md`
- [x] Preserve offline smoke paths and avoid full RADLADS numerical parity claims

## P41 — QRWKV-XLA JAX Checkpoint -> HF/Safetensors Export Smoke

- [x] Add `qrwkv_xla.export` with HF-style safetensors export and reload helpers
- [x] Add `scripts/export_student_hf_safetensors.py`
- [x] Add `scripts/run_export_smoke.py`
- [x] Write `config.json`, `model.safetensors`, `qrwkv_xla_export.json`, and `weight_map.json`
- [x] Make the smoke write `export_smoke_report.json` and `P41_EXPORT_SMOKE_REPORT.md`
- [x] Compare original checkpoint outputs against reloaded export outputs on a tiny CPU/local batch
- [x] Add `tests/test_hf_safetensors_export.py`
- [x] Add `docs/HF_SAFETENSORS_EXPORT.md`
- [x] Fail clearly when `safetensors` is unavailable
- [x] Preserve scope: no production HF model class, `lm_eval`, Qwen-scale export, sharded export, model quality claim, Pallas/WKV kernels, or unrelated cleanup

## P42 — QRWKV-XLA lm_eval Toy Exported-Student Integration

- [x] Add `qrwkv_xla.eval.exported_student` adapter for P41 exports
- [x] Score deterministic token-id continuation loglikelihoods on CPU/local JAX
- [x] Add tiny toy eval fixture under `tests/fixtures/eval/`
- [x] Add `scripts/run_lm_eval_smoke.py`
- [x] Reuse or generate `artifacts/p41_hf_safetensors_export_smoke`
- [x] Fail clearly for partial/missing required export files
- [x] Write `results.json`, `P42_RESULTS.md`, and `p42_results_bundle.tar.gz`
- [x] Add `tests/test_lm_eval_smoke.py`
- [x] Add `docs/LM_EVAL_SMOKE.md`
- [x] Add optional `eval` extra without changing default dependencies
- [x] Explicitly defer official `lm_eval` execution while documenting the toy harness scope
- [x] Preserve scope: no meaningful benchmark, Qwen-scale eval, production HF model class, training, pjit/sharding, Pallas/WKV kernels, WandB, or unrelated cleanup

## P43 — QRWKV-XLA WKV7 / Pallas Correctness Fixture Harness

- [x] Add `qrwkv_xla.kernels` WKV7 fixture, comparison, and candidate modules
- [x] Add deterministic tiny WKV7 recurrence/state core cases
- [x] Add fixture generation and comparison scripts
- [x] Write P43 artifact manifest, summary, comparison report, and case NPZs
- [x] Include reference candidate support
- [x] Include unsupported Pallas placeholder
- [x] Report full-scan versus stepwise equivalence
- [x] Add CPU-only fixture tests
- [x] Add WKV7 correctness fixture docs
- [x] Update snapshot, roadmap, entrypoint, and append-only Bible

## P44 — QRWKV-XLA Streaming Data Pipeline Dry-Run

- [x] Add `qrwkv_xla.data` streaming dataset manifest and iterator helpers
- [x] Build streaming shards from the existing tokenized corpus artifact path
- [x] Preserve LM/trainer batch keys: `input_ids`, `labels`, `attention_mask`, and `label_mask`
- [x] Support deterministic order, optional shuffle plus seed, and resume cursor replay
- [x] Validate attention and loss masks against tokenizer padding
- [x] Keep validation finite and CPU/local/offline by default
- [x] Add `scripts/build_streaming_data_dry_run.py`
- [x] Add `scripts/run_streaming_data_dry_run.py`
- [x] Add `scripts/run_streaming_trainer_dry_run.py`
- [x] Write artifacts under `artifacts/data/p44_streaming_dry_run`
- [x] Add `tests/test_streaming_data_pipeline.py`
- [x] Add `docs/STREAMING_DATA_PIPELINE.md`
- [x] Preserve scope: no cloud ingest, distributed input, `pjit`/sharding, Pallas, WandB, long training, or quality claims

## P45 — Multi-Scale Model Config Dry-Runs

- [x] Add explicit QRWKV 0.5B, 1.5B, and 7B-stretch planning profiles
- [x] Add P45 hardware planning profiles for local CPU, Colab/Kaggle TPU, and grant TPU slices
- [x] Add `scripts/generate_multiscale_configs.py` with `--out`, `--profiles`, `--hardware`, and `--overwrite`
- [x] Add `scripts/run_multiscale_shape_dry_run.py` with `--scale-plan`, `--metadata-only`, profile selection, and safe init policy
- [x] Write artifacts under `artifacts/scale/p45_multiscale_dry_run`
- [x] Write `P45_RESULTS.md`, `P45_SCALE_PLAN_REPORT.md`, `scale_plan_report.json`, and `fit_matrix.json`
- [x] Generate per-profile config YAML files under `configs/`
- [x] Generate per-profile metadata dry-run JSON under `dry_runs/*/metadata_dry_run.json`
- [x] Generate checkpoint skeleton manifest/config/metadata bundles and validate readback
- [x] Include parameter, optimizer, activation/sequence, target/logits, checkpoint, and total fit components
- [x] Include explicit validation status and warnings for model shapes and parameter bands
- [x] Add `tests/test_multiscale_config_dry_runs.py`
- [x] Add `docs/MULTISCALE_MODEL_DRY_RUNS.md`
- [x] Preserve scope: no real training, `pjit`/sharding, distributed execution, Pallas kernels, WandB, measured full-scale memory, Qwen0.5B teacher target generation, or one-device 7B claims

## P46 — Tiny pjit / Sharding Compile Smoke

- [x] Add `qrwkv_xla.sharding` package
- [x] Add mesh creation helpers with single-device fallback metadata
- [x] Add clean `require_multi_device` failure behavior
- [x] Add `data_parallel_single_axis` policy metadata and explicit shardings
- [x] Add tiny sharding-aware forward/loss/update compile smoke
- [x] Add `scripts/run_pjit_sharding_smoke.py`
- [x] Write `P46_RESULTS.md` and `pjit_sharding_smoke_report.json`
- [x] Add `tests/test_pjit_sharding_smoke.py`
- [x] Add `docs/PJIT_SHARDING_SMOKE.md`
- [x] Update snapshot, roadmap, phase checklist, and agent entrypoint
- [x] Append Bible entry
- [x] Run required P46 validation commands

## P47 — Experiment Tracking / WandB Smoke

- [x] Extend existing `qrwkv_xla.tracking` package with experiment tracker surfaces
- [x] Add mandatory local experiment tracker
- [x] Add optional import-safe WandB adapter
- [x] Add tiny deterministic tracking smoke
- [x] Add `scripts/run_tracking_smoke.py`
- [x] Write P47 local artifacts under `artifacts/p47_experiment_tracking_smoke`
- [x] Record run metadata, exact config, metrics, summary, and artifact manifest
- [x] Add local report helpers for JSON and Markdown
- [x] Add `tests/test_experiment_tracking.py`
- [x] Add `docs/EXPERIMENT_TRACKING.md`
- [x] Update snapshot, roadmap, phase checklist, and agent entrypoint
- [x] Append Bible entry

## P48 — RADLADS LoRA Rank Math Surface

- [x] Add explicit RADLADS-compatible math flags to `RWKV7QwenReferenceConfig`
- [x] Add source-shaped `w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`, `k_k/k_a/r_k`, and `ln_x` parameter leaves
- [x] Keep legacy/default slow-reference behavior gated off by default
- [x] Implement audited low-rank decay and ICLR formulas behind flags
- [x] Thread `v_first` for flagged value residual mixing
- [x] Add flagged balance-state terms and optional attention head group norm
- [x] Keep `r_k` represented but inactive because the inspected source line is commented out
- [x] Add `scripts/run_radlads_lora_rank_math_smoke.py`
- [x] Write P48 smoke/report artifacts under `artifacts/p48_radlads_lora_rank_math`
- [x] Update RADLADS parameter-surface map statuses and caveats
- [x] Add `tests/test_radlads_lora_rank_math.py`
- [x] Add `docs/RADLADS_LORA_RANK_MATH.md`
- [x] Add `docs/RADLADS_LORA_RANK_MATH_AUDIT.md`
- [x] Update snapshot, roadmap, phase checklist, and agent entrypoint
- [x] Append Bible entry

## P49 — RADLADS Tiny Numerical Parity Fixtures

- [x] Add P49 numerical fixture schema, validation, import, comparison, and report helpers
- [x] Add minimal RADLADS parameter mapping statuses for tiny fixture surfaces
- [x] Add `scripts/generate_radlads_tiny_numerical_fixtures.py`
- [x] Add `scripts/import_radlads_tiny_numerical_fixtures.py`
- [x] Add `scripts/compare_radlads_tiny_numerical_fixtures.py`
- [x] Keep live RADLADS execution optional and env-gated
- [x] Use `/home/nyx/.openclaw/workspace/_refs/RADLADS` as the live source path
- [x] Mark offline/default payloads as `missing_source` and not RADLADS outputs
- [x] Report per-case pass/fail/unsupported/missing_source and overall pass/pass_with_known_differences/fail/source_unavailable
- [x] Include float32 and bfloat16 tolerance policy metadata
- [x] Add `docs/RADLADS_NUMERICAL_PARITY_FIXTURES.md`
- [x] Add `tests/test_radlads_numerical_parity.py`
- [x] Update snapshot, roadmap, phase checklist, and agent entrypoint
- [x] Append Bible entry

## P50 — RADLADS Parameter Replay Compatibility

- [x] Add RADLADS parameter-value replay importer
- [x] Add explicit import report buckets for mapped/defaulted/excluded/unsupported/shape_mismatch/missing_required
- [x] Add q_proj/k_proj/v_proj bias support behind replay/config flags
- [x] Add source-backed g1/g2 replay mode for RADLADS gate_rank_type=2
- [x] Keep QRWKV-only surfaces deterministic and reported instead of silently random
- [x] Add replay comparison report for P49 hidden/logit/state and stepwise surfaces
- [x] Add `configs/parity/radlads_tiny_replay.yaml`
- [x] Add `scripts/replay_radlads_tiny_numerical_fixtures.py`
- [x] Add `docs/RADLADS_PARAMETER_REPLAY_COMPATIBILITY.md`
- [x] Add `tests/test_radlads_parameter_replay.py`
- [x] Preserve scope: no Pallas, TPU perf work, real training, full checkpoint import, or HF model class

## Phase 51 — RADLADS Replay Non-Finite Diagnosis and Stabilization

- [x] Reproduce the P50 replay-side `non_finite` failure.
- [x] Add replay tensor summaries and first-nonfinite detection.
- [x] Add parameter sanity reports for the RADLADS payload.
- [x] Add `scripts/diagnose_radlads_replay_nonfinite.py`.
- [x] Fix replay profile selection so simple P49 fixtures do not force all RADLADS math.
- [x] Re-run stabilized replay with P51 reports.
- [x] Add `tests/test_radlads_replay_diagnostics.py`.
- [x] Add `docs/RADLADS_REPLAY_NONFINITE_DIAGNOSTICS.md`.
- [x] Preserve scope: no Pallas, TPU perf work, real training, full checkpoint import, or HF model class.

## Phase 53 — RADLADS vs QRWKV Comparable Output Fixture Parity

- [x] Add `src/qrwkv_xla/parity/radlads_head_to_head.py`.
- [x] Add `scripts/generate_radlads_qrwkv_head_to_head_fixtures.py`.
- [x] Add `scripts/compare_radlads_qrwkv_head_to_head.py`.
- [x] Use the P52 `deterministic_finite` parameter path with seed `5353`.
- [x] Reuse the existing QRWKV RADLADS replay importer for QRWKV outputs.
- [x] Report RADLADS mapped/defaulted/missing/unsupported/shape-mismatch buckets and exact live execution blockers.
- [x] Add per-case/per-surface comparison report fields for shape, dtype, finite flags, and error metrics.
- [x] Add `tests/test_radlads_qrwkv_head_to_head.py`.
- [x] Add `docs/RADLADS_QRWKV_HEAD_TO_HEAD_PARITY.md`.
- [x] Preserve scope: no Pallas, TPU perf work, real training, Qwen-scale export, HF model class, or tolerance loosening.

## P54 — RADLADS Clean Payload Loading and Export

- [x] Add `src/qrwkv_xla/parity/radlads_clean_loader.py`.
- [x] Add `scripts/export_radlads_clean_payload_outputs.py`.
- [x] Extend `scripts/compare_radlads_qrwkv_head_to_head.py` for optional RADLADS and QRWKV output manifests.
- [x] Reuse the clean loader and output-manifest helpers from the P53 head-to-head path.
- [x] Add `tests/test_radlads_clean_loader.py`.
- [x] Add `docs/RADLADS_CLEAN_PARAMETER_LOADER.md`.
- [x] Update snapshot, roadmap, and agent entrypoint.
- [x] Preserve scope: no RADLADS repo vendoring, no Pallas, and no tolerance loosening.

## P55 — RADLADS State/Layout Parity Diagnostics

- [ ] Add surface layout audit and candidate normalization reports.
- [ ] Normalize hidden_states convention explicitly.
- [ ] Diagnose and fix wkv_matrix_state layout/pre-post mismatch if proven.
- [ ] Clarify stepwise surface coverage and status classification.
- [ ] Preserve passing logits and shift_state comparisons.
- [ ] Add docs, snapshot, and Bible updates.
- [ ] Run Ruff and full pytest gates.


## P56 — RADLADS WKV State Residual Trace

- [x] Capture RADLADS and QRWKV WKV traces on `tiny_no_mask`.
- [x] Compare semantic trace stages and identify the first divergent WKV stage.
- [x] Run update-order candidate analysis.
- [x] Preserve passing logits and shift_state.
- [x] Update docs, snapshot, and Bible notes.
- [x] Run full Ruff and pytest gates.

## P57 — RADLADS log_w Decay Parity Caliper

- [x] Add `src/qrwkv_xla/parity/radlads_log_w_parity.py`.
- [x] Load RADLADS `log_w` rows from JSONL trace artifacts.
- [x] Capture current QRWKV `log_w` via diagnostics.
- [x] Compare RADLADS and QRWKV `log_w` rows with existing strict tolerances.
- [x] Evaluate candidate formula variants for orientation, sign, activation, base-term, dtype, and axis handling.
- [x] Add `scripts/compare_radlads_qrwkv_log_w.py`.
- [x] Add `tests/test_radlads_log_w_decay_parity.py`.
- [x] Add `docs/RADLADS_LOG_W_DECAY_PARITY.md`.
- [x] Preserve scope: diagnostic-only, no model patch unless a separate source-backed fix phase is opened.

## P58 — RADLADS log_w / Decay Source-Backed Fix

- [x] Keep the low-rank decay path active on the simple tiny replay profile.
- [x] Record the low-rank decay head-split diagnostic so the candidate caliper can align.
- [x] Re-run the log_w caliper and confirm an exact pass on `tiny_no_mask`.
- [x] Re-run the WKV trace and head-to-head comparison after the fix.
- [x] Update the Bible, snapshot, roadmap, and entrypoint notes.
- [x] Write the P58 reports and preserve before/after artifacts.

## P59 — RADLADS WKV State Provenance

- [x] Add `src/qrwkv_xla/parity/radlads_wkv_state_provenance.py`.
- [x] Add trace and compare scripts for WKV state handoff provenance.
- [x] Add JSONL writer/reader roundtrips and schema validation.
- [x] Compare initial state, token carry, full-vs-stepwise, and mask behavior.
- [x] Use synthetic/tmp_path tests so CI does not depend on ignored artifacts.
- [x] Preserve P58 log_w fixes and avoid recurrence-math rewrites.
- [x] Add `docs/RADLADS_WKV_STATE_PROVENANCE.md`.

## P60 — Real RADLADS/QRWKV WKV State Provenance

- [x] Add `scripts/run_real_radlads_qrwkv_wkv_state_provenance.py`.
- [x] Add `scripts/compare_real_radlads_qrwkv_wkv_state_provenance.py`.
- [x] Reuse P59 provenance schema and P58/P54 cached real artifacts.
- [x] Emit explicit real/synthetic/self/cached/regenerated metadata labels.
- [x] Fail `--strict-real-artifacts` instead of falling back to synthetic rows.
- [x] Add case, comparison, trace provenance, and hidden-state dependency reports.
- [x] Add `tests/test_real_wkv_state_provenance.py`.
- [x] Preserve scope: no Pallas, no broad math rewrite, no tolerance widening, no synthetic fallback.

## P61 — WKV Matrix-State Export Convention Audit

- [x] Add `docs/RADLADS_WKV_STATE_EXPORT_CONVENTION.md`.
- [x] Add `src/qrwkv_xla/parity/radlads_wkv_state_convention.py`.
- [x] Add `scripts/inspect_radlads_qrwkv_wkv_state_slots.py`.
- [x] Add `scripts/compare_radlads_qrwkv_head_to_head_normalized_state.py`.
- [x] Add `tests/test_radlads_wkv_state_convention.py`.
- [x] Keep the audit source-backed and diagnostic-first; no tolerance widening, no recurrence rewrite, no Pallas.

## P62 — QRWKV-XLA WKV Update-Term / State-After Residual Parity Fix

- [x] Add `docs/RADLADS_WKV_UPDATE_RESIDUAL_PARITY.md`.
- [x] Add `src/qrwkv_xla/parity/radlads_wkv_update_residual.py`.
- [x] Add `scripts/trace_radlads_qrwkv_wkv_update_residual.py`.
- [x] Add `scripts/compare_radlads_qrwkv_wkv_update_residual.py`.
- [x] Add `tests/test_radlads_wkv_update_residual.py`.
- [x] Produce `artifacts/p62_wkv_update_residual` from real paired post-P58 trace artifacts.
- [x] Record explicit unavailable reasons for missing update-stage surfaces.
- [x] Include first residual reconstruction, outer-product convention, decay application, dtype accumulation, and mask/update interaction audits.
- [x] Preserve P58 `log_w` fix and P61 slot/export conclusion.
- [x] Leave `kernel_ready: no`; no source-backed numeric fix was proven.
- [x] Preserve scope: no Pallas, no tolerance loosening, no broad recurrence rewrite, no RADLADS output changes.

## P63 — QRWKV-XLA WKV Live Update Hooks

- [x] Add `docs/RADLADS_WKV_LIVE_UPDATE_HOOKS.md`.
- [x] Extend live hook support in `src/qrwkv_xla/parity/radlads_wkv_live_update_hooks.py`.
- [x] Add `scripts/trace_radlads_qrwkv_wkv_live_update_hooks.py`.
- [x] Add `scripts/compare_radlads_qrwkv_wkv_live_update_hooks.py`.
- [x] Add `tests/test_radlads_wkv_live_update_hooks.py`.
- [x] Expose or explicitly label live vs reconstructed substages for the remaining WKV update path.
- [x] Keep the phase diagnostic-first; no recurrence rewrite, no Pallas, no tolerance widening.

## P64 — WKV Composite Balance-State Hook

- [x] Add `docs/RADLADS_WKV_COMPOSITE_BALANCE_HOOK.md`.
- [x] Add P64 locator, extractor, and comparison scripts with `--help`.
- [x] Extend the P63 live-hook helper with `composite_balance_update_term`
  labels, live source aliases, and labeled reconstruction handling.
- [x] Add CI-safe P64 tests with tmp-path JSONL fixtures.
- [x] Preserve P58/P61/P63 behavior: no recurrence math changes, no Pallas,
  no tolerance loosening, no broad rewrite.

## P65 — Balance-State Experiment Surface

- [x] Add `docs/RADLADS_WKV_BALANCE_STATE_EXPERIMENT.md`.
- [x] Reuse `radlads_balance_state_terms` and `radlads_balance_state` as the
  explicit experimental switches.
- [x] Add `scripts/run_balance_state_experiment.py`.
- [x] Add `scripts/run_balance_state_stability_smoke.py`.
- [x] Add `tests/test_balance_state_experiment.py`.
- [x] Compare off vs experimental mode on tiny fixture inputs for `log_w`,
  logits, hidden states, WKV matrix state, shift state, finite counts, and first
  divergent stage.
- [x] Preserve default/off behavior, P58 `log_w`, and P63/P64 behavior.
- [x] Keep scope local/CPU/tiny with no Pallas, no tolerance loosening, and no
  default promotion.

## P66 — Balance-State Experimental vs RADLADS Three-Way Parity

- [x] Add `src/qrwkv_xla/parity/radlads_balance_state_three_way.py`.
- [x] Add `scripts/run_balance_state_radlads_three_way.py`.
- [x] Add `docs/RADLADS_BALANCE_STATE_THREE_WAY_PARITY.md`.
- [x] Add `tests/test_balance_state_three_way_parity.py`.
- [x] Produce `artifacts/p66_balance_state_radlads_three_way/` with three raw
  update-boundary JSONL traces and compact markdown reports.
- [x] Compare RADLADS, QRWKV off, and QRWKV experimental balance-state rows
  without changing recurrence semantics.
- [x] Emit exactly one decision recommendation: `P67 promote/harden balance-state compatibility path`.
- [x] Preserve default/off behavior, P58 `log_w`, strict real-artifact
  provenance, and no synthetic fallback.
- [x] Keep scope local/CPU/tiny with no Pallas, no tolerance loosening, no
  default promotion, and no model-quality claim.
