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

Checkpoint: `hf_safetensors_export_smoke`. This checkpoint writes
`config.json`, `model.safetensors`, `qrwkv_xla_export.json`, `weight_map.json`,
and smoke reports for a tiny CPU checkpoint only. It does not add a production
HF model class, Qwen-scale export, sharding, `lm_eval`, or model quality claims.

## Phase 42 — lm_eval-Style Exported-Student Smoke

Goal: load the P41 exported-student artifact and score deterministic tiny
continuation examples through a local lm_eval-style harness.

Current checkpoint: `lm_eval_toy_exported_student_smoke`. This checkpoint writes
`results.json`, `P42_RESULTS.md`, and `p42_results_bundle.tar.gz` under
`artifacts/eval/p42_lm_eval_smoke`. Official `lm_eval` task execution remains
deferred and optional; P42 does not claim benchmark quality, Qwen-scale eval, a
production HF model class, training, sharding, or optimized kernels.

## Phase 43 — WKV7 Correctness Fixture Harness

Goal: add deterministic tiny correctness fixtures for the extracted WKV7
recurrence/state core before implementing optimized kernels.

Current checkpoint: `wkv7_correctness_fixture_harness`. This checkpoint writes
fixtures under `artifacts/kernels/p43_wkv7_correctness`, compares the
`reference` candidate, records full-scan versus stepwise equivalence, and keeps
`pallas` as an explicit unsupported placeholder. It does not add TPU
benchmarking, optimized Pallas kernels, full Qwen/RADLADS parity, or scale
execution.

## Phase 44 — Streaming Data Pipeline Dry-Run

Goal: add a larger local/offline streaming dataset abstraction that can consume
the existing tokenized corpus artifact path, expose LM/trainer-compatible
batches, and prove deterministic cursor resume.

Current checkpoint: `streaming_data_pipeline_dry_run`. This checkpoint writes a
root manifest plus shard files under `artifacts/data/p44_streaming_dry_run`,
emits dataset/streaming/trainer Markdown+JSON reports, validates deterministic
iteration, optional seeded shuffle, resume cursor replay, attention/loss masks,
and bounded token accounting. It is not a real training phase and does not
prove full-scale throughput, real training quality, cloud or distributed input,
sharding, Pallas, WandB, or Qwen-scale target generation.

## Phase 45 — Multi-Scale Model Config Dry-Runs

Goal: generate bounded planning configs and metadata-only dry-run artifacts for
QRWKV 0.5B, 1.5B, and 7B-stretch model profiles across local/TPU planning
hardware profiles.

Current checkpoint: `multiscale_model_config_dry_runs`. This checkpoint writes
`P45_RESULTS.md`, `P45_SCALE_PLAN_REPORT.md`, `scale_plan_report.json`,
`fit_matrix.json`, per-profile config YAML files, per-profile metadata dry-run
JSON, and checkpoint skeleton manifest/config/metadata bundles under
`artifacts/scale/p45_multiscale_dry_run`. It validates model shapes and
parameter bands, reports component memory estimates and fit classification, and
keeps full large-model init blocked by default. It does not prove real training,
pjit/sharding, distributed execution, Pallas kernels, WandB, measured full-scale
memory, Qwen teacher target generation, or one-device 7B training.

## Phase 46 — Tiny pjit / Sharding Compile Smoke

Goal: add a bounded pjit/jit-with-shardings compile smoke that creates a named
JAX mesh, applies an explicit data-parallel sharding policy, and verifies a
finite tiny forward/loss/update path.

Current checkpoint: `pjit_sharding_compile_smoke`. This checkpoint writes
`P46_RESULTS.md` and `pjit_sharding_smoke_report.json` under
`artifacts/p46_pjit_sharding_smoke`. It records backend/platform/device/mesh
metadata, compile API, policy metadata, finite loss, update status, and honest
single-device fallback. It does not prove large-model sharding, throughput,
production sharded checkpointing, multi-host training, Pallas kernels, Qwen
scale training, or that P45 large profiles are trainable.

## Phase 47 — Experiment Tracking / WandB Smoke

Goal: extend the existing local tracking package with a small experiment
tracker abstraction, a durable local tracker, an optional import-safe WandB
adapter, and a tiny deterministic smoke report.

Current checkpoint: `experiment_tracking_smoke`. This checkpoint writes
`P47_RESULTS.md`, `tracking_smoke_report.json`, and a `local_run/` directory
with `run_metadata.json`, `config.json`, `metrics.jsonl`, `summary.json`,
`artifacts_manifest.json`, and copied artifacts under `files/`. It records run
metadata, exact smoke config, required training metrics, local artifact hashes,
and optional WandB handoff when explicitly requested. It does not require WandB
for normal development or CI and does not prove real training, production
dashboards, sweeps, online logging, official benchmarks, or model quality.

## Phase 48 — RADLADS LoRA Rank Math Surface

Goal: complete the slow-reference RADLADS low-rank math surface for the
Qwen/RADLADS reference backend while preserving legacy defaults.

Current checkpoint: `radlads_lora_rank_math_surface`. This checkpoint adds
explicit flags for low-rank decay, low-rank ICLR, value residual mixing,
balance-state terms, attention group norm, and an overall
`radlads_compatible_math` mode. It adds source-shaped parameter leaves for
`w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`, `k_k/k_a/r_k`, and `ln_x`, plus a P48
smoke that writes `P48_RESULTS.md`, `lora_rank_math_report.json`,
`P48_PARAMETER_SURFACE_MAP.md`, and `parameter_surface_map.json` under
`artifacts/p48_radlads_lora_rank_math`.

P48 is CPU/offline slow-reference work only. It does not claim full RADLADS
numerical parity, fitted conversion, optimized WKV/Pallas kernels, TPU
performance, Qwen-scale execution, or active `r_k` residual math.

## Phase 49 — RADLADS Tiny Numerical Parity Fixtures

Goal: add a bounded real-tiny numerical fixture import/comparison path for
RADLADS source arrays while keeping normal CI offline-safe.

Current checkpoint: `radlads_tiny_numerical_parity_fixtures`. This checkpoint
adds P49 manifest validation, fixture import, optional env-gated live-source
generation hooks, comparison reports, and minimal parameter mapping statuses.
Required tiny cases cover no-mask, attention-mask, prefix/left-padding,
stepwise state, and all explicit P48 RADLADS math flags enabled. Default
payload generation writes QRWKV-XLA current-behavior arrays only and marks
cases `missing_source`; these payloads are not RADLADS outputs.

P49 does not claim full RADLADS numerical parity, training, checkpoint import,
Pallas or optimized WKV kernels, a Hugging Face model class, Qwen-scale
execution, or large-scale parity.

## Phase 50 — RADLADS Parameter Replay Compatibility

Goal: load the tiny real RADLADS P49 parameter payload into the explicit
QRWKV-XLA slow replay mode and produce honest import plus numerical replay
reports.

Current checkpoint: `radlads_parameter_replay_compatibility`. This checkpoint
adds q/k/v projection bias support, source-backed `g1/g2` gate replay for
RADLADS `gate_rank_type == 2`, deterministic/defaulted reporting for QRWKV-only
surfaces, and replay comparisons for P49 hidden, logits, WKV state, shift state,
and stepwise surfaces.

P50 remains a bounded replay-compatibility phase. It does not claim full
RADLADS parity, full checkpoint import, real training, Pallas kernels, TPU
performance, Qwen-scale execution, or a Hugging Face model class.

## Phase 51 — RADLADS Replay Non-Finite Diagnosis and Stabilization

Next checkpoint: `radlads_replay_nonfinite_diagnostics`.

This phase instruments replay tensors, writes parameter sanity reports, and
keeps replay profiles aligned with the original P49 fixture math flags. The
goal is finite slow-reference replay for at least the simple real fixtures
before any Pallas/kernel work resumes.

## Phase 53 — RADLADS vs QRWKV Comparable Output Fixture Parity

Current checkpoint: `radlads_qrwkv_head_to_head_parity`.

This checkpoint adds a tiny head-to-head fixture path under
`artifacts/p53_radlads_qrwkv_head_to_head`. It generates the clean parameter
payload through the existing P52 `deterministic_finite` path with seed `5353`,
runs QRWKV-XLA through the existing replay importer, and attempts live RADLADS
execution against the same payload. Reports include per-case/per-surface status,
shape, dtype, finite flags, error metrics, and exact blockers when RADLADS
cannot load or execute.

P53 does not add Pallas, TPU optimization, real training, Qwen-scale export,
HF `PreTrainedModel` support, multi-host sharding, tolerance loosening, or
fabricated RADLADS outputs.

## Phase 54 — RADLADS Clean Payload Loading and Export
Goal: load the clean deterministic_finite RADLADS payload against the live
RADLADS boundary, classify unsupported leaves and gate-rank shape mismatches
explicitly, and export runnable RADLADS outputs for comparison.

Current checkpoint: `radlads_clean_payload_loader`. This checkpoint adds the
clean loader, output exporter, and optional RADLADS/QRWKV output-manifest
consumption path for the P53/P54 tiny fixture flow.

P54 remains bounded loader/export work only. It does not add Pallas, repo
vendoring, tolerance loosening, or full RADLADS parity claims.

## Phase 55 — RADLADS State/Layout Parity Diagnostics
Goal: diagnose the remaining tiny parity differences with explicit surface
layout and candidate-transform analysis before any kernel work.

Checkpoints:
- P55A: surface layout audit
- P55B: candidate normalization analysis
- P55C: hidden_states convention normalization
- P55D: wkv_matrix_state pre/post or axis resolution
- P55E: stepwise coverage cleanup

Current checkpoint: `radlads_state_layout_parity_diagnostics`.

P55 remains diagnostic-first. It does not add Pallas, TPU optimization, real
training, Qwen-scale export, or model-quality claims.


## Phase 56 — RADLADS WKV State Residual Trace
Goal: trace the remaining finite WKV residual after P55 ruled out simple layout/export conventions.

Checkpoints:
- P56A: trace capture on `tiny_no_mask`
- P56B: trace comparison and first-divergence identification
- P56C: update-order candidate analysis
- P56D: preserve logits and shift_state

Current checkpoint: `radlads_wkv_state_residual_trace`.

P56 is diagnostic-first and does not add Pallas, TPU optimization, real training, or model-quality claims.

## Phase 57 — RADLADS log_w Decay Parity Caliper
Goal: isolate the P56 first divergent stage with a dedicated source-audit
comparison for `log_w`.

Checkpoints:
- P57A: load RADLADS `log_w` rows from JSONL trace artifacts
- P57B: capture QRWKV `log_w` from the current diagnostics path
- P57C: compare `log_w` values and write parity artifacts
- P57D: evaluate non-mutating formula candidates for orientation/sign/activation/base-term/dtype/axis

Current checkpoint: `radlads_log_w_decay_parity`.

P57 is diagnostic-only. It does not patch model math, loosen tolerances, add
Pallas, run TPU optimization, or claim broader RADLADS parity.

## Phase 58 — RADLADS log_w / Decay Source-Backed Fix
Goal: apply the smallest source-backed QRWKV-XLA fix for the tiny `log_w` /
decay divergence and verify the downstream trace impact.

Current checkpoint: `radlads_log_w_decay_source_fix`.

P58 keeps the low-rank decay path active for the simple tiny replay profile,
re-runs the log_w caliper, traces the downstream WKV state, and preserves the
passing logits and shift_state surfaces.

P58 does not implement Pallas, TPU optimization, real training, or a claim of
full RADLADS parity.

## Phase 59 — RADLADS WKV State Provenance
Goal: make WKV state handoff provenance explicit before any further recurrence
or kernel work.

Checkpoints:
- P59A: JSONL schema and report helpers for provenance rows
- P59B: QRWKV synthetic trace over initial state, token carry, full-vs-stepwise,
  and mask behavior
- P59C: provenance JSONL comparison script
- P59D: focused tmp_path tests and documentation

Current checkpoint: `radlads_wkv_state_provenance`.

P59 is diagnostic-only. It does not change recurrence math, widen tolerances,
alter P58 log_w behavior, add Pallas, or claim full RADLADS parity.

## Phase 60 — Real RADLADS/QRWKV WKV State Provenance
Goal: bind WKV state provenance to real paired tiny RADLADS and QRWKV cached
artifacts without introducing synthetic substitutions.

Checkpoints:
- P60A: derive provenance JSONL from `artifacts/p54_confirmation` outputs
- P60B: reuse P58 post-fix WKV trace rows for initial/token-carry state checks
- P60C: compare RADLADS and QRWKV provenance with deterministic first divergence
- P60D: write source provenance, case, mask/padding, and hidden-state dependency reports

Current checkpoint: `real_radlads_qrwkv_wkv_state_provenance`.

P60 is diagnostic/reporting work only. It does not add Pallas, widen
tolerances, rewrite recurrence math, or fabricate source traces. Strict
real-artifact mode fails when only cached-derived outputs are available.

## Phase 61 — WKV Matrix-State Export Convention Audit
Goal: audit and normalize the remaining RADLADS-vs-QRWKV WKV matrix-state
export/slot convention gap without broad math changes.

Checkpoints:
- P61A: slot/export audit docs and scripts
- P61B: source-backed normalization helper for WKV matrix-state comparison
- P61C: normalized comparison report and hidden-state side audit
- P61D: CI-safe tmp_path tests plus real-artifact local verification

Current checkpoint: `wkv_matrix_state_export_convention_audit`.

P61 stays diagnostic-first. It does not add Pallas, widen tolerances, rewrite
recurrent math, or change RADLADS outputs.
