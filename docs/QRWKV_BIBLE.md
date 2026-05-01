# QRWKV-XLA Bible

This file is append-only. New phases and decisions should be appended below rather than rewriting earlier history.

## Phase 0 / Prompt A — Project Birth

QRWKV-XLA begins as a JAX/XLA-first reimplementation of the RADLADS-style recurrent conversion idea.

The existing RADLADS repository remains the reference implementation and design ancestor, but this project is not a port. The original repo includes GPU-oriented dependencies and CUDA/Triton paths, so QRWKV-XLA will be built with XLA and TPU constraints as first-class design requirements.

Initial target:
- Teacher: Qwen3.latest policy, with Qwen3.0 fallback
- Student: RWKV7-style recurrent architecture
- Runtime: CPU debug first, TPU smoke second, TPU scale later
- Workflow: Nova design/specs, Nyx/Codex implementation

## Phase 0.5 — Foundation Normalization and Contracts

The initial scaffold existed but needed normalization into valid, readable
multiline files. This pass updates the workflow model from A/B/C prompts to Nyx
as the primary implementation agent with Codex as a sub-agent, then adds the
first config and target artifact contract layer.

## Phase 1 — Target Artifact Store Foundation

This phase adds the first durable data contract between future PyTorch/Hugging
Face teacher extraction and future JAX/XLA student training. The project can
now create fake teacher target bundles, write and read manifest JSON, validate
NPZ shards, inspect bundle metadata, and test the artifact store without
requiring GPU, TPU, PyTorch, JAX, or network access.

## Phase 2 — Teacher Exporter Interface + Fake Export Pipeline

This phase adds the exporter-shaped side of the pipeline. QRWKV-XLA now has a
teacher export configuration schema, export request/result contracts, a
TeacherExporter protocol, a deterministic fake exporter, and a CLI entrypoint
that writes valid target bundles through the artifact store. Real
Qwen/PyTorch/Hugging Face loading remains intentionally deferred.

## Phase 2.5 — Editable Install Validation and CI Stabilization

This stabilization phase keeps the P2 behavior intact while standardizing how
the repository is validated. QRWKV-XLA now treats `python -m pip install -e ".[dev]"`
as the blessed development setup, with scripts and tests importing the installed
package instead of using `PYTHONPATH=src` or script-local path patching.

Local validation mirrors the core CI commands through `scripts/validate_local.py`
without installing dependencies. CI runs the same validation sequence on
Python 3.11 and 3.12. Fake teacher export artifacts remain generated local state
under the gitignored `artifacts/` directory.

## Phase 4 — RWKV7 Reference Recurrent Core

This phase adds `rwkv7_reference` as the first RWKV7-style recurrent core for
the JAX student path. It is an XLA-friendly recurrent reference implementation
for shape contracts, masking behavior, JIT compatibility, gradient flow, and
student smoke training. It is not the final optimized RWKV7 kernel.

The validation path now includes both the tiny student smoke command and the
`rwkv7_reference` smoke command after fake teacher target export and inspection.
This preserves the smallest trainer test double while adding recurrent
reference coverage for Phase 4.

## Phase 5 — Distillation Stage Runtime

This phase adds the first configured distillation runtime while keeping the
dependency boundary CPU-only and JAX-first. Stage YAML now loads into typed
dataclasses, losses are registered and composed with explicit weights, and the
existing smoke training path accepts a distillation objective rather than being
replaced by a separate trainer.

The runnable stage performs hidden-state MSE distillation over target bundles,
records metric history summaries, and exposes `scripts/run_distill_stage.py`
for local and CI smoke validation. Logits KL is present as opt-in plumbing with
clear validation, but actual student logits remain deferred until a student
head is added.

## Phase 6 — XLA Discipline and TPU Smoke Readiness

This phase prepares QRWKV-XLA for real TPU smoke testing without requiring TPU
in CI. The project now has JAX runtime inspection, XLA/static-shape smoke
helpers, TPU-ready distillation smoke scripts, and documentation for running
the repo on Kaggle/Colab-style TPU environments. TPU hard-fail behavior is
opt-in through `--require-tpu`.

## Phase 7 — Optional Hugging Face Teacher Export Backend

This phase adds a real teacher-export backend boundary without changing the
default validation surface. PyTorch and Hugging Face Transformers remain
optional through the `teacher-hf` extra; the fake exporter remains the default
for CI and local validation. The HF backend lazily imports its dependencies,
uses fixed-length prompt tokenization, exports hidden states and optional logits
through the existing target bundle writer, and records model-derived hidden
shape metadata in the manifest.

Qwen smoke execution, network-dependent model downloads, GPU assumptions, and
generated `artifacts/` outputs remain outside default validation.

## Phase 8 — Offline Qwen Policy Prep

This phase adds a local Qwen policy resolver and dry-run export preparation
layer above the optional HF exporter. `Qwen3.latest` is treated as a local YAML
label, not a web lookup, and the default policy intentionally leaves concrete
Qwen model ids unresolved.

Default validation now checks policy parsing and Qwen dry-run behavior without
installing `teacher-hf`, importing torch/transformers, loading a model, writing
a Qwen bundle, or touching the network. Real Qwen export remains manual-only.

## Phase 9 — Canonical Pipeline Validation Harness

This phase adds `qrwkv_xla.validation.pipeline` and
`scripts/validate_pipeline.py` as the canonical end-to-end validation surface.
The default command remains CPU-safe, offline, and available under `.[dev]`: it
prints environment/runtime info, runs CPU and TPU-safe smokes, checks unresolved
Qwen policy dry-runs, exports and inspects fake targets, smokes both
`tiny_student` and `rwkv7_reference`, runs the distillation stage, and runs the
TPU-ready distillation smoke without requiring TPU.

Optional tiny HF validation is available only through `--include-hf`. Hard TPU
validation is available only through `--require-tpu`. CI and
`scripts/validate_local.py` use the default pipeline path and do not imply
either optional mode.

## Phase 10 — Checkpoint/Resume + Staged Continuation

This phase adds local checkpointing for distillation stages using a JSON
manifest plus NumPy NPZ parameter archive. Checkpoints are written only under
the gitignored `checkpoints/` directory and remain CPU-safe and offline by
default.

The distill runner can now save a final checkpoint and resume from an existing
checkpoint. Resume validates the student architecture plus `vocab_size`,
`hidden_size`, and `num_layers` before training. On resume, `max_steps` means
additional steps for the current invocation, so a checkpoint at step N resumed
with M steps ends at N + M.

Orbax remains deferred. The current requirement is inspectable single-process
staged continuation for hidden-state distillation. A richer checkpoint manager
can be revisited after optimizer state, multi-device training, and release
artifact requirements are concrete.

The staged plan is hidden-only continuation first, followed later by a logits
continuation phase once students emit logits and logits targets are enabled.
## P11 - Local Run Tracking

P11 adds opt-in local run tracking for distillation runs. Tracking writes
`run.json`, `metrics.jsonl`, and `summary.json` under `runs/<run_id>/`; when no
checkpoint output is configured, the final checkpoint defaults to
`runs/<run_id>/checkpoints/final`. The feature is disabled by default and uses
only durable local files.

The canonical CLI flags are `--track-run`, `--run-root`, `--run-name`,
repeatable `--run-tag`, repeatable `--run-note`, and `--run-overwrite`.
Metadata capture for git and JAX runtime state is best effort and must not make
training fail when unavailable.

## P12 - Prompt Corpora

P12 adds file-based prompt corpora as JSONL, one object per line, with canonical
split labels `train`, `validation`, `test`, and `unspecified`. Corpus manifests
record ordered SHA-256 hashes, record counts, split counts, and tag counts.
Splitting is deterministic by seed and assigns at least one validation example
for small multi-record corpora when validation is requested.

Teacher export now accepts `targets.prompt_corpus` with optional split, tag, and
limit filters. Inline/file prompts remain supported, but corpus prompts are
mutually exclusive with those sources. Target manifests record prompt
provenance metadata without storing full prompt texts.

Default validation inspects the smoke corpus, creates its manifest, and dry-runs
the Qwen corpus config without importing Hugging Face modules or requiring
`teacher-hf`. The HF corpus export remains opt-in through `--include-hf`.

## Phase 13 — Student LM Head and Logits KL Continuation

This phase gives QRWKV-XLA students an optional LM head so they can emit logits
in addition to hidden states. The distillation runtime can train with logits KL
when teacher logits are available, enabling a staged path from hidden-only
alignment into output-behavior distillation. Fake logits smoke configs keep the
default validation path CPU-only and network-free.
