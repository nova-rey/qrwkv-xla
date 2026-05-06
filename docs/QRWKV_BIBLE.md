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

## Phase 14 — Generation Smoke and Tiny Evaluation Harness

This phase adds the first inference-facing path. QRWKV-XLA can now load a
logits-capable student checkpoint, encode smoke prompts with a dependency-free
tokenizer, run short greedy generation, and write generation artifacts. This is
a wiring and sanity check rather than a model-quality benchmark.

## Phase 15 — Evaluation Harness and Fixed Regression Prompts

This phase adds the first repeatable generation evaluation layer. QRWKV-XLA can
evaluate logits-capable checkpoints on fixed prompt corpora, write generation
snapshots, run simple sanity checks, and compare snapshots across checkpoints or
runs. These checks are regression and wiring tools rather than model-quality
benchmarks.

## Phase 16 — Adam and AdamW Optimizers

This phase adds an internal optimizer package with SGD, Adam, and AdamW. SGD
remains the default smoke path. Adam uses standard bias-corrected moments, and
AdamW uses decoupled weight decay applied to parameter leaves rather than as
gradient L2 regularization.

Distillation configs and CLI flags can select the optimizer and hyperparameters.
Checkpoints now persist optimizer config and optimizer state in the existing
JSON + NPZ format so Adam/AdamW resumes continue their moment slots. Default CI
stays CPU-only, offline, and dependency-light.

## Phase 17 — Learning Rate Scheduling

This phase adds learning rate schedules to the distillation runtime. QRWKV-XLA
now supports constant and warmup-cosine schedules, uses resume-aware global step
counting, records scheduled learning rates in metrics, and stores schedule
metadata in checkpoints and tracked runs.

## Phase 18 — Gradient Clipping

This phase adds simple global gradient norm clipping to the distillation train
step. The runner computes pre-clip norm, applies an optional global scale before
the optimizer update, and logs post-clip norm, scale, clipped flag, and max norm
per step. Checkpoints and tracked runs record gradient config metadata, and the
validation pipeline includes a clipped AdamW smoke while preserving unclipped
paths.

## Phase 19 — Stage 3 Cross-Entropy Fine-Tuning

This phase adds the student-only Stage 3 language-model path. The `qrwkv_xla.lm`
package reads prompt corpora, tokenizes with `SmokeTokenizer`, builds static
next-token batches, and trains logits-capable students with masked CE. It does
not require teacher hidden states, teacher logits, or target bundles.

Stage 3 reuses the existing optimizer, learning-rate schedule, gradient
clipping, simple checkpoint, and local tracking layers. Checkpoints record
`next_token_ce` loss metadata and prompt-corpus provenance while keeping the
default validation path CPU-only, offline, and dependency-light.
## Phase 20 — Stage 1 Attention/Mixer Target Distillation

This phase implements the missing Stage 1 conversion path. QRWKV-XLA can now
export attention/mixer target vectors, expose recurrent student mixer outputs,
and train them with layerwise MSE before hidden-state, logits, or Stage 3 CE
training. The default validation path uses fake attention targets, while real
HF/Qwen attention capture remains manual-only.

## Phase 21 — Multi-Device TPU Sharding Smoke

This phase adds QRWKV-XLA’s first multi-device training path. Parameters and
optimizer state are replicated across devices, batches are sharded across the
leading axis, gradients and metrics are averaged with `pmean`, and checkpoints
are saved from unreplicated state. The goal is data-parallel smoke validation,
not full model-parallel scaling.

## Phase 22 — Real Qwen Tokenizer Integration

This phase adds a tokenizer abstraction and registry while preserving
`SmokeTokenizer` as the default offline and CI-safe backend. `smoke` remains the
dependency-free byte tokenizer; `hf` is an optional Hugging Face tokenizer
wrapper loaded only when requested; `qwen` is a registry alias for `hf`.

Stage 3 LM data loading now goes through the registry. LM configs accept either
`tokenizer: smoke` or a mapping with backend, tokenizer ID, vocab, EOS, PAD,
revision, and trust-remote-code fields. The LM runner uses tokenizer metadata
for EOS appending, padding masks, and student vocab compatibility validation.
Real HF tokenizer coverage is env-gated and skipped by default so the baseline
test path stays CPU-only, offline, and dependency-light.

## Phase 23 — Real Tokenized Data Pipeline

This phase adds the first reusable tokenized-corpus artifact for Stage 3 CE
training. Prompt JSONL can now be tokenized once into `manifest.json` plus
`shards/*.npz`, with deterministic concat-pack semantics, tokenizer provenance,
per-shard hashes, and static Stage 3 arrays (`input_ids`, `labels`,
`attention_mask`, `loss_mask`).

The Stage 3 LM path now accepts either raw prompt JSONL or a tokenized corpus
artifact without making HF, network access, GPU, or TPU mandatory. Default CI
stays on the offline `SmokeTokenizer` path while the validation pipeline gains a
small tokenized-corpus smoke.

## Phase 24 — RWKV7 Math Parity Audit

This phase audits the current `rwkv7_reference` student against the local
RADLADS RWKV7 reference checkout. The result is explicit: the current JAX
student is a simplified placeholder, not RADLADS math parity. It keeps a
scalar-channel `[B, H]` recurrent state and simple sigmoid decay, while RADLADS
uses head-wise matrix WKV state, `exp(-exp(.))` decay semantics, `a`/`b`
in-context state updates, normalized key terms, Qwen projection structure,
RoPE, and cache semantics.

P24 therefore preserves the existing CPU-only smoke role instead of pretending
checkpoint compatibility exists. The layer now exposes an optional initial
state and returned final state so local recurrence invariants can be tested.
A pure NumPy harness mirrors the current placeholder recurrence, and focused
tests cover all-at-once vs token-by-token state equivalence, batched vs
unbatched equivalence, eager vs JIT equivalence, finite gradients, and a tiny
optimizer-step no-NaN check.

## Phase 25 — Tiny Real Teacher Target Export Proof

This phase hardens the teacher target export artifact path. The fake exporter
remains the default offline test double, while the HF exporter can produce
target bundles from prompt text, prompt corpora, or tokenized corpus artifacts
without making torch or transformers part of the student training path.

Target bundle shards now carry `input_ids`, `attention_mask`, and `loss_mask`
as required arrays, plus optional hidden states, logits, and attention targets.
Manifests record per-shard relative paths, deterministic ordered SHA-256
hashes, example counts, array names, prompt/tokenized-corpus provenance, and HF
model provenance including revision, dtype, trust-remote-code, and
local-files-only. Bundle loading validates the manifest, shard shapes,
sequence length, hashes, array names, and example counts before returning
arrays to consumers.

## Phase 26 — Tiny Real Teacher-to-Student Distillation Proof

This phase connects the P25 teacher target bundle artifact to the existing JAX
student distillation runner. The proof remains tiny, CPU-safe, and offline by
default: fake teacher targets are exported through the normal target-bundle
writer, then `scripts/run_distill_stage.py` trains a student with hidden-state
MSE, writes metrics, writes a checkpoint, reloads that checkpoint, and resumes
for another step.

The target-training path now carries `loss_mask` from target bundles into JAX
batches and uses it for token-level hidden-state MSE, logits KL, and
attention/mixer MSE averaging. `attention_mask` remains the model input mask,
while `loss_mask` controls which target positions contribute to averaged
distillation losses.

The HF teacher exporter now resolves `teacher.device: auto` to `cuda`, `mps`, or
`cpu` before moving the model or encoded batches. The default remains `cpu`, and
real HF execution stays optional and env-gated so CI does not require torch,
transformers, network access, CUDA, or MPS.

Tiny local proof commands:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub.yaml
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --targets artifacts/teacher_targets/fake_export --student-architecture tiny_student --max-steps 1 --checkpoint-out checkpoints/p26_tiny_first --checkpoint-overwrite --track-run --run-root runs/p26 --run-name p26_tiny_first --run-overwrite
python scripts/run_distill_stage.py --config configs/distill_stage0_stub.yaml --targets artifacts/teacher_targets/fake_export --student-architecture tiny_student --max-steps 1 --resume-from checkpoints/p26_tiny_first --checkpoint-out checkpoints/p26_tiny_resume --checkpoint-overwrite --track-run --run-root runs/p26 --run-name p26_tiny_resume --run-overwrite
```

## Phase 27 — RADLADS-Aligned JAX RWKV7 Reference Math

This phase adds `rwkv7_radlads_reference` as a separate student backend. It is
not a replacement for `rwkv7_reference`: the older backend remains the stable
tiny smoke placeholder, while the new backend is a slow, scan-based JAX
reference for the RADLADS RWKV7 recurrence shape.

The new backend is CPU-first and deliberately slow. It uses a head-wise matrix
recurrent state with shape `[B, H, N, N]`, simple JAX projections, and a
`jax.lax.scan` recurrence that tracks the broad RADLADS update pattern: decay,
outer-product value/key injection, in-context update terms, and recurrent-state
readout through receptance/query-like projections. Padding masks zero the value
stream while still running the recurrence update, matching the RADLADS
left-padding behavior more closely than freezing state on masked tokens.

The implementation is still explicitly partial. It does **not** claim full
RADLADS parity, Qwen block parity, RoPE parity, Triton kernel parity, grouped
KV parity, or checkpoint weight compatibility. Those gaps are recorded in
`docs/RWKV7_REFERENCE_AUDIT.md` so later kernel work can optimize against an
honest reference instead of a vague placeholder.

Tiny local proof command:

```bash
python scripts/run_distill_stage.py --config configs/distill_stage0_radlads_reference_stub.yaml --targets artifacts/teacher_targets/fake_export --max-steps 1 --learning-rate 0.01
```

## Phase 28 — RADLADS Reference Backend Validation + Teacher Export Path Cleanup

This phase validates `rwkv7_radlads_reference` through the actual distillation
trainer checkpoint boundary: the trainer writes a JSON + NPZ checkpoint, resumes
from it, preserves the global step, and records the RADLADS backend student
config. This is a RADLADS-shaped JAX reference backend, not full RADLADS parity.
The older `rwkv7_reference` placeholder backend remains unchanged and continues
to serve its smoke/offline role.

Teacher export config path handling is now consistent. Relative config paths
for prompt files, prompt corpora, tokenized corpora, Qwen policy files, and
`runtime.output_dir` resolve relative to the YAML file that declares them.
Command-line overrides such as `--out`, `--prompt-file`, `--prompt-corpus`,
`--tokenized-corpus`, and `--qwen-policy` keep normal CLI semantics and remain
relative to the current working directory unless absolute paths are provided.

The proof is intentionally narrow: no TPU, pjit, Pallas, kernel, quality,
checkpoint parity, or numerical parity claim is added.

## Phase 29 — Tiny HF Teacher Export to RADLADS Reference Resume Smoke

This phase adds a checked-in tiny real-HF teacher smoke path using
`sshleifer/tiny-gpt2`, `corpora/hf_tiny_smoke_prompts.jsonl`, and
`configs/teacher_export_tiny_hf_smoke.yaml`. The export config is CPU-oriented,
uses config-relative paths, and writes under
`artifacts/teacher_targets/tiny_hf_smoke` when run from the repository root.

The paired distill config is
`configs/distill_stage0_radlads_reference_tiny_hf.yaml`. It selects
`rwkv7_radlads_reference` with tiny-GPT2-compatible dimensions inferred from
the target bundle: hidden size 2, two layers, one head, and vocab size 50257.
The smoke is hidden-MSE only. The HF target bundle includes logits so the real
target key contract is visible, but `logits_kl` is disabled and the student is
configured with `emit_logits: false`, so logits are explicitly not consumed.

Default tests remain offline-safe. They use fake/minimal HF-shaped fixtures to
check config loading, target keys, tiny dimensions, checkpoint save, resume,
and finite resumed loss. The live HF export-to-distill proof is still opt-in
behind `QRWKV_RUN_HF_INTEGRATION=1` and optional `teacher-hf` dependencies.
This remains a RADLADS-shaped JAX reference backend, not full RADLADS parity.

## P30 — Real HF Target Loss Hardening + Checked-In Smoke

This phase keeps the P29 hidden-only path intact and adds
`configs/distill_stage0_radlads_reference_tiny_hf_logits.yaml` for the tiny HF
target bundle. The new config selects `rwkv7_radlads_reference` with
`emit_logits: true` and enables a combined hidden-MSE plus `logits_kl` loss, so
teacher logits from `sshleifer/tiny-gpt2` can be consumed by the RADLADS-shaped
reference student in the opt-in live smoke.

Default validation remains offline-safe. Focused tests cover logits config
parsing, hidden-only config parsing, clear failure when logits KL is requested
without teacher logits, `loss_mask` handling for logits KL, and a tiny fake
logits run through `rwkv7_radlads_reference`. The real HF export-to-logits
distill-to-resume path is still gated by `QRWKV_RUN_HF_INTEGRATION=1` and the
optional HF dependencies.

P30 proves target-loss plumbing with tiny real HF targets; it does not prove model quality or RADLADS parity. `rwkv7_radlads_reference` remains a RADLADS-shaped JAX reference backend, not full RADLADS parity.

## Phase 31 — RADLADS RWKV7 Gap Audit

This phase is an audit and planning checkpoint. It inspects the current
QRWKV-XLA `rwkv7_radlads_reference` backend, the older `rwkv7_reference`
placeholder, the distillation and target-bundle plumbing, and the local
RADLADS RWKV7Qwen2 source files.

The audit result is blunt: QRWKV-XLA has useful recurrence-shaped JAX plumbing,
but it does not yet have RADLADS model parity. The current backend carries
head-wise matrix state shaped `[layers,batch,heads,head_size,head_size]`,
projects `wr/ww/wk/wv/wa/wb/wg/wo` plus `time_bias`, applies RADLADS-style
decay and matrix-state update shape, and proves hidden/logit loss plumbing. It
does not implement the Qwen decoder block shell, RoPE, grouped KV heads,
shift-state cache, RADLADS LoRA-rank parameterization, parameter mapping, or
numerical comparison against RADLADS PyTorch/Triton/CUDA outputs.

The wording remains: RADLADS-shaped JAX reference backend, not full RADLADS
parity.

The recommended P32 scope is math-completion and Qwen block compatibility before
kernel work: add a slow JAX block-compatible target with RMSNorm, RWKV7
attention, residuals, MLP, RoPE, grouped KV shape semantics, KV plus shift state
contract, and RADLADS-named parameter mapping stubs. Pallas, TPU kernels,
`pjit`, sharding, export hardening, and quality work remain deferred until the
slow reference can catch math and cache regressions.

## Phase 32 — Qwen-Compatible RADLADS Slow Reference Backend

This phase adds `rwkv7_qwen_reference`, a selectable backend separate from
`rwkv7_radlads_reference` and the older `rwkv7_reference` placeholder. The
required scope statement is: Qwen/RADLADS-compatible slow JAX reference path,
not optimized kernel parity.

The new backend implements the highest-priority block-compatibility pieces in
small CPU-safe JAX code: a Qwen decoder shell
input -> RMSNorm -> RWKV/time-mix/reference attention -> residual -> RMSNorm
-> MLP -> residual, slow deterministic RoPE, grouped KV head expansion,
validated `num_heads`/`num_kv_heads`/`head_size`, explicit pytree state with
`wkv_matrix_state`, `shift_state`, and `next_position`, and a nested parameter
surface using Qwen/RADLADS-oriented names.

The tiny config `configs/distill_stage0_qwen_reference_tiny_hf.yaml` targets
`artifacts/teacher_targets/tiny_hf_smoke` with hidden-only loss and tiny-GPT2
dimensions inferred from the target bundle. Default behavior remains
offline-safe; live HF tests are not made mandatory.

P32 establishes the reference target that later Pallas/XLA kernels must match.
It does not add optimized kernels, and it does not claim full RADLADS numerical
parity, checkpoint compatibility, TPU readiness, export compatibility, large
training readiness, or model quality.

## Phase 33 — Tiny Qwen Reference Parity Fixture Harness

This phase adds deterministic tiny fixtures for `rwkv7_qwen_reference` without
changing backend semantics. The generator is
`scripts/generate_qwen_reference_fixtures.py`; the checked-in bundle is
`tests/fixtures/qwen_reference/`, and the same script can write local smoke
artifacts under `artifacts/p33_qwen_reference_fixtures`.

The fixture config is deliberately small: vocab size 32, hidden size 8, two
layers, two attention heads, one KV head, batch size 2, sequence length 5, seed
1234, `float32`, logits enabled, and mixer outputs enabled. Each case records
full-sequence hidden outputs, logits, mixer outputs, final KV matrix state,
shift state, and `next_position`, plus the matching stepwise outputs/logits and
final state. The manifest records backend/config/seed/dtype/shapes, payload
hashes, parameter-surface hashes, and full-vs-step max absolute/relative
errors.

The mask fixtures cover no mask, interior masks, and a prefix/left-padding
shape case. P33 documents current masked-token behavior instead of changing it:
masked tokens zero the value stream, attention output, and MLP output for that
token, while recurrence decay/update from previous matrix state still runs,
`shift_state` updates to the token representation, and `next_position` advances
through masked positions. The prefix/left-padding case is a shape/current
behavior fixture, not a claim of separate left-padding position semantics.

P33 also adds `docs/RWKV7_QWEN_REFERENCE_PARAM_SURFACE.md`, which snapshots the
tiny flattened parameter paths and shapes. This is local fixture/caliper work,
not RADLADS PyTorch/Triton/CUDA parity, not checkpoint compatibility, and not
kernel readiness.

## Phase 34 — Model Scale Planner and Config Generator

This phase adds `qrwkv_xla.scale_planner`, a conservative planning package for
estimating QRWKV-XLA model scale before attempting larger training runs. It adds
validated dataclass profiles for model shapes, hardware budgets, and training
modes; built-in profiles cover tiny debug shapes, CPU/local probes, Colab TPU
smokes, approximate Qwen-scale candidates, local CPUs, single GPUs, TPU
placeholders, and hidden/logits distillation modes.

The planner estimates the current `rwkv7_qwen_reference` parameter surface:
token embeddings, Qwen/RADLADS-oriented attention projections, time parameters,
RMSNorm weights, gated MLP weights, and optional LM head parameters. Memory is
reported by component: weights, gradients, optimizer moments, optional fp32
master weights, activations, recurrent WKV/shift state, hidden targets,
full-vocab logits targets, input ids/masks, checkpoint reference size, and
overhead reserve. The full-logits target component is explicit and warns when
it dominates.

The CLI is `scripts/plan_model_scale.py`. It prints a readable summary and can
write YAML or JSON plans. Auto mode tries conservative batch and sequence
reductions without changing architecture, and may recommend disabling logits KL
when full-vocab logits targets dominate memory. It also emits a planning-only
distill config skeleton marked as not hardware validated.

P34 is not TPU profiling, not pjit/model sharding, not a Qwen-scale training
run, not model export, and not a RADLADS parity claim. Fit classes are
conservative estimates using roughly 60% for `yes`, 85% for `maybe`, and above
85% for `no`.
