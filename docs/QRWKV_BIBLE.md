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

## Phase 35 — Parameter Compatibility Bridge Status

No P35 implementation is present in this repo state. The RADLADS checkpoint
import/export compatibility bridge remains future work. P36 must not be read as
evidence of RADLADS checkpoint compatibility; it only exercises the native
QRWKV-XLA checkpoint/resume path for a tiny reference student.

## Phase 36 — Colab TPU Smoke Harness Hardening

This phase adds a manual, opt-in Colab TPU smoke harness for the tiny
hidden-only `rwkv7_qwen_reference` path. The checked-in config is
`configs/distill_stage0_qwen_reference_colab_tpu_smoke.yaml`, and the launcher
is `scripts/run_colab_tpu_smoke.py`.

The harness prints Python, JAX, backend, device, and git metadata; requires the
JAX default backend to be TPU; runs a tiny JAX matmul; exports deterministic
fake hidden teacher targets; runs one distill step to
`checkpoints/p36_tpu_qwen_reference_first`; then resumes for one more step to
`checkpoints/p36_tpu_qwen_reference_resume`. It normalizes run artifacts to
stable paths under `runs/p36/`, validates required checkpoint/run files,
validates finite `final_loss` and `final_hidden_mse`, and checks checkpoint
step progression from 1 to 2.

Successful manual runs write
`artifacts/p36_colab_tpu_smoke/P36_RESULTS.md` and
`artifacts/p36_colab_tpu_smoke/p36_results_bundle.tar.gz`. The Colab copy/paste
flow and known warnings live in `docs/COLAB_TPU_SMOKE.md`.

P36 remains deliberately narrow. It does not add Pallas/WKV7 optimized kernels,
`pjit`, sharding, multi-host TPU support, real Qwen-scale training, real HF
teacher export on TPU, logits KL on TPU, quality evaluation, WandB, or HF
student export. Normal local and CI validation stays CPU-only.

## Phase 37 — Colab TPU Logits-KL Smoke Harness

This phase adds a second manual, opt-in Colab TPU smoke harness for the tiny
`rwkv7_qwen_reference` path with logits enabled. The hidden-only P36 command
remains unchanged as `python scripts/run_colab_tpu_smoke.py`; P37 adds
`scripts/run_colab_tpu_logits_smoke.py`.

The checked-in distill config is
`configs/distill_stage0_qwen_reference_colab_tpu_logits_smoke.yaml`. It points
at deterministic fake logits-bearing targets under
`artifacts/teacher_targets/p37_colab_tpu_logits_smoke`, sets
`student.emit_logits: true`, and enables both `hidden_mse` and `logits_kl`.
The companion fake teacher export config is
`configs/teacher_export_qwen_reference_colab_tpu_logits_smoke.yaml`.

The shared Colab TPU smoke helpers now validate the TPU backend message,
required target logits in the manifest and shard, required checkpoint/run
artifacts, finite `loss`, `hidden_mse`, and `logits_kl`, and optimizer step
progression from 1 to 2 across checkpoint resume. Successful manual runs write
`artifacts/p37_colab_tpu_logits_smoke/P37_RESULTS.md` and
`artifacts/p37_colab_tpu_logits_smoke/p37_results_bundle.tar.gz`.

P37 remains deliberately narrow. It does not add real Qwen-scale training, real
HF teacher export on TPU, Pallas/WKV7 optimized kernels, `pjit`, sharding,
multi-host TPU training, quality evaluation, WandB, HF student export, or
RADLADS parameter import/export compatibility.

## Phase 38 — Real Tiny HF Teacher Targets to TPU Distill Smoke

This phase adds a third manual, opt-in TPU smoke. P38 keeps the P36 hidden-only
and P37 fake-logits commands unchanged, and adds
`scripts/run_tiny_hf_tpu_smoke.py`.

The checked-in HF export config is
`configs/teacher_export_p38_tiny_hf_logits_smoke.yaml`. It uses the existing
HF teacher exporter path with `sshleifer/tiny-gpt2`, `include_logits: true`,
sequence length 8, and stable output directory
`artifacts/teacher_targets/p38_tiny_hf_logits_smoke`. The distill config is
`configs/distill_stage0_qwen_reference_p38_tiny_hf_tpu_smoke.yaml`; it uses the
tiny `rwkv7_qwen_reference` student with logits enabled and combines
`hidden_mse` plus `logits_kl`.

The shared Colab TPU smoke helpers now support fake-target P36/P37 specs and
the real-HF P38 spec. P38 exports the HF target bundle, validates that the
manifest and first shard contain `input_ids`, `attention_mask`, `loss_mask`,
`hidden_states`, and `logits`, checks basic target dimensions against the
manifest, runs one TPU distill step, resumes for one more step, and validates
finite `loss`, `hidden_mse`, `logits_kl`, and optimizer/checkpoint progression
from 1 to 2. Successful manual runs write
`artifacts/p38_tiny_hf_tpu_smoke/P38_RESULTS.md` and
`artifacts/p38_tiny_hf_tpu_smoke/p38_results_bundle.tar.gz`.

P38 is documented for both Colab and Kaggle TPU sessions in
`docs/COLAB_TPU_TINY_HF_SMOKE.md`. It remains deliberately narrow: no
Qwen-scale teacher export or training, no multi-host TPU or `pjit` sharding, no
Pallas/WKV7 optimized kernels, no model-quality claim, no WandB or lm-eval, no
HF student export, and no full RADLADS numerical parity claim.

## Phase 39 — Planner-Generated Tiny HF TPU Smoke

This phase connects the scale planner to the real tiny HF TPU smoke path. It
adds the RoPE-valid planner profile `p39_tiny_hf_qwen_rope_smoke`, shaped for
`sshleifer/tiny-gpt2` targets and the current `rwkv7_qwen_reference` backend:
vocab size 50257, hidden size 2, two layers, one query head, one KV head, and
head size 2. It also adds the `kaggle_tpu_v5e_8` hardware profile as an
aggregate TPU v5e planning budget.

The generated planner outputs live under
`artifacts/p39_planner_tpu_smoke/`: `scale_plan.yaml`, `scale_plan.json`,
`generated_distill.yaml`, and `teacher_export.yaml`. The teacher export config
is generated for P39 rather than reused from P38; it still uses the same real
tiny HF model family, `sshleifer/tiny-gpt2`, and writes targets to
`artifacts/teacher_targets/p39_tiny_hf_logits_smoke`.

The manual launcher is `scripts/run_planner_tpu_smoke.py`. It prints runtime
metadata, requires the existing exact TPU backend check, runs a tiny JAX
matmul, regenerates the P39 planner artifacts, validates that the tiny plan fit
is `yes` or `maybe`, exports real HF targets with hidden states and logits,
validates required target keys, runs one TPU distill step, resumes for one more
step, validates finite `loss`, `hidden_mse`, and `logits_kl`, and checks
optimizer/checkpoint step progression from 1 to 2. Successful manual runs write
`artifacts/p39_planner_tpu_smoke/P39_RESULTS.md` and
`artifacts/p39_planner_tpu_smoke/p39_results_bundle.tar.gz`.

P39 is documented in `docs/COLAB_TPU_PLANNER_SMOKE.md` and
`docs/KAGGLE_TPU_PLANNER_SMOKE.md`. Planner fit remains planning-only except
for the tiny validated execution path. P39 does not add Qwen-scale or long
training, pjit/sharding/multi-host TPU, Pallas kernels, lm-eval, WandB, or HF
student export.

## Phase 40 — RADLADS Source Parity Fixture Bridge

This phase starts the explicit RADLADS source parity track. It does not claim
full numerical equivalence. Instead, it adds a canonical fixture schema,
checked-in deterministic tiny cases, import/report tooling, and an honest
parameter-surface map so later work can measure what is comparable and what is
still unsupported.

The canonical checked-in fixture set lives under
`tests/fixtures/radlads_source_parity`. Those payloads are intentionally marked
`unsupported`: they record QRWKV-XLA current behavior only and are not
fabricated RADLADS outputs. Real source-produced fixtures can be copied into the
same schema with `scripts/import_radlads_source_fixtures.py`, while
`scripts/compare_radlads_source_fixtures.py` and
`scripts/map_radlads_parameter_surface.py` write reports under
`artifacts/parity/radlads_source_bridge/`.

The three tiny cases are `tiny_no_mask`, `tiny_attention_mask`, and
`tiny_prefix_padding_or_left_padding`. The left-padding/prefix case remains an
explicitly unsupported/current-behavior-only bridge point until a fair
source-to-QRWKV comparison is available.

The initial parameter map records direct Qwen/RADLADS-to-QRWKV role matches for
embedding, RMSNorm, Q/K/V/O projections, MLP, and LM head surfaces, and marks
known gaps such as `w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`, gate variants,
`k_k`/`k_a`/`r_k`, and optional `ln_x` as unsupported. P40 therefore makes
RADLADS compatibility measurable without overclaiming live source parity,
training quality, kernel correctness, checkpoint compatibility, or optimized
Pallas/WKV behavior.

## Phase 41 — QRWKV-XLA Checkpoint to HF/Safetensors Export Smoke

This phase adds a bounded HF-style safetensors export smoke for QRWKV-XLA
student checkpoints. The new `qrwkv_xla.export` package can export an existing
JSON + NPZ checkpoint to a directory containing `config.json`,
`model.safetensors`, `qrwkv_xla_export.json`, and `weight_map.json`, then reload
that directory back into QRWKV-XLA helper objects.

The CLI entrypoints are `scripts/export_student_hf_safetensors.py` for direct
checkpoint export and `scripts/run_export_smoke.py` for the end-to-end smoke.
The smoke creates a deterministic tiny logits-capable checkpoint under its
artifact directory, exports it, reloads it, compares hidden states and logits on
a fixed CPU batch, and writes `export_smoke_report.json` plus
`P41_EXPORT_SMOKE_REPORT.md`.

The export helper lazy-imports `safetensors`; if unavailable, it fails with the
explicit install message documented in `docs/HF_SAFETENSORS_EXPORT.md`. P41 is
intentionally narrow: it proves only tiny local checkpoint export/reload parity.
It does not add a production Hugging Face model class, Qwen-scale export,
sharded or `pjit` export, `lm_eval`, Pallas/WKV optimized kernels, or model
quality claims.

## Phase 42 — QRWKV-XLA lm_eval Toy Exported-Student Integration

This phase adds a bounded lm_eval-style smoke around the P41 exported-student
artifact path. The new `qrwkv_xla.eval.exported_student` adapter loads a P41
HF/safetensors export with `load_hf_safetensors_export`, runs logits-capable
student inference locally, and scores deterministic continuation
loglikelihoods from tiny token-id fixtures under `tests/fixtures/eval/`.

The smoke entrypoint is `scripts/run_lm_eval_smoke.py`. It uses the implemented
P41 export path `artifacts/p41_hf_safetensors_export_smoke`; if that directory
does not exist, it generates it through the P41 smoke, while partial exports
fail with explicit missing-file errors. Successful runs write
`artifacts/eval/p42_lm_eval_smoke/results.json`,
`artifacts/eval/p42_lm_eval_smoke/P42_RESULTS.md`, and
`artifacts/eval/p42_lm_eval_smoke/p42_results_bundle.tar.gz`.

P42 deliberately delivers an lm_eval-style toy harness rather than official
`lm_eval` execution. The optional `eval` extra records the dependency path for
manual future integration, but the default smoke does not import official
`lm_eval`, does not touch the network, and does not run large task suites. This
phase makes no benchmark, Qwen-scale, production HF model class, training,
pjit/sharding, Pallas/WKV kernel, or model-quality claim.

## Phase 43 — QRWKV-XLA WKV7 / Pallas Correctness Fixture Harness

This phase adds the deterministic correctness cage for future optimized WKV7
kernel work. The new `qrwkv_xla.kernels` package generates tiny offline fixtures
for the extracted WKV7 recurrence/state core, records manifest metadata and
hashes, compares candidate implementations against the saved expected outputs
and next-state tensors, and reports explicit statuses for pass/fail,
unsupported, missing fixture payloads, shape/dtype mismatches, non-finite
values, and candidate execution errors.

The canonical artifact directory is `artifacts/kernels/p43_wkv7_correctness`.
It contains `manifest.json`, `P43_WKV7_FIXTURE_SUMMARY.md`,
`comparison_report.json`, `P43_WKV7_COMPARISON_REPORT.md`, and per-case
`inputs.npz` / `expected.npz` payloads. The fixture set covers six deterministic
cases including no-mask, masked, prefix-padding/reset-style, explicit non-zero
state, stepwise-vs-full-scan, and extreme-but-finite decay scenarios.

P43 proves only correctness-harness plumbing for the recurrence core. It does
not implement an optimized Pallas kernel, does not benchmark TPU performance,
does not prove full RADLADS numerical parity, and does not claim training or
model-quality improvements. Its job is to provide the gate future WKV7 kernels
must pass before speed work is trusted.

## Phase 44 — QRWKV-XLA Streaming Data Pipeline Dry-Run

This phase adds a larger/local/offline streaming data pipeline dry-run. The new
`qrwkv_xla.data` package defines a manifest-backed streaming dataset, shard
metadata with tokenized-corpus provenance, a bounded batch iterator,
deterministic order, optional shuffle plus seed, resume cursors, and explicit
attention/loss-mask validation.

The default builder intentionally reuses the existing tokenized corpus surface:
`scripts/build_streaming_data_dry_run.py` writes deterministic synthetic prompts,
packs them with `qrwkv_xla.lm.tokenized_corpus`, then converts the validated
shards into a root artifact set under `artifacts/data/p44_streaming_dry_run/`
with `manifest.json`, `shards/*.npz`, and `P44_DATASET_SUMMARY.md`. Streaming
batches keep the same keys expected by LM/trainer code: `input_ids`, `labels`,
`attention_mask`, and `label_mask`.

`scripts/run_streaming_data_dry_run.py` writes
`streaming_dry_run_report.json`, `P44_STREAMING_DRY_RUN_REPORT.md`, and
`resume_cursor.json` with token availability, consumption,
padding/discard accounting, exact integer post-resume replay checks,
deterministic replay status, mask validation status, approximate tokens per
second, and peak memory when the platform exposes it.
`scripts/run_streaming_trainer_dry_run.py` runs a tiny CPU-only train-step
ingestion check over streaming batches using the existing trainer batch
contract and writes `trainer_dry_run_report.json` plus
`P44_TRAINER_DRY_RUN_REPORT.md`.

P44 is not a real training phase. It proves larger/streaming dry-run plumbing
only. It does not prove full-scale throughput, real training quality, cloud or
distributed ingest, `pjit`/sharding, Pallas, WandB, or Qwen0.5B-scale target
generation.

## Phase 45 — QRWKV-XLA Multi-Scale Model Config Dry-Runs

This phase adds a bounded multi-scale planning and metadata dry-run surface for
explicit QRWKV student profiles: `qrwkv_qwen_0_5b_candidate`,
`qrwkv_qwen_1_5b_candidate`, and `qrwkv_qwen_7b_stretch`. The hardware matrix
covers local CPU debug, Colab/Kaggle TPU planning profiles, and grant TPU v5e
8/32/64-device planning slices.

`scripts/generate_multiscale_configs.py` writes the planning artifacts under
`artifacts/scale/p45_multiscale_dry_run`: `scale_plan_report.json`,
`fit_matrix.json`, `P45_SCALE_PLAN_REPORT.md`, `P45_RESULTS.md`, and
`configs/*.yaml`. The fit matrix records component estimates for parameter
memory, optimizer memory, activation/sequence memory, hidden/logits target
memory, checkpoint memory, overhead reserve, and total memory, with explicit
fit classification and memory interpretation.

`scripts/run_multiscale_shape_dry_run.py` consumes the scale plan and keeps the
default path metadata-only. It writes one `metadata_dry_run.json` per profile,
plus a checkpoint skeleton bundle per profile containing
`checkpoint_manifest.json`, `model_config.yaml`, and
`checkpoint_metadata.json`. The dry-run validates model shapes, parameter bands,
safe init policy, and checkpoint skeleton readback without allocating full large
model arrays.

P45 is not a real training phase. It does not implement or prove pjit/sharding,
distributed execution, Pallas kernels, WandB, full-scale measured memory,
Qwen0.5B teacher target generation, official benchmarks, or one-device 7B
training.

## Phase 46 — Tiny pjit / Sharding Compile Smoke

This phase adds the first explicit sharding-aware compile entrypoint for
QRWKV-XLA. The new `qrwkv_xla.sharding` package creates a named JAX mesh,
records backend/platform/device metadata, exposes a minimal
`data_parallel_single_axis` policy, and runs a tiny forward/loss/update smoke
through either `jax.jit(..., in_shardings=..., out_shardings=...)` or
`jax.experimental.pjit.pjit` when requested.

The canonical artifact directory is `artifacts/p46_pjit_sharding_smoke`. The
smoke writes `P46_RESULTS.md` and `pjit_sharding_smoke_report.json` with
created-at time, compile API, policy, batch/sequence shape, finite loss,
step/update status, mesh metadata, honest single-device fallback details, and
explicit limitations.

P46 proves tiny sharding compile plumbing only. It does not prove large-model
sharding, real training throughput, production sharded checkpointing,
0.5B/1.5B/7B training feasibility, or multi-device TPU behavior unless a real
multi-device TPU run is performed later.

## Phase 47 — Experiment Tracking / WandB Smoke

This phase extends the existing `qrwkv_xla.tracking` package rather than
creating a separate tracking system. The new surface adds an experiment tracker
protocol/config, a local experiment tracker, an optional WandB adapter, report
helpers, and a deterministic tiny tracking smoke.

The canonical local output is `artifacts/p47_experiment_tracking_smoke/`.
It contains `P47_RESULTS.md`, `tracking_smoke_report.json`, and
`local_run/` files for `run_metadata.json`, `config.json`, `metrics.jsonl`,
`summary.json`, `artifacts_manifest.json`, and copied/logged artifacts under
`files/`. The local tracker is the source of truth. WandB is imported only when
`wandb-offline` or `wandb-online` is requested.

The smoke records phase, UTC creation time, repo commit, git dirty
classification, Python/JAX metadata, backend/default backend, device counts,
device kinds/platforms, hostname when available, command/script name, tracking
mode, and artifact root. It writes the exact smoke config and required metrics:
`train/loss`, `train/loss_is_finite`, `train/tokens_seen`,
`train/examples_seen`, and `step`.

P47 does not require WandB credentials or network access for normal development
or CI. It does not prove real Qwen0.5B training, long-running training,
multi-host tracking, production dashboard design, sweeps/MLOps, model quality,
or official benchmark reporting.

## Phase 48 — RADLADS LoRA Rank Math Surface

P48 completes a bounded slow-reference math-surface pass for the
`rwkv7_qwen_reference` backend. It adds explicit config gates for low-rank
RADLADS decay, low-rank ICLR, value residual mixing, balance-state terms,
attention head group norm, and an overall `radlads_compatible_math` mode. The
legacy/default path remains disabled for these branches unless the new flags are
selected.

The new represented parameter leaves are `w0/w1/w2`, `a0/a1/a2`, `v0/v1/v2`,
`k_k/k_a/r_k`, and `ln_x.weight/bias`. The implemented formulas are sourced
from `/home/nyx/.openclaw/workspace/_refs/RADLADS/rwkv7qwen2/modeling_rwkv7qwen2.py`:
low-rank decay uses `w0 + (tanh(xw @ w1) @ w2).float()`, low-rank ICLR uses
`sigmoid(a0 + (xa @ a1) @ a2)`, later layers can mix values against layer-0
`v_first`, balance-state terms use the source `kk/k/k_a/k_k` branches, and
optional attention group norm uses `eps=head_dim * 1e-5`.

`r_k` is represented honestly as a parameter surface only. The inspected
RADLADS source has the residual line using `r_k` commented out, so P48 does not
activate that math or claim it as active parity.

The canonical smoke command is:

```bash
python scripts/run_radlads_lora_rank_math_smoke.py
```

The canonical artifact directory is `artifacts/p48_radlads_lora_rank_math/` and
contains `P48_RESULTS.md`, `lora_rank_math_report.json`,
`P48_PARAMETER_SURFACE_MAP.md`, and `parameter_surface_map.json`.

P48 remains CPU/offline slow-reference work. It does not prove full RADLADS
numerical parity, fitted checkpoint conversion, Pallas or optimized WKV kernels,
TPU performance, Qwen-scale execution, or model quality.

## Phase 49 — RADLADS Tiny Numerical Parity Fixtures

P49 adds bounded infrastructure for tiny real RADLADS numerical fixtures. The
new module `qrwkv_xla.parity.radlads_numerical_fixtures` defines
`radlads_tiny_numerical_parity.v1` manifests, validates NPZ payloads, imports
canonical fixture directories, compares declared arrays, and writes JSON plus
Markdown reports: `P49_RESULTS.md`, `numerical_parity_report.json`,
`P49_SURFACE_COMPARISON.md`, and `surface_comparison.json`. The companion
`radlads_parameter_mapping` module records
minimal mapping statuses: `mapped_exact`, `mapped_renamed`, `shape_mismatch`,
`missing_in_qrwkv`, `missing_in_radlads`, `unsupported`, and
`source_not_found`.

The required tiny cases are `tiny_no_mask`, `tiny_attention_mask`,
`tiny_prefix_or_left_padding`, `tiny_stepwise_state`, and
`tiny_all_radlads_math_enabled`. Reports distinguish per-case `pass`, `fail`,
`unsupported`, `missing_source`, and `fail_known_difference`, and summarize as
`pass`, `pass_with_known_differences`, `fail`, or `source_unavailable`.

The scripts are:

```bash
python scripts/import_radlads_tiny_numerical_fixtures.py
python scripts/generate_radlads_tiny_numerical_fixtures.py
python scripts/compare_radlads_tiny_numerical_fixtures.py
```

Live RADLADS execution is optional and gated by
`QRWKV_XLA_RUN_RADLADS_LIVE_FIXTURES=1` (the older `QRWKV_RUN_RADLADS_LIVE=1`
alias still works).
The generator probes the actual local RADLADS checkout at
`/home/nyx/.openclaw/workspace/_refs/RADLADS`. If live execution is unavailable
or fails, P49 records `source_unavailable` or `execution_failed`; it does not
fabricate RADLADS arrays from QRWKV-XLA.

P49 remains tiny fixture infrastructure only. It does not add training,
checkpoint import, Pallas or optimized WKV kernels, a Hugging Face model class,
Qwen-scale execution, or large-scale RADLADS parity claims.

## Phase 50 — RADLADS Parameter Replay Compatibility

P50 adds an explicit bounded replay path for the real tiny P49 RADLADS
parameter payload. The new importer loads
`artifacts/p49_radlads_numerical_parity/radlads_fixtures/radlads_parameters.npz`,
normalizes RADLADS parameter names/shapes, maps values into
`rwkv7_qwen_reference`, and reports mapped, defaulted, excluded, unsupported,
shape-mismatch, and missing-required surfaces. QRWKV-only surfaces such as
`a_proj.weight`, `b_proj.weight`, `g_proj.weight`, `w_proj.weight`, `time_mix`,
`time_bias`, and `lm_head.bias` are reported and deterministically defaulted
rather than left as random replay inputs.

The slow reference now has explicit replay-gated support for q/k/v projection
biases and the audited RADLADS `gate_rank_type == 2` expression
`sigmoid(xg @ g1) @ g2`. The inspected RADLADS token-shift parameters and `r_k`
residual contribution remain commented out in the local source path, so P50
does not invent those semantics.

`scripts/replay_radlads_tiny_numerical_fixtures.py` writes P50 import and replay
reports for the P49 hidden, logits, WKV matrix state, shift state, and stepwise
surfaces. Failing numerical comparisons are acceptable and expected at this
stage; unsupported or missing surfaces never count as passes. P50 does not add
Pallas, TPU performance work, real training, full checkpoint import, or a
Hugging Face model class.

## Phase 51 — RADLADS Replay Non-Finite Diagnosis and Stabilization

P51 adds diagnosis-first replay tooling for the real P49 RADLADS tiny fixtures.
The new diagnostics path summarizes replay tensors, detects the first
non-finite tensor in replay order, writes parameter sanity reports for the
shared RADLADS payload, and records which replay stages were instrumented.

The key source-backed fix is replay-profile selection: P50 forced the
all-math RADLADS path for every fixture, but P49 generated four of the five
real cases with `all_radlads_math=False`. P51 now replays those simple cases
with the matching non-all-math profile, preserving q/k/v bias import while
avoiding inactive low-rank paths that were poisoning replay with suspicious
source parameters. The explicit all-math fixture remains diagnostic and may
still fail if the source payload itself is non-finite.

P51 does not add Pallas, full RADLADS parity claims, throughput claims,
training claims, or model-quality claims.

## Phase 53 — RADLADS vs QRWKV Comparable Output Fixture Parity

P53 adds a bounded head-to-head fixture path for live RADLADS outputs versus
QRWKV-XLA replay outputs. The clean parameter payload is generated through the
existing P52 `deterministic_finite` path with seed `5353`; QRWKV uses the
existing P50/P52 replay importer, and RADLADS is attempted only through the
local live source runtime.

The new scripts are:

```bash
python scripts/generate_radlads_qrwkv_head_to_head_fixtures.py
python scripts/compare_radlads_qrwkv_head_to_head.py
```

Artifacts are written under `artifacts/p53_radlads_qrwkv_head_to_head/`,
including `manifest.json`, `head_to_head_comparison_report.json`, and
`P53_HEAD_TO_HEAD_REPORT.md`. Reports preserve mapped/defaulted/missing/
unsupported/shape-mismatch parameter buckets and per-case/per-surface shape,
dtype, finite flags, error metrics, and reasons.

If live RADLADS cannot load or execute the clean payload, P53 records the exact
blocker and marks comparisons unsupported. It does not fabricate RADLADS
outputs from QRWKV behavior.

P53 does not add Pallas, TPU optimization, real training, Qwen-scale export,
HF `PreTrainedModel` support, multi-host sharding, or tolerance loosening.

## Phase 54 — RADLADS Clean Payload Loading and Export

P54 adds a clean payload loader for the tiny deterministic_finite RADLADS
parameter archive used by P53. The new loader classifies live RADLADS boundary
surfaces explicitly, keeps the 16 unsupported leaves and 4 gate shape
mismatches visible in the report, and applies only deterministic safe defaults
with parity-risk caveats instead of random initialization.

P54 also adds a standalone exporter for clean RADLADS outputs and output
manifests so the head-to-head comparison can consume precomputed RADLADS and
QRWKV outputs when they are available. The comparison path remains honest when
either side is missing and continues to avoid touching the RADLADS repo
contents.

P54 does not add Pallas, tolerance loosening, RADLADS repo vendoring, or any
claim of full RADLADS parity.

P54 implementation note: the local clean loader now loads the tiny deterministic
payload successfully, adapts the four gate surfaces with deterministic rank
truncation, excludes 34 payload-only surfaces as not needed for the tiny case,
and leaves no missing-required leaves in the live path.

## Phase 55 — RADLADS State/Layout Parity Diagnostics

P55 separates the remaining parity issue into two honest pieces: hidden_states
convention mismatch and a small but finite wkv_matrix_state residual. It adds
surface layout audit reports, non-mutating candidate-transform analysis, and a
clearer stepwise classification so non-stepwise cases are marked not_applicable
instead of vaguely missing_source.

P55 does not implement Pallas. It does not prove training throughput or model
quality. It only diagnoses and, where possible, fixes tiny local CPU
state/layout parity.


## 2026-05-13 — Weekly distillation
- P56 adds a trace-first WKV residual pass after P55. The first divergent stage is `log_w`, the WKV state-after trace row is now explicit, and update-order candidate analysis says `as_is` is still best while the residual remains finite.
- P56 is diagnostic-only: no recurrence math fix landed, logits and shift_state stay green, and Pallas remains blocked until the WKV matrix-state residual is explained.

## Phase 57 — RADLADS log_w Decay Parity Caliper

P57 narrows the P56 first-divergence result to a dedicated `log_w` source audit.
It loads RADLADS `log_w` rows from JSONL trace artifacts, captures QRWKV
`log_w` from a current model run through diagnostics, compares the surfaces, and
evaluates candidate formula calipers for orientation, sign, activation,
base-term, dtype, and axis handling.

Artifacts live under `artifacts/p57_log_w_decay_parity/`:
`log_w_parity_report.json`, `P57_LOG_W_PARITY.md`, `log_w_values.npz`,
`P57_LOG_W_CANDIDATES.md`, and `log_w_candidate_report.json`.

P57 is diagnostic-only. It does not patch model math, loosen tolerances, claim
broader RADLADS parity, or unblock Pallas by itself.

Observed run metrics:
- log_w first mismatch: `tiny_no_mask / L0 / H0 / T0` with max abs error `0.003779083490371704`
- trace rerun first divergent stage: `log_w`
- head-to-head summary: `attempted_comparisons=40`, `pass=12`, `fail=12`, `not_applicable=16`
- best passing surface stays `tiny_no_mask:logits`; `hidden_states` and `wkv_matrix_state` remain the main failures

## Phase 58 — RADLADS log_w / Decay Source-Backed Fix

P58 applies the tiny-replay source-backed fix for the RADLADS-vs-QRWKV
`log_w` / decay divergence identified by P56 and instrumented by P57. It keeps
the low-rank decay path active on `tiny_no_mask`, matches the inspected
RADLADS source formula, reruns the log_w caliper, and confirms the first WKV
trace divergence moves downstream.

P58 does not implement Pallas. It does not prove training throughput or model
quality. It only proves tiny local CPU log_w/decay parity status, and Pallas
remains blocked until WKV matrix-state parity is credible.

## Phase 59 — RADLADS WKV State Provenance

P59 adds a provenance-tracing layer for WKV state handoff. It records
initial-state equivalence, explicit-versus-implicit initial-state handoff,
token carry between `step` calls, full-sequence versus stepwise state/output
equivalence, and masked-token state deltas through the existing QRWKV student
state APIs.

The new artifacts live under `artifacts/p59_wkv_state_provenance/` when the
trace script is run locally. The module and scripts are diagnostic-only:
`src/qrwkv_xla/parity/radlads_wkv_state_provenance.py`,
`scripts/trace_radlads_qrwkv_wkv_state_provenance.py`, and
`scripts/compare_radlads_qrwkv_wkv_state_provenance.py`.

P59 does not rewrite recurrence math, widen tolerances, alter the P58 `log_w`
fix, implement Pallas, or claim broader RADLADS parity.

## Phase 60 — Real RADLADS/QRWKV WKV State Provenance

P60 converts the P59 provenance layer from synthetic QRWKV-only diagnostics to
paired real tiny artifacts. The runner derives rows from cached
`artifacts/p54_confirmation` RADLADS/QRWKV outputs and the P58 post-fix WKV
trace, then writes explicit provenance labels for real-vs-synthetic and
cached-vs-regenerated status.

Artifacts live under `artifacts/p60_real_wkv_state_provenance/`:
`real_wkv_state_provenance_radlads.jsonl`,
`real_wkv_state_provenance_qrwkv.jsonl`,
`real_provenance_metadata.json`, `p60_real_state_provenance_report.json`,
`P60_RESULTS.md`, `TRACE_PROVENANCE.md`, case reports, hidden-state dependency
reporting, and `comparison/p60_real_wkv_state_provenance_report.json`.

P60 does not add Pallas, broaden the recurrence math rewrite, widen
tolerances, or synthesize missing RADLADS traces. Strict real-artifact mode
fails when only cached-derived outputs are available. The real comparison
still fails deterministically at `tiny_attention_mask /
initial_state_handoff / wkv_matrix_state`.

## Phase 61 — WKV Matrix-State Export Convention Audit

P61 audits the RADLADS-vs-QRWKV WKV matrix-state export and slot convention
after P60 narrowed the remaining real-artifact residual to handoff/export-style
comparisons. It identifies whether the remaining mismatch comes from the wrong
slot, pre/post-update convention, full-vs-stepwise export, cached artifact
semantics, or true recurrence math. It may apply only a minimal
source-backed comparison/export normalization and does not implement Pallas.

## Phase 62 — QRWKV-XLA WKV Update-Term / State-After Residual Parity

P62 adds a diagnostic trace layer for the remaining WKV update-term and
state-after residual after P58 fixed `log_w` parity and P61 preserved the
`as_is` slot/export conclusion. It normalizes the existing paired post-P58 WKV
trace artifacts into explicit stages: `state_before`, `decay_value`,
`decayed_state`, `k_for_update`, `v_for_update`, `update_outer_product`,
`update_term`, `state_after`, `state_after_for_next_token`, and
`state_after_exported`.

Artifacts live under `artifacts/p62_wkv_update_residual/`:
`wkv_update_residual_radlads.jsonl`, `wkv_update_residual_qrwkv.jsonl`,
`wkv_update_residual_comparison_report.json`, `P62_WKV_UPDATE_RESIDUAL.md`,
`wkv_update_residual_values.npz`, `wkv_update_residual_manifest.json`, and
`P62_RESULTS.md`.

Current P62 status is diagnostic failure with `kernel_ready: no`. The report
finds `decayed_state` as the first unavailable/divergent comparison stage,
counts `32` passing rows, `40` failing rows, and `84` unavailable rows, and
records the first RADLADS reconstruction residual at `tiny_no_mask / L0 / H0 /
T1` with max abs error `0.0005597441340796649` when reconstructing only
`decayed_state + update_outer_product`. That reconstruction deliberately omits
the uncaptured composite balance-state matmul term.

No source-backed fix is applied in P62. The next phase should capture
source-backed RADLADS and QRWKV `update_term` and `decayed_state` rows in the
live recurrence hooks, including the balance-state matmul term, before any
numeric correction.

## Phase 63 — WKV Live Update Hooks

P63 completes the live WKV update-hook surface after P62 narrowed the residual
to decayed_state/update_term/state_after but still lacked a source-backed
composite balance-state intermediate. It exposes or explicitly labels live vs
reconstructed rows for decayed_state, update_outer_product,
balance_state_matmul/composite_update_term, update_term, and state_after on
real paired RADLADS-vs-QRWKV artifacts. It does not implement Pallas and only
allows instrumentation/comparison fixes unless a separate source-backed
recurrence-fix phase is opened.

P63 keeps the recurrence semantics untouched unless a tiny source-backed
instrumentation bug is proven.

## Phase 64 — WKV Composite Balance-State Hook

P64 narrows the P63 live-hook surface to the balance-state WKV addend and the
composite update labels around it. The helper now recognizes
`composite_balance_update_term` alongside `balance_state_matmul` and
`composite_update_term`, with source aliases for live rows and labeled
diagnostic reconstruction where adjacent captured rows are sufficient.

Artifacts default to `artifacts/p64_composite_balance_hook/` and are produced
by `scripts/locate_radlads_qrwkv_wkv_composite_hook.py`,
`scripts/extract_radlads_qrwkv_wkv_composite_hook.py`, and
`scripts/compare_radlads_qrwkv_wkv_composite_hook.py`. If an external RADLADS
checkout is unavailable, the locator uses the local source tree as the
RADLADS-equivalent audit target and records that choice explicitly.

P64 preserves P58/P61/P63 behavior. It does not change recurrence math, add
Pallas, loosen tolerances, or broaden the update-residual diagnosis beyond the
composite/balance-state WKV hook extraction and comparison.
