# Decisions

## D001 — Rebuild instead of port
QRWKV-XLA will be a new JAX/XLA-first implementation inspired by RADLADS, not a
direct port of the PyTorch/CUDA/Triton repo.

## D002 — Teacher extraction remains PyTorch/HF
Teacher models will initially be loaded using PyTorch/Hugging Face tooling and
exported into reusable target artifacts.

## D003 — Student training is JAX/XLA-first
The recurrent student trainer will be implemented in JAX and designed for CPU
debug and TPU execution.

## D004 — Primary student architecture is RWKV7-style
RWKV7-style recurrence is the primary destination architecture.

## D005 — Qwen3.latest is a policy label
The primary teacher target is the latest viable Qwen3-line open-weight model
available at experiment time. Each run must resolve this to a concrete model ID
in metadata.

teacher:
  family: qwen
  primary_policy: latest_qwen3_open_weight_available_at_experiment_time
  current_primary_label: Qwen3.latest
  fallback_label: Qwen3.0

student:
  primary_architecture: RWKV7-style
  fallback_architecture: none currently
  optional_reference_architecture: RWKV6-style only if needed for
    debugging/comparison

## D006 — No disposable toy architecture
Early configs may be tiny, but all modules should be shaped like the real
system.

## D007 — Nyx/Subagent Workflow
QRWKV-XLA uses Nyx as the primary implementation agent, with Codex available as
a sub-agent for mechanical implementation, tests, and lint cleanup. The older
A/B/C prompt split is optional and only used when it helps.

## D008 — Dataclasses Before Heavy Schema Frameworks
Initial config and artifact schemas use standard-library dataclasses plus
explicit validation. Pydantic or other schema frameworks may be considered later
if the project needs stronger runtime validation or serialization features.

## D009 — NPZ Shards for Initial Target Bundles
Initial teacher target bundles use a manifest JSON plus NumPy `.npz` shards.
This keeps the first artifact store simple, CPU-only, inspectable, and testable.
Larger-scale storage formats such as Zarr, safetensors, or memory-mapped arrays
may be considered later when real teacher extraction scale demands it.

## D010 — Fake Exporter as Permanent Test Double
QRWKV-XLA will keep a deterministic fake teacher exporter as a permanent test
double. It exercises the same artifact-writing pathway as future real
exporters, allowing the pipeline to be tested without network, GPU, TPU,
PyTorch, or Hugging Face dependencies.

## D011 — Real Teacher Loading Deferred
P2 intentionally does not load Qwen, PyTorch, Hugging Face Transformers, or
tokenizers. The project will first stabilize exporter contracts and artifact
writing before adding heavyweight model dependencies.

## D012 — Strategy A Editable Install Workflow
QRWKV-XLA uses editable install as the blessed development and CI workflow:
`python -m pip install -e ".[dev]"`.

Scripts, tests, and validation commands should import `qrwkv_xla` through the
installed package metadata instead of setting `PYTHONPATH=src` or mutating
`sys.path` inside entrypoint scripts. `scripts/validate_local.py` mirrors CI but
does not install dependencies; environment setup is an explicit caller
responsibility.

## D013 — Local Validation Mirrors CI
Local validation should mirror the CI command sequence through
`scripts/validate_local.py` or the equivalent individual command list. CI and
local handoff checks should stay aligned so failures reproduce before handoff.

## D016 — RWKV7 Reference Core Before Optimized Kernel
Phase 4 introduces `rwkv7_reference` as an XLA-friendly recurrent reference
implementation for the student path. It exists to validate shape contracts,
masking semantics, JIT compatibility, and gradient flow before any optimized
RWKV7 kernel work. It must not be described as the final optimized kernel.

## D017 — Use `jax.lax.scan` for Token Recurrence
The RWKV7 reference path uses `jax.lax.scan` across sequence length so token
recurrence is represented in an XLA-friendly form rather than as a Python-side
token loop.

## D018 — Distillation Runtime Reuses the Smoke Trainer
Phase 5 extends the existing JAX smoke training path with a composed
distillation objective instead of creating a parallel trainer stack. The default
behavior remains hidden-state MSE, while stage configs can opt into registered
weighted loss terms.

## D019 — Logits KL Is Plumbed Before Student Logits
The Phase 5 loss registry includes `logits_kl`, but current student
implementations do not emit logits. Enabling logits KL therefore validates that
both student logits and target logits exist and fails clearly until a student
logits head is introduced.

## D020 — TPU Smoke Scripts Must Degrade Gracefully
TPU smoke scripts run on the available JAX backend by default and only fail for
missing TPU when `--require-tpu` is explicitly passed. This keeps CI and CPU-only
development reliable while still supporting hard TPU validation in real TPU
environments.

## D021 — `distill` Is Canonical
The canonical distillation package is `qrwkv_xla.distill`, and the canonical
CLI is `scripts/run_distill_stage.py`. Any `distillation` package or
`run_distillation_stage.py` script is a compatibility alias and should remain
thin if retained.

## D022 — Sharding Deferred Until After Single-Device Smoke
QRWKV-XLA will validate single-device XLA/JAX runtime behavior before adding
multi-device TPU sharding, `pjit`, or Pallas kernels.

## D023 — Hugging Face Teacher Export Is Optional

The HF/PyTorch teacher exporter is available through `.[teacher-hf]` only.
Torch and Transformers must not become base or dev dependencies, and default
local/CI validation must continue to use fake exported targets without network,
GPU, TPU, or Qwen smoke requirements.

## D062 — Data Parallel pmap Before Model Parallelism

QRWKV-XLA implements replicated-parameter data parallelism with `pmap` before
attempting model parallelism or parameter sharding. This proves the training
loop can use multiple TPU devices while keeping checkpoint and optimizer
semantics simple.

## D063 — Multi-Device Smoke Tests Must Be Skip-Safe

Default validation must not require multiple devices. Multi-device smoke
scripts should skip cleanly unless explicitly run with
`--require-multiple-devices`.

## D064 — P21 Checkpoints Save Unreplicated State

P21 saves checkpoints from first-device/unreplicated params and optimizer
state. Sharded checkpoint formats are deferred until parameter sharding or
multi-host training requires them.

## D024 — Tiny HF Model Before Qwen
P7 validates the first real teacher exporter with a tiny public Hugging Face
causal language model rather than Qwen3.latest. Qwen export remains the target,
but the backend should be proven on small models before larger policy-driven
runs.

## D025 — Hidden Dimensions Are Inferred From Real Teacher Outputs
For real HF exports, `hidden_size` and `num_layers` in the manifest are
inferred from actual model outputs rather than trusted from config. Prompt
batches define output shards.

## D026 — Qwen Policy Resolution Is Offline
`Qwen3.latest` and related labels resolve only through local policy files.
There is no automatic web/API lookup for latest Qwen models.

## D027 — Qwen Export Is Manual-Only
Default validation may dry-run Qwen policy prep, but it must not run real Qwen
export, download models, or require `teacher-hf`.

## D028 — HF Exporter Stays Generic
Qwen-specific policy behavior lives above the generic HF exporter. The HF
backend remains a model-id-driven exporter and should avoid Qwen-specific
special cases unless a concrete compatibility issue requires one.

## D029 — Pipeline Validation Is Canonical
`scripts/validate_pipeline.py` is the canonical end-to-end validation harness
for the safe local/CI pipeline. It must stay CPU-safe, offline, and runnable
with only `.[dev]` by default.

## D030 — Optional Validation Must Stay Explicit
Tiny HF validation requires `--include-hf`, and hard TPU validation requires
`--require-tpu`. Neither optional path should become part of default CI or
default handoff validation.

## D031 — Checkpoints Are JSON + NPZ
P10 checkpointing uses an inspectable JSON manifest plus `params.npz` arrays.
Pickle, object arrays, Orbax, and framework-specific checkpoint stacks are not
required for the local distillation continuation path.

## D032 — Checkpoints Stay Local Under `checkpoints/`
Generated checkpoints must live under `checkpoints/`, which is gitignored.
Default validation may create checkpoint smoke artifacts there, but they are
local state and must not be committed.

## D033 — Resume Steps Are Additional
When resuming distillation, `max_steps` means additional steps for that
invocation. The final checkpoint step is `loaded_step + max_steps`.

## D034 - Run Tracking Stays Local and Opt-In

Distillation tracking writes only local files under `runs/`: `run.json`,
`metrics.jsonl`, `summary.json`, and optionally a final checkpoint below the
run directory. It is disabled by default. External tracking services, databases,
and framework-specific logging stacks are not required for the baseline.

## D036 — Prompt Corpora Are File-Based JSONL

QRWKV-XLA uses local JSONL prompt corpora before adopting external dataset
tooling. This keeps prompt inputs inspectable, portable, dependency-light, and
easy to hash.

## D037 — Corpus Hashes Are Part of Export Provenance

Teacher target manifests record prompt corpus identity and hash when corpus
prompts are used. This lets later runs, checkpoints, and comparisons trace back
to the exact prompt set used for export.

## D038 — Prompt Order Affects Corpus Hash

P12 corpus hashes preserve record order. This is acceptable for now because
prompt order can affect batching and export behavior, and the goal is to track
exact export inputs rather than only unordered prompt membership.
## D039 — LM Head Before Generation

QRWKV-XLA adds student logits output before implementing text generation.
Logits are first needed for output-distribution distillation through KL loss;
autoregressive decoding can come later.

## D040 — Hidden-Only Checkpoints Can Become Logits Continuation Runs

The staged distillation path is hidden-state alignment first, then resumed
continuation with logits KL once a student LM head exists. Missing LM head
parameters may be initialized during a controlled resume from hidden-only
checkpoints.

## D041 — Tied Embeddings Are Optional

Student LM heads support optional tied embeddings where practical, but untied
LM heads remain the simple default for clear checkpoint behavior and shape
validation.
## D042 — Generation Smoke Before Quality Evaluation

QRWKV-XLA implements a tiny greedy generation smoke path before adding benchmark
or quality evaluation. The first goal is to verify checkpoint loading, logits
availability, decoding loop behavior, and artifact writing.

## D043 — Gradient Clipping Is Simple Global Norm

P18 uses dependency-light global norm clipping inside the distillation train
step. Clipping is applied after gradient computation and before optimizer
updates. Per-layer clipping, value clipping, adaptive clipping, distributed norm
reduction, and loss scaling remain deferred.

## D043 — SmokeTokenizer Avoids Required Tokenizer Dependencies

P14 uses a dependency-free smoke tokenizer for default generation tests so CI
remains CPU-only, network-free, and independent of Hugging Face tokenizer
packages.

## D044 — Cached Recurrent Inference Is Deferred

P14 generation may recompute the full sequence each token. Optimized recurrent
cached-state inference is deferred until the basic generation path is validated.

## D045 — Evaluation Snapshots Before Benchmarks

QRWKV-XLA adds fixed generation snapshots and simple sanity checks before
adopting full benchmark tooling. The goal is reproducible inspection and
regression detection, not quality scoring.

## D046 — Sanity Checks Are Not Quality Claims

P15 sanity checks only detect basic generation problems such as empty outputs or
extreme repetition. Passing sanity checks does not imply the model is useful,
correct, aligned, or instruction-following.

## D047 — Snapshot Comparison Reports Difference, Not Better/Worse

Evaluation snapshot comparison reports whether outputs changed for fixed
prompts. It does not judge whether the new output is better.

## D048 — Optimizers Stay Dependency-Light

P16 adds SGD, Adam, and AdamW directly instead of adding Optax. The local
optimizer surface is small enough to keep in-repo for now, and default CI must
remain CPU-only and network-free.

## D049 — AdamW Uses Decoupled Weight Decay

AdamW applies weight decay directly to parameters during the update. It does
not fold weight decay into gradients before Adam moment updates.

## D051 — Warmup-Cosine Before Advanced Schedules

QRWKV-XLA implements constant and warmup-cosine learning rate schedules before
adding more advanced scheduling strategies. Warmup-cosine matches the immediate
needs of real distillation while keeping scheduler behavior inspectable.

## D052 — Schedules Use Global Step Across Resume

Learning rate schedules are evaluated using global training step, not local
invocation step. This prevents LR warmup or decay from restarting accidentally
when resuming from checkpoints.

## D053 — Scheduler Metadata Is Provenance, Not Hard Resume Lock

Checkpoints record scheduler metadata for provenance. A resumed run may
intentionally use a different schedule, but this should be visible in run
metadata and summaries.
## D059 — Stage 1 Targets Attention Output Vectors, Not Weights

QRWKV-XLA targets teacher attention or mixer output vectors shaped
`[batch, layers, sequence, hidden]` rather than attention probability matrices.

## D060 — HF Attention Capture Is Manual-Only

Real HF/Qwen attention-output capture depends on model-specific hook layout, so
default CI stays fake-target based and manual capture remains opt-in.

## D061 — Student Mixer Outputs Are First-Class Training Signals

Students can expose per-layer mixer outputs so Stage 1 can train the recurrent
replacement pathway directly.
