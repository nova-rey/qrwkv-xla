# Radjax Architecture Extraction Notes

P91 starts the post-Pallas architecture extraction path. The project is still
QRWKV-XLA, but the next phases are shaping it toward a teacher-agnostic and
student-runtime-agnostic Radjax-style platform.

## Current Extracted Boundaries

VocabContract:

- `VocabContract`: `src/qrwkv_xla/contracts/vocab.py`
- `validate_vocab_compatibility()`: `src/qrwkv_xla/contracts/compatibility.py`

This declares tokenizer and vocabulary identity emitted by a teacher or stored
target artifact. P97 keeps direct logits loss compatibility explicit and does
not add tokenizer remapping.

StudentConfig Selection:

- `SelectedStudentConfig`: `src/qrwkv_xla/students/config_selection.py`
- `qrwkv_student_config_from_vocab_contract()`

This instantiates vocab-dependent student dimensions from a `VocabContract`
while keeping architecture and runtime selection separate.

Teacher/Student Compatibility:

- `TeacherStudentCompatibility`: `src/qrwkv_xla/contracts/compatibility.py`
- `validate_direct_logit_eligibility()`
- `validate_store_for_student_config()`

This validates a teacher artifact contract against a selected student contract.
P99 classifies results as compatible, incompatible, or unsupported.

Direct Logit Eligibility:

Direct logits are allowed only for matching tokenizer/vocab contracts. P99 does
not add tokenizer remapping, cross-vocab adapters, or projection support.

TeacherBackend:

- `TeacherBackend` protocol: `src/qrwkv_xla/teachers/backend.py`
- `SyntheticTeacherBackend`: `src/qrwkv_xla/teachers/synthetic.py`
- `HFTeacherBackend`: `src/qrwkv_xla/teachers/hf.py`

This is the source of teacher targets. P94 only includes deterministic
synthetic emission; live HF/Qwen teacher backends are future work. Future real
teacher backends must emit or declare a `VocabContract`.

HFTeacherBackend:

This is the first generic optional backend for HF-style causal LM target
emission. P98 keeps it cache/local-files-only by default, optional-dependency
safe, model-id configurable, and not Qwen- or GPT-2-specific.

Tiny HF Causal-LM Teacher Specimen:

- `run_hf_teacher_specimen_smoke()`:
  `src/qrwkv_xla/teachers/hf_specimen_smoke.py`
- `run_hf_teacher_specimen_swap_smoke()`:
  `src/qrwkv_xla/teachers/hf_specimen_smoke.py`
- `scripts/run_hf_teacher_specimen_smoke.py`
- `scripts/run_hf_teacher_specimen_swap_smoke.py`

This is an optional cache-local live specimen smoke for the generic
`HFTeacherBackend`. P104 can emit and validate a real `full_logits`
`TeacherTargetStore` artifact when optional HF dependencies and local model
files are available. The specimen model id is configurable and does not become
a special architecture path. P105 proves the same path can represent multiple
teacher specimens with distinct model id, tokenizer id, and vocab metadata.

TeacherTargetStore:

- `TargetStoreMetadata`: `src/qrwkv_xla/targets/schema.py`
- `TeacherTargetStore`: `src/qrwkv_xla/targets/store.py`
- `iter_target_store_shard_ids()`:
  `src/qrwkv_xla/targets/multishard.py`
- `iter_offline_target_batches()`: `src/qrwkv_xla/targets/multishard.py`

This is the versioned offline target artifact contract. P93 stores metadata in
`metadata.json` and target arrays in local `.npz` shards. It decouples teacher
execution from student runtime. The existing model/tokenizer/vocab metadata can
reconstruct a `VocabContract`. P98 stores `full_logits` artifacts emitted by
synthetic or generic HF teacher backends in the same canonical layout.

Multi-Shard TargetStore:

P106 stores target artifacts across multiple canonical shard files, supports
deterministic shard-id iteration, validates shard count/presence/shape/dtype,
and preserves the `metadata.json` plus `shards/shard-XXXXX.npz` layout.

Tiny Dataset Pipeline:

- `TinyTextExample`: `src/qrwkv_xla/data/tiny_dataset.py`
- `batch_tiny_text_examples()`: `src/qrwkv_xla/data/tiny_dataset.py`
- `run_tiny_dataset_pipeline_smoke()`:
  `src/qrwkv_xla/data/tiny_dataset_pipeline.py`

P107 proves tiny raw text examples can become deterministic batches, flow
through fake HF-style teacher emission, land in sharded `TeacherTargetStore`
artifacts, validate through P106 multi-shard helpers, and consume through the
P95 offline target path. It does not add large dataset plumbing, streaming,
training, Qwen-specific code, or tokenizer remapping.

Checkpoint / Resume / Export Rehearsal:

- `run_checkpoint_resume_export_rehearsal()`:
  `src/qrwkv_xla/checkpointing/rehearsal.py`
- `scripts/run_checkpoint_resume_export_rehearsal.py`

P108 proves a tiny student checkpoint can save, reload, export through the
existing HF/safetensors interchange path, reload that export, and preserve
hidden-state/logit outputs. It does not add production checkpointing,
distributed checkpointing, production HF model classes, training, Qwen-specific
code, or tokenizer remapping.

P108.1 closes the resume-update gap with
`run_checkpoint_resume_update_rehearsal()`: a tiny deterministic MSE state
saves after update steps, reloads exactly, resumes at least one more update,
and reports finite resumed loss plus correct step advancement.

Runtime Environment Preflight:

- `run_runtime_environment_preflight()`:
  `src/qrwkv_xla/xla/environment_preflight.py`
- `scripts/run_runtime_environment_preflight.py`

P109 adds read-only JAX/TPU/transparent-hugepage inspection before burn
readiness. It reports runtime visibility and environment hygiene without
training, benchmarking, pjit, sharding changes, or Pallas promotion.

Mini Eval Harness:

- `run_mini_eval_harness()`: `src/qrwkv_xla/eval/mini_eval.py`
- `scripts/run_mini_eval_harness.py`

P110 adds compatibility-gated tiny artifact evaluation. It selects a student
backend through the registry, iterates offline target shards, and reports
finite MSE loss plus tiny count metrics. It does not add lm_eval, benchmarks,
training, model-quality claims, Qwen-specific code, or tokenizer remapping.

OfflineTargetConsumption:

- `OfflineTargetBatch`: `src/qrwkv_xla/targets/consumption.py`
- `load_offline_target_batch()`
- `mse_logits_loss()`

This is the first student-side consumption boundary for stored teacher
targets. P95 loads validated offline shards and computes a finite logits MSE
against the current student logits surface without requiring a live teacher.

TinyOverfitRehearsal:

- `TinyOverfitResult`: `src/qrwkv_xla/training/tiny_overfit.py`
- `run_tiny_overfit_rehearsal()`

This is the first tiny stored-target update loop. P96 uses offline synthetic
targets, a tiny trainable logit head, P95 logits MSE, and local SGD to prove
loss can move in a deterministic CPU-friendly rehearsal. It does not prove
full QRWKV training readiness.

Real-Teacher Offline Consumption:

- `RealTeacherConsumptionResult`:
  `src/qrwkv_xla/targets/real_teacher_consumption.py`
- `run_real_teacher_offline_consumption_smoke()`

This consumes generic `HFTeacherBackend`-style artifacts through the
compatibility-gated offline target path. P100 requires P99 direct-logit
eligibility before loading a batch or computing student-side loss.

Real-Teacher Overfit Rehearsal:

- `RealTeacherOverfitResult`:
  `src/qrwkv_xla/training/real_teacher_overfit.py`
- `run_tiny_real_teacher_overfit_rehearsal()`

This consumes a generic `HFTeacherBackend`-style artifact through the registry,
compatibility gate, offline target path, and a tiny trainable logit head. P103
proves the real-teacher-style target path can drive finite downward loss
movement without full QRWKV parameter training.

StudentBackend:

- `StudentBackend` protocol: `src/qrwkv_xla/students/backend.py`
- `CurrentQRWKVStudentBackend`: `src/qrwkv_xla/students/current_backend.py`

This is the architecture-facing wrapper around the validated QRWKV student
core. A selected architecture consumes a compatible student config; P97 does
not make CurrentQRWKV the permanent architecture.

StudentBackend Registry:

- `create_student_backend()`: `src/qrwkv_xla/students/registry.py`
- Architecture ids: `current_qrwkv`, `tiny_debug`

This selects a student architecture by `architecture_id`. P101 added the
registry slot; P102 adds `tiny_debug` as a socket-test second backend. Vocab
contract, student architecture, and runtime remain separate choices.

TinyDebugStudentBackend:

- `TinyDebugStudentBackend`: `src/qrwkv_xla/students/tiny_debug_backend.py`

This is a deterministic debug/socket-test backend proving architecture
swappability. It is not a production model and does not support Pallas.

StudentRuntime:

- `StudentRuntime` protocol: `src/qrwkv_xla/students/student_runtime.py`
- `ReferenceJaxStudentRuntime`
- `PallasStudentRuntime`

This is the execution path boundary. The current runtime choices are
`reference_jax` and `pallas`. Runtime remains separate from architecture and
vocab contract selection.

These wrappers delegate to existing QRWKV student and WKV behavior. They do not
change recurrence math, WKV runtime policy, trainer behavior, teacher export,
target storage, checkpoint formats, Pallas semantics, or fixture tensors.

## Runtime Policy

- `reference` remains the default WKV runtime.
- `pallas` remains opt-in.

## Later Boundaries

Current modularity status:

- teacher backend boundary: present and selectable by artifact source
- vocab contract boundary: present and selected from teacher metadata
- student architecture boundary: present and selectable through registry
- runtime boundary: present and selectable separately
- second student backend smoke: present
- real-teacher-style tiny rehearsal: present
- optional HF causal-LM specimen smoke: present
- second teacher-specimen swap smoke: present
- multi-shard target-store smoke: present
- tiny dataset pipeline smoke: present
- checkpoint/resume/export rehearsal: present
- resume-update closure: present
- runtime environment preflight: present
- mini eval harness: present

Likely future extraction layers:

- broader student backend support
- burn readiness
- broader target support
- real training/eval
- Trainer boundary
- Evaluator and parity gates

P96 implements only a tiny controlled rehearsal. It does not add live teacher
backends, broad trainer extraction, Qwen loading, GPU/TPU requirements, or
model-quality claims.

P97 implements only token/vocab contracts and current-student config selection.
It does not add real HF/Qwen teachers, tokenizer remapping, architecture
registry work, or runtime behavior changes.

P98 implements only generic HF-style teacher emission. It does not add
student consumption, tokenizer remapping, Qwen-specific code, training, or
runtime behavior changes.

P99 implements only teacher/student compatibility validation and direct-logit
eligibility classification. It does not consume real teacher artifacts with a
student or add remapping/adapters.

P100 implements only compatibility-gated real-teacher offline consumption smoke.
It does not train, update parameters, remap tokenizers, or make Qwen-specific
claims.

P107 implements only a tiny deterministic dataset pipeline smoke. It does not
add large dataset pipelines, streaming, training, Qwen-specific support,
tokenizer remapping, runtime changes, or Pallas promotion.

P108 implements only a tiny checkpoint/resume/export rehearsal. It does not
add production checkpointing, distributed checkpointing, production HF export,
training, Qwen-specific support, tokenizer remapping, runtime changes, or
Pallas promotion.

P108.1 implements only the missing resume-after-load update proof. It does not
add production checkpointing, distributed checkpointing, new export formats,
large training, Qwen-specific support, tokenizer remapping, runtime changes, or
Pallas promotion.

P109 implements only runtime environment hygiene. It does not train,
benchmark, add pjit/sharding, require TPU/GPU/JAX in baseline tests, change
StudentRuntime or StudentBackend semantics, or promote Pallas.

P110 implements only a mini eval/reporting smoke over tiny target artifacts. It
does not add benchmark datasets, lm_eval integration, training, optimizer
loops, model-quality claims, Qwen-specific support, tokenizer remapping,
runtime changes, or Pallas promotion.

P101 implements only the StudentBackend registry slot and default
`current_qrwkv` architecture selection. It does not add a second backend.

P102 adds only `tiny_debug`, a non-production second backend smoke. It closes
the current modularity arc without changing CurrentQRWKV behavior.

P103 adds only a tiny real-teacher-style overfit rehearsal. It uses fake
HFTeacherBackend-style baseline artifacts, keeps the compatibility gate
mandatory, and does not add Qwen-specific code, tokenizer remapping, full
training, or runtime behavior changes.

P104 adds only an optional tiny HF causal-LM specimen smoke. It keeps baseline
tests fake/mock-safe, defaults to local-files-only behavior, and does not add
student consumption, training, tokenizer remapping, Qwen-specific behavior, or
GPT-2-specific architecture assumptions.

P105 adds only a second teacher-specimen swap smoke. It proves model-id-driven
HF teacher specimen swapping without adding student consumption, training,
tokenizer remapping, Qwen-specific behavior, or GPT-2-specific architecture
assumptions.

P106 adds only a multi-shard TargetStore smoke. It preserves the canonical
store layout, validates/iterates multiple shards, and does not add a dataset
pipeline, training, tokenizer remapping, or runtime behavior changes.
