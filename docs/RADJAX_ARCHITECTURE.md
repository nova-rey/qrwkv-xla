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
safe, and not Qwen-specific.

TeacherTargetStore:

- `TargetStoreMetadata`: `src/qrwkv_xla/targets/schema.py`
- `TeacherTargetStore`: `src/qrwkv_xla/targets/store.py`

This is the versioned offline target artifact contract. P93 stores metadata in
`metadata.json` and target arrays in local `.npz` shards. It decouples teacher
execution from student runtime. The existing model/tokenizer/vocab metadata can
reconstruct a `VocabContract`. P98 stores `full_logits` artifacts emitted by
synthetic or generic HF teacher backends in the same canonical layout.

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

StudentBackend:

- `StudentBackend` protocol: `src/qrwkv_xla/students/backend.py`
- `CurrentQRWKVStudentBackend`: `src/qrwkv_xla/students/current_backend.py`

This is the architecture-facing wrapper around the validated QRWKV student
core. A selected architecture consumes a compatible student config; P97 does
not make CurrentQRWKV the permanent architecture.

StudentBackend Registry:

- `create_student_backend()`: `src/qrwkv_xla/students/registry.py`
- Current architecture id: `current_qrwkv`

This selects a student architecture by `architecture_id`. P101 registers only
the current QRWKV backend and keeps vocab contract, architecture, and runtime as
separate choices.

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

Likely future extraction layers:

- second debug backend smoke
- broader student backend support
- second student backend smoke
- tiny real-teacher overfit rehearsal
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

P101 implements only the StudentBackend registry slot and default
`current_qrwkv` architecture selection. It does not add a second backend.
