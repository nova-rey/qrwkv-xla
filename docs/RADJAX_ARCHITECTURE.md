# Radjax Architecture Extraction Notes

P91 starts the post-Pallas architecture extraction path. The project is still
QRWKV-XLA, but the next phases are shaping it toward a teacher-agnostic and
student-runtime-agnostic Radjax-style platform.

## Current Extracted Boundaries

TeacherBackend:

- `TeacherBackend` protocol: `src/qrwkv_xla/teachers/backend.py`
- `SyntheticTeacherBackend`: `src/qrwkv_xla/teachers/synthetic.py`

This is the source of teacher targets. P94 only includes deterministic
synthetic emission; live HF/Qwen teacher backends are future work.

TeacherTargetStore:

- `TargetStoreMetadata`: `src/qrwkv_xla/targets/schema.py`
- `TeacherTargetStore`: `src/qrwkv_xla/targets/store.py`

This is the versioned offline target artifact contract. P93 stores metadata in
`metadata.json` and target arrays in local `.npz` shards. It decouples teacher
execution from student runtime.

StudentBackend:

- `StudentBackend` protocol: `src/qrwkv_xla/students/backend.py`
- `CurrentQRWKVStudentBackend`: `src/qrwkv_xla/students/current_backend.py`

This is the architecture-facing wrapper around the validated QRWKV student
core.

StudentRuntime:

- `StudentRuntime` protocol: `src/qrwkv_xla/students/student_runtime.py`
- `ReferenceJaxStudentRuntime`
- `PallasStudentRuntime`

This is the execution path boundary. The current runtime choices are
`reference_jax` and `pallas`.

These wrappers delegate to existing QRWKV student and WKV behavior. They do not
change recurrence math, WKV runtime policy, trainer behavior, teacher export,
target storage, checkpoint formats, Pallas semantics, or fixture tensors.

## Runtime Policy

- `reference` remains the default WKV runtime.
- `pallas` remains opt-in.

## Later Boundaries

Likely future extraction layers:

- offline target consumption
- tiny overfit rehearsal
- small Qwen-family smoke
- Trainer boundary
- Evaluator and parity gates

Those are not implemented in P94.
