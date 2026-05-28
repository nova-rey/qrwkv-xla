# Radjax Architecture Extraction Notes

P91 starts the post-Pallas architecture extraction path. The project is still
QRWKV-XLA, but the next phases are shaping it toward a teacher-agnostic and
student-runtime-agnostic Radjax-style platform.

## P91 Scope

P91 introduces only the student-side wrapper boundary:

- `StudentBackend` protocol: `src/qrwkv_xla/students/backend.py`
- `CurrentQRWKVStudentBackend`: `src/qrwkv_xla/students/current_backend.py`

The concrete backend delegates to existing QRWKV student behavior. It does not
change recurrence math, WKV runtime policy, trainer behavior, teacher export,
target storage, checkpoint formats, Pallas semantics, or fixture tensors.

## Runtime Policy

- `reference` remains the default WKV runtime.
- `pallas` remains opt-in.

## Later Boundaries

Likely future extraction layers:

- StudentRuntime split
- TeacherBackend
- TeacherTargetStore
- Trainer boundary
- Evaluator and parity gates

Those are not implemented in P91.
