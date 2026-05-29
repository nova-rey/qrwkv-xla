# StudentBackend Registry

P101 adds the student backend registry and architecture selection slot. P102
adds the first second-backend socket test, `tiny_debug`.

## Purpose

The registry prevents the current QRWKV backend from being an implicit forever
assumption. Student architecture is now selected by `architecture_id` while the
vocab contract and runtime remain separate inputs.

## Architecture ID Selection

The registry lives in `src/qrwkv_xla/students/registry.py`.

Registered architecture ids:

```text
current_qrwkv
tiny_debug
```

`architecture_id=None` defaults to `current_qrwkv`.

## Current Registered Architectures

`current_qrwkv` creates `CurrentQRWKVStudentBackend` over the existing
`rwkv7_qwen_reference` student config path. This preserves current behavior.

`tiny_debug` creates `TinyDebugStudentBackend`, a deterministic socket-test
backend added in P102 to prove registry swappability. It is not a production
architecture.

## Factory Inputs

`create_student_backend()` accepts:

- `vocab_contract`
- `architecture_id`
- `base_config`
- `runtime`

The vocab contract drives vocab-dependent student config dimensions. The
runtime remains a separate selection, with reference as default and Pallas
explicit opt-in.

## Separation

P101 preserves the architecture law:

```text
Teacher picks the language/vocab contract.
Student is born with that contract.
Student architecture is selected separately.
Runtime is selected separately.
```

## What P101/P102 Prove

P101/P102 prove:

- available architecture IDs can be listed
- default selection returns `CurrentQRWKVStudentBackend`
- explicit `current_qrwkv` selection works
- explicit `tiny_debug` selection works
- unknown IDs fail clearly
- vocab contracts with different vocab sizes produce matching backend logits
- reference remains default
- Pallas remains opt-in

## What P101/P102 Do Not Prove

P101 did not add a second backend. P102 adds only a debug/socket-test backend.
Neither phase trains, adds optimizer loops, adds Qwen-specific student code,
changes runtime semantics, changes recurrence math, changes fixture tensors, or
promotes Pallas.

## Future Phases

The real-teacher rehearsal and burn-readiness arc can continue without changing
the registry contract: `current_qrwkv` remains the default, and `tiny_debug`
remains a non-production socket-test backend.
