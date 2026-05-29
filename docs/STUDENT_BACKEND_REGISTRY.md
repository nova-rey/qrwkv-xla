# StudentBackend Registry

P101 adds the student backend registry and architecture selection slot. It does
not add a second backend.

## Purpose

The registry prevents the current QRWKV backend from being an implicit forever
assumption. Student architecture is now selected by `architecture_id` while the
vocab contract and runtime remain separate inputs.

## Architecture ID Selection

The registry lives in `src/qrwkv_xla/students/registry.py`.

The first registered architecture id is:

```text
current_qrwkv
```

`architecture_id=None` defaults to `current_qrwkv`.

## Current Registered Architecture

`current_qrwkv` creates `CurrentQRWKVStudentBackend` over the existing
`rwkv7_qwen_reference` student config path. This preserves current behavior and
does not add another architecture.

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

## What P101 Proves

P101 proves:

- available architecture IDs can be listed
- default selection returns `CurrentQRWKVStudentBackend`
- explicit `current_qrwkv` selection works
- unknown IDs fail clearly
- vocab contracts with different vocab sizes produce matching backend logits
- reference remains default
- Pallas remains opt-in

## What P101 Does Not Prove

P101 does not add a second backend, implement `TinyDebugStudentBackend`, train,
add optimizer loops, add Qwen-specific student code, change runtime semantics,
change recurrence math, change fixture tensors, or promote Pallas.

## Future Phases

P102 should add a tiny/debug second student backend to prove architecture
swappability through this registry without changing teacher, vocab, or runtime
contracts.
