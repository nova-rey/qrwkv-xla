# Teacher/Student Compatibility

P99 adds the explicit teacher/student compatibility gate for direct-logit
eligibility.

## Purpose

Teacher artifacts and students must agree on tokenizer and vocabulary identity
before direct logits can be compared. P99 prevents silent vocab mismatch and
makes compatibility decisions structured rather than implicit.

Direct logits require matching vocab/tokenizer contracts. P99 does not remap
tokenizers or vocabularies.

## Statuses

`CompatibilityStatus` lives in `src/qrwkv_xla/contracts/compatibility.py`:

- `compatible`: the requested target/loss mode is allowed
- `incompatible`: the mode is supported, but teacher and student contracts do
  not match
- `unsupported`: P99 does not support the requested target type, loss mode, or
  student contract source

`TeacherStudentCompatibility` carries status, reason, target type, loss mode,
teacher contract, student contract, and a `compatible` boolean property.

## Direct-Logit Rules

For P99, direct-logit eligibility supports `target_type` values `full_logits`
and synthetic full-logits-style artifacts. Supported loss modes are
`direct_logits` and `mse_logits`.

The result is compatible only when:

- `tokenizer_id` matches
- `vocab_size` matches
- `tokenizer_hash` matches when both sides provide one
- special token mappings match when both sides provide mappings

Mismatches return `incompatible` with a specific reason. Hidden-state targets,
projection losses, and other unimplemented modes return `unsupported`.

## Examples

Compatible:

```text
teacher synthetic-a vocab 8 + student synthetic-a vocab 8 + full_logits
```

Incompatible:

```text
teacher synthetic-a vocab 8 + student synthetic-a vocab 16
teacher synthetic-a + student synthetic-b
teacher hash a + student hash b
```

Unsupported:

```text
hidden_states + direct_logits
full_logits + hidden_projection
student config without a vocab_contract
```

## What P99 Does Not Do

P99 does not add tokenizer remapping, cross-vocab adapters, hidden-state
projection, real HF artifact consumption by a student, training, optimizer
loops, Qwen-specific code, StudentBackend behavior changes, StudentRuntime
behavior changes, recurrence math changes, or Pallas promotion.

P99 validates compatibility for the selected student config/backend. It does
not assume CurrentQRWKV is the only future student architecture. Future
StudentBackend registry phases should provide their own student contract
extraction.

## Future Phases

P100 can consume a real-teacher offline target only after this compatibility
gate passes. Later phases can add architecture registries and additional
eligibility modes without implying tokenizer remapping.
