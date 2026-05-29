# Vocab Contract

P97 adds a first-class token/vocabulary contract between teacher metadata,
stored target artifacts, and student configuration selection.

The governing rule is:

```text
Teacher picks the language/vocab contract.
Student is born with that contract.
Student architecture is selected separately.
Runtime is selected separately.
```

## Purpose

Direct logits losses require the teacher target artifact and student logits to
mean the same token ids. P97 makes that compatibility explicit instead of
relying on hard-coded vocab assumptions.

## Contract Fields

`VocabContract` lives in `src/qrwkv_xla/contracts/vocab.py` and currently
records:

- `tokenizer_id`
- `vocab_size`
- `tokenizer_hash`
- `model_id`
- `model_family`
- `special_tokens`
- `chat_template_hash`

The minimal validation rules are: `tokenizer_id` must be non-empty,
`vocab_size` must be positive, and any special token ids must be inside the
vocabulary range.

## Flow

Teacher and target metadata declare the tokenizer/vocab identity. A
`VocabContract` can be reconstructed from `TargetStoreMetadata`, then passed to
student config selection so vocab-dependent student dimensions are born from
the same contract.

P97 adds a current QRWKV selection helper in
`src/qrwkv_xla/students/config_selection.py`. It preserves non-vocab
architecture fields, sets `vocab_size` from the contract, and keeps runtime
selection separate.

## Compatibility

`validate_vocab_compatibility()` lives in
`src/qrwkv_xla/contracts/compatibility.py`.

For P97, direct logits consumption is compatible only when:

- `tokenizer_id` matches
- `vocab_size` matches
- `tokenizer_hash` matches when both sides provide one

P97 does not support arbitrary tokenizer remapping. It does not map vocab A to
vocab B, add token adapters, or reinterpret logits across tokenizer contracts.

## Future Phases

Future real teacher backends should emit or declare this contract. Future
student backends and registries should consume the same contract without making
CurrentQRWKV the permanent architecture. P97 does not add a real HF/Qwen
teacher, a student architecture registry, training, recurrence math changes, or
runtime semantic changes.
