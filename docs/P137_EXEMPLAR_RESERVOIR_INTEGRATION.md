# P137 Exemplar Reservoir Integration

P137 adds an optional high-resolution exemplar reservoir to
`behavioral_fingerprint` artifacts. It is a sibling path to the P133 corridor
target loader and does not mix exemplar loss with corridor training yet.

## Manifest Extension

Artifacts may include:

```json
"exemplar_reservoir": {
  "enabled": true,
  "payload_type": "dense_probs",
  "loss": "kl",
  "num_records": 4,
  "shards": [
    {
      "path": "exemplars/exemplars-00000.jsonl",
      "num_records": 4
    }
  ]
}
```

The field is optional. Existing P132-P136.1 artifacts without exemplars remain
valid. Loading exemplars with `require_exemplars=True` fails clearly when the
reservoir is absent.

P137 supports `dense_probs` only. Each exemplar row requires:

- `example_id`
- fixed-length `input_ids`
- `position`
- dense `teacher_probs`
- non-negative `weight`

Optional row metadata includes `mode_id`, `interestingness_score`, and
`reason_codes`.

## Validation

`validate_fingerprint_artifact(...)` now validates exemplar reservoirs when
present:

- reservoir `enabled`, `payload_type`, `loss`, shard paths, and record counts
- required exemplar fields
- `len(input_ids) == sequence.max_seq_len`
- positions inside `[0, max_seq_len)`
- token ids inside the teacher vocabulary
- `len(teacher_probs) == teacher.vocab_size`
- finite, non-negative probabilities summing to `1.0 +/- 1e-5`
- finite, non-negative weights
- optional `mode_id` references known modes
- optional metadata has the expected type

## Loader

The public loader is:

```python
from qrwkv_xla.artifacts import load_fingerprint_exemplars

dataset = load_fingerprint_exemplars(
    "tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny",
    batch_size=2,
)
```

It returns a `FingerprintExemplarDataset` with:

- `num_records`
- `vocab_size`
- `max_seq_len`
- `iter_records()`
- `iter_batches()`

Batch fields are fixed-shape NumPy arrays for `input_ids`, `position`,
`teacher_probs`, `weight`, `mode_id`, and `interestingness_score`. Missing
`mode_id` uses sentinel `-1`.

The loader preserves manifest order unless `shuffle=True`; shuffle is
deterministic under `seed`. `max_records` and `drop_remainder` match the P133
target loader behavior.

## Loss

The public loss utilities are:

```python
from qrwkv_xla.training import (
    compute_fingerprint_exemplar_loss,
    compute_fingerprint_exemplar_loss_at_positions,
)
```

`compute_fingerprint_exemplar_loss(student_logits, batch)` consumes logits
shaped `[batch, vocab]` and computes KL teacher-to-student from dense teacher
probabilities. The position wrapper consumes `[batch, seq, vocab]` logits and
selects `batch.position` before computing the same loss.

## CLI

Use:

```bash
python scripts/inspect_fingerprint_exemplars.py \
  tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny \
  --batch-size 2 \
  --max-batches 1
```

The CLI reports artifact shape, exemplar count, batch shapes, mode ids, and
reason codes.

## Claims

P137 proves only that tiny synthetic dense exemplar reservoirs can be validated,
loaded, inspected, and scored with standalone KL loss utilities. It does not
add mixed corridor + exemplar training, main runner integration, teacher
generation, real student-backend training, CSL/cascaded exemplar compression,
dynamic top-k, learned critics, TPU/GPU burns, benchmark quality claims, or
WKV/runtime math changes.
