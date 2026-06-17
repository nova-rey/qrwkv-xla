# P132 Behavioral Fingerprint Artifact Schema

P132 introduces the first `behavioral_fingerprint` artifact contract and a
validator for version `0.1`. This phase is intentionally limited to schema and
structural validation. It does not generate teacher fingerprints, load
fingerprint batches for training, compute student distribution statistics,
implement corridor loss, or add exemplar reservoirs.

## Layout

```text
fingerprint_artifact/
  manifest.json
  modes.json
  targets/
    targets-00000.jsonl
  exemplars/                 # optional from P137 onward
    exemplars-00000.jsonl
```

The schema is sharded from the start. Each manifest shard entry names a JSONL
target shard and its expected `num_records`.

## Manifest

`manifest.json` must declare:

- `artifact_type: behavioral_fingerprint`
- `artifact_version: 0.1`
- `created_by`
- teacher metadata: `model_name`, `tokenizer_name`, positive `vocab_size`, `dtype`
- sequence metadata: positive `max_seq_len`, optional `target_positions`
- non-empty `stats.tracked`
- `modes_file`
- non-empty `target_shards`
- optional `exemplar_reservoir`

If `sequence.target_positions` is present, it must equal the total records
across all listed target shards.

P137 extends the manifest with an optional `exemplar_reservoir` block for dense
landmark examples. The block is valid when absent. When present for P137 it
must use `payload_type: dense_probs`, `loss: kl`, a non-negative `num_records`,
and non-empty shard entries with physical JSONL counts that match the manifest.

## Modes

`modes.json` contains a non-empty `modes` list. Each mode has a unique integer
`mode_id`, a non-empty `name`, and bounds for every stat in
`manifest.stats.tracked`.

Bounds require numeric `min` and `max`, `min <= max`, and optional `mean` inside
the same interval. Probability-like stats are constrained to `[0.0, 1.0]`;
`entropy` must be non-negative.

Tracked stat meanings are fixed by P134:

- `entropy`: natural-log distribution entropy, `-sum(p * ln(p))`.
- `top1_margin`: probability-space `p_top1 - p_top2`.
- `top8_mass`: cumulative probability mass of the largest 8 tokens.
- `top32_mass`: cumulative probability mass of the largest 32 tokens.
- `tail_mass`: probability mass outside the top-32 tokens, `1.0 - top32_mass`.

## Target Rows

Each JSONL row describes one corridor target:

- string `example_id`
- non-negative integer `position`
- non-empty integer `input_ids`
- known integer `mode_id`
- row-level bounds for every tracked stat

All token ids must satisfy `0 <= token_id < teacher.vocab_size`, and
`len(input_ids)` must not exceed `sequence.max_seq_len`.

P133 adds the stricter loader boundary for training-shaped batches:
`len(input_ids)` must equal `sequence.max_seq_len`. P132 remains the broader
schema validator.

Row bounds must remain inside the selected mode bounds:

```text
mode.min <= row.min <= row.max <= mode.max
```

P132 keeps this strict so P133 can load fixed-shape batches against a predictable
contract.

## Exemplar Rows

P137 exemplar rows are optional and live under shards referenced by
`exemplar_reservoir`. Each row requires:

- string `example_id`
- integer `position` inside `[0, sequence.max_seq_len)`
- fixed-length integer `input_ids` matching `sequence.max_seq_len`
- dense `teacher_probs` of length `teacher.vocab_size`
- finite, non-negative `teacher_probs` summing to `1.0 +/- 1e-5`
- finite, non-negative `weight`

Optional `mode_id` must reference a known mode when present. Optional
`reason_codes` must be strings, and optional `interestingness_score` must be
finite.

## Validator

Use:

```bash
python scripts/validate_fingerprint_artifact.py \
  tests/fixtures/behavioral_fingerprint/v0_1_valid_tiny
```

Expected success output shape:

```text
status=pass artifact_type=behavioral_fingerprint version=0.1 shards=1 records=8 modes=2 warnings=0
```

The Python API is:

```python
from qrwkv_xla.artifacts import validate_fingerprint_artifact
```

The validator returns a `ValidationResult` with `ok`, `blockers`, `warnings`,
and summary `metadata`.

## Claims

P132 proves only that a tiny synthetic behavioral fingerprint artifact can be
validated and that malformed artifacts fail with actionable blockers. It does
not prove teacher generation, student training, corridor loss, model quality,
production readiness, Qwen support, tokenizer remapping, Pallas readiness, or
WKV/runtime changes.
