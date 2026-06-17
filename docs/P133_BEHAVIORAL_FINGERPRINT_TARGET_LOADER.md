# P133 Behavioral Fingerprint Target Loader

P133 adds the bridge from a validated `behavioral_fingerprint` artifact to
typed Python records and fixed-shape NumPy batches. It reuses the P132 validator
by default and does not implement teacher generation, student statistics,
corridor loss, exemplar reservoirs, or trainer integration.

## API

```python
from qrwkv_xla.artifacts import (
    FingerprintLoaderConfig,
    FingerprintTargetDataset,
    load_fingerprint_targets,
)
```

`load_fingerprint_targets(...)` returns a `FingerprintTargetDataset` with:

- `num_records`
- `vocab_size`
- `max_seq_len`
- `tracked_stats`
- `iter_records()`
- `iter_batches()`

Records are represented as `FingerprintTargetRecord`. Batches are represented
as `FingerprintBatch` and contain NumPy arrays for `input_ids`, `position`,
`mode_id`, every corridor bound, and `weight`.

## Fixed-Shape Boundary

P132 validates rows with `len(input_ids) <= sequence.max_seq_len`. P133 is
stricter: every loaded row must have `len(input_ids) == sequence.max_seq_len`.

There is no hidden pad token or implicit padding rule in P133. Future schema
versions can add an explicit pad policy if needed.

## Ordering

With `shuffle=False`, records are loaded in manifest shard order and physical
JSONL file order.

With `shuffle=True`, shuffling uses NumPy's deterministic random generator and
is stable for a given seed.

`max_records` truncates the manifest/file-order record stream before optional
shuffling. `drop_remainder` controls final partial batch emission.

## Inspect CLI

```bash
python scripts/inspect_fingerprint_targets.py \
  tests/fixtures/behavioral_fingerprint/v0_1_valid_tiny \
  --batch-size 2 \
  --max-batches 1
```

Expected output includes:

```text
artifact_type=behavioral_fingerprint
artifact_version=0.1
num_records=8
batch_0_input_ids_shape=(2, 16)
```

## Claims

P133 proves only that validated behavioral fingerprint targets can be loaded
into fixed-shape NumPy batches. It does not compute student distribution
statistics, apply corridor loss, modify training objectives, prove model
quality, claim production readiness, add Qwen/tokenizer remapping support,
promote Pallas, or change WKV/runtime math.
