# P125 TPU 100-Example Training Smoke Result

P125 attempted to consume the P124 100-example tomes on a TPU v6e-16 slice.
JAX device listings observed 4 processes and 16 TPU devices.

## Cascaded Path

The `cascaded_soft_labels_v1` tome was uploaded and available on the TPU, but
the P117.1 real burn path rejected it before training:

```text
ValueError: P117.1 real training supports dense TeacherTextbook target types {'full_logits', 'synthetic'}, got 'cascaded_soft_labels_v1'
```

P125 therefore did not prove cascaded real-burn training.

## Dense Path

The dense tome was built on the TPU from the uploaded 100-line JSONL, but its
canonical `dense_logits` metadata was rejected by the P117.1 burn path because
that path still expected the legacy `full_logits` name.

## Salvage Path

For a dense TPU salvage smoke, the dense artifact metadata was patched from
`dense_logits` to `full_logits`. That patched dense artifact completed a
confirmed real TPU train-step smoke:

- Status: `pass`
- Device backend: `tpu`
- Steps completed/requested: `25/25`
- Batch size: `4`
- Examples available: `100`
- Examples consumed: `100`
- Unique examples consumed: `100`
- Reuse count: `0`
- Checkpoint written and nonzero: `true`
- Loss trace: finite
- Blockers/warnings: empty

Recorded loss movement:

- Initial loss: `0.0003387359611224383`
- Final loss: `0.00032750004902482033`
- Delta: `-1.1235912097617984e-05`

## Observability Issue

Per-worker reports showed distinct hostnames, but multiple workers reported
`worker_id: "0"`. Each worker also appeared to consume the full 100-example
set. P125 is therefore not evidence of correct distributed example sharding,
and the current burn path should be treated as unsharded until proven
otherwise.

## Claims Not Made

P125 does not prove model quality, distributed training readiness, production
training readiness, cascaded training readiness, Qwen support, tokenizer
remapping support, or large-scale performance.
