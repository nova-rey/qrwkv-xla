# P138 Mixed Corridor + Exemplar Smoke

P138 adds the first standalone behavioral fingerprint smoke that consumes both
pieces of a fingerprint artifact:

- corridor map targets from the P133 loader
- dense exemplar landmarks from the P137 loader

The smoke remains tiny, CPU-only, and independent of the main staged runner.

## Objective

Each optimizer step consumes one cycled corridor batch and one cycled exemplar
batch. The objective is explicit:

```text
mixed_loss =
    corridor_loss_weight * corridor_loss
  + exemplar_loss_weight * exemplar_kl_loss
```

Both weights must be non-negative, and at least one must be positive. A
single-branch weighted run is allowed, but the mixed smoke still loads both
branches and requires nonzero consumed batches for both.

## API

```python
from qrwkv_xla.training import (
    FingerprintMixedSmokeConfig,
    run_mixed_fingerprint_training_smoke,
)

result = run_mixed_fingerprint_training_smoke(
    FingerprintMixedSmokeConfig(
        artifact_dir="tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny",
        output_dir="/tmp/qrwkv_fingerprint_p138_mixed_smoke",
        steps=3,
        corridor_batch_size=2,
        exemplar_batch_size=2,
    )
)
```

The result reports requested/completed optimizer steps, corridor and exemplar
batches consumed, initial/final mixed loss, branch loss deltas, finite/non-
negative status, and diagnostic loss non-increase.

## CLI

The existing smoke CLI keeps corridor-only behavior by default and adds a mixed
mode:

```bash
python scripts/run_fingerprint_smoke.py \
  --mode mixed \
  --artifact tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny \
  --steps 3 \
  --corridor-batch-size 2 \
  --exemplar-batch-size 2 \
  --corridor-loss-weight 1.0 \
  --exemplar-loss-weight 1.0 \
  --output-dir /tmp/qrwkv_fingerprint_p138_mixed_smoke
```

## Metrics

P138 emits mixed, corridor, exemplar, and batch-consumption metrics, including:

- `fingerprint/mixed_loss_total`
- `fingerprint/corridor_loss_total`
- `fingerprint/corridor_inside_all_rate`
- `fingerprint/exemplar_loss_total`
- `fingerprint/exemplar_kl_loss`
- `fingerprint/exemplar_cross_entropy`
- `fingerprint/exemplar_teacher_entropy`
- `fingerprint/corridor_batches_consumed`
- `fingerprint/exemplar_batches_consumed`
- `fingerprint/optimizer_steps_completed`

## Report

The report identifies the path as `standalone_mixed_fingerprint_smoke` and keeps
limitations explicit:

- `smoke_student_kind: tiny_position_logit_head`
- `smoke_student_uses_input_ids: false`
- `main_runner_integrated: false`
- `real_student_backend_integrated: false`
- `teacher_required: false`
- `exemplar_reservoir_enabled: true`
- `hf_download_required: false`
- `gpu_or_tpu_required: false`

Pass/fail requires completed requested optimizer steps, nonzero corridor and
exemplar batch consumption, finite mixed losses/metrics, and non-negative
initial/final mixed losses. `mixed_loss_non_increasing` remains diagnostic only.

## Claims

P138 proves only that the tiny standalone smoke can combine corridor-map loss
and dense exemplar-landmark KL in one optimizer loop. It does not add real
student-backend training, main runner integration, teacher generation, live
teacher queries, CSL/cascaded exemplar payloads, dynamic top-k, learned critics,
TPU/GPU burns, quality claims, production readiness, Pallas promotion, or
WKV/runtime math changes.
