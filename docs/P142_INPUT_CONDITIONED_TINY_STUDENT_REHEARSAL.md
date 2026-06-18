# P142 Input-Conditioned Tiny Student Rehearsal

P142 strengthens the P141 `fingerprint_corridor` runner mode with an
input-conditioned rehearsal. It still uses synthetic behavioral fingerprint
targets and remains corridor-only, but it now verifies that the training path
is attached to real student behavior.

## Purpose

P141 proved that the main runner can optimize P135 corridor loss over a real
registered student backend. P142 adds evidence that:

- different `input_ids` produce different student logits
- optimizer steps modify real trainable parameters
- corridor metrics and rehearsal diagnostics are finite
- checkpoint/report/summary artifacts remain coherent
- no teacher backend is required

This is not a quality result.

## Runner Mode

P142 does not add a new training mode. It uses:

```text
distillation.mode: fingerprint_corridor
fingerprint.input_conditioned_rehearsal: true
```

Example:

```bash
python scripts/run_distill_stage.py \
  --distill-mode fingerprint_corridor \
  --fingerprint-artifact tests/fixtures/behavioral_fingerprint/v0_1_with_exemplars_tiny \
  --student-backend current_qrwkv \
  --steps 4 \
  --batch-size 2 \
  --learning-rate 0.01 \
  --fingerprint-input-conditioned-rehearsal \
  --output-dir /tmp/qrwkv_p142_input_conditioned_rehearsal
```

The existing tiny P137/P140 fixture is sufficient because the registered
`current_qrwkv` backend emits different logits for distinct input sequences at
initialization.

## Diagnostics

P142 adds rehearsal metrics to the P141 metric surface:

- `fingerprint/rehearsal/input_conditioning_detected`
- `fingerprint/rehearsal/input_conditioning_delta_norm`
- `fingerprint/rehearsal/params_changed`
- `fingerprint/rehearsal/param_delta_norm`
- `fingerprint/rehearsal/initial_loss`
- `fingerprint/rehearsal/final_loss`
- `fingerprint/rehearsal/loss_delta`
- `fingerprint/rehearsal/loss_non_increasing`

`loss_non_increasing` remains diagnostic. The pass condition requires finite
losses, non-negative loss, completed optimizer steps, nonzero batch
consumption, detected input conditioning, changed params, and finite metrics.

## Report

When `input_conditioned_rehearsal` is enabled, the runner report uses:

- `phase: P142`
- `distill_mode: fingerprint_corridor`
- `training_path_kind: main_runner_fingerprint_corridor`
- `real_student_backend_integrated: true`
- `main_runner_integrated: true`
- `student_uses_input_ids: true`
- `teacher_required: false`
- `exemplar_reservoir_enabled: false`
- `input_conditioning_detected: true`
- `params_changed: true`
- `loss_non_increasing_required: false`

The Markdown summary states that this is an input-conditioned tiny rehearsal,
uses the main `fingerprint_corridor` runner mode, trains a real registered
student backend, requires no teacher backend, has no exemplar/mixed objective,
and leaves teacher-side capture for future work.

## Checkpoint Behavior

The rehearsal writes normal runner checkpoint artifacts through the existing
checkpoint helper:

- `checkpoints/final/checkpoint.json`
- `checkpoints/final/params.npz`

The checkpoint contains student params, optimizer state, and a fingerprint
target manifest that identifies `distill_mode: fingerprint_corridor`.

## Claims

P142 proves that the synthetic corridor-only fingerprint training receiver is
attached to input-conditioned real student behavior and can move real
parameters through the main runner.

P142 does not prove teacher-side fingerprint capture, exemplar or mixed
training, real teacher artifact quality, baseline superiority, quality-per-byte
improvement, TPU/GPU behavior, Pallas readiness, or production readiness.

## Next

P143 should begin teacher-side fingerprint capture skeleton work. P142 does not
start producer-side capture.
