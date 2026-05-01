# Learning Rate Schedules

QRWKV-XLA supports a tiny local learning-rate scheduler surface for distillation:

- `constant`: always uses `optimizer.learning_rate`.
- `warmup_cosine`: linearly warms up, then cosine-decays toward `min_learning_rate`.

The default is `constant`, so existing configs preserve fixed learning-rate behavior.

## Config

```yaml
distillation:
  optimizer:
    type: adamw
    learning_rate: 0.001
    weight_decay: 0.01
  lr_schedule:
    type: warmup_cosine
    warmup_steps: 100
    total_steps: 10000
    min_learning_rate: 0.00001
```

`warmup_cosine` requires `total_steps > warmup_steps`. `min_learning_rate` must
be non-negative and no larger than the base optimizer learning rate.

## CLI

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_logits_stub.yaml \
  --optimizer adamw \
  --learning-rate 0.001 \
  --weight-decay 0.01 \
  --lr-schedule warmup_cosine \
  --warmup-steps 10 \
  --total-steps 1000 \
  --min-learning-rate 0.00001
```

The scheduled smoke config is `configs/distill_stage0_adamw_schedule_stub.yaml`.

## Resume Behavior

Schedules are evaluated with the global training step. A checkpoint saved at
step 100 and resumed for 10 more local steps evaluates schedule steps 100
through 109 and saves the final checkpoint at step 110.

Checkpoints record scheduler metadata for provenance. A resumed run may use a
different schedule; the runner allows it and records a note when metadata
differs.

## Tracking

Tracked runs include scheduler config metadata. Per-step metrics include
`learning_rate`, `base_learning_rate`, `global_step`, and `local_step`; metric
extras include `lr_schedule_type`.

## Limitations

P17 did not add Optax, one-cycle or polynomial schedules, per-layer learning
rates, parameter freezing, or sharded scheduler state. P18 adds simple global
gradient norm clipping as a separate train-step guardrail.
