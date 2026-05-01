# Gradient Clipping

QRWKV-XLA supports simple global gradient norm clipping in the distillation
runner. The implementation is local JAX code and does not add Optax or another
training dependency.

## Behavior

When enabled, the train step computes the global norm across floating gradient
leaves after loss/gradient computation and before the optimizer update:

```text
global_norm = sqrt(sum(sum(grad_leaf ** 2)))
clip_scale = min(1.0, max_grad_norm / (global_norm + clip_epsilon))
```

The optimizer receives the clipped gradients, so Adam and AdamW moment state is
updated from clipped gradients. If `max_grad_norm` is `null` or disabled by CLI,
the original gradient tree is passed through unchanged.

## Config

```yaml
distillation:
  gradients:
    max_grad_norm: 1.0
    clip_epsilon: 0.000001
```

`max_grad_norm: null` disables clipping. `clip_epsilon` must be positive.

The clipped AdamW smoke config is
`configs/distill_stage0_adamw_clipped_stub.yaml`.

## CLI

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --max-grad-norm 1.0 \
  --clip-epsilon 0.000001
```

Use `--disable-grad-clipping` to override a config that enables clipping.
`--max-grad-norm` and `--disable-grad-clipping` are mutually exclusive.

## Metrics and Metadata

Per-step metrics include:

- `grad_global_norm`
- `grad_clipped_global_norm`
- `grad_clip_scale`
- `grad_was_clipped`
- `max_grad_norm`

Tracked runs and checkpoints record the gradient clipping config for
provenance. Existing checkpoints without gradient metadata still load.
