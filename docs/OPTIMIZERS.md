# Optimizers

QRWKV-XLA has a small internal optimizer layer for staged distillation. It is
intentionally dependency-light and does not require Optax, Flax, Equinox, or
Orbax.

Supported optimizer types:

- `sgd`
- `adam`
- `adamw`

The default remains SGD for smoke compatibility.

## Config

Distillation configs use the `distillation.optimizer` section:

```yaml
distillation:
  optimizer:
    type: adamw
    learning_rate: 0.0003
    beta1: 0.9
    beta2: 0.999
    epsilon: 1.0e-8
    weight_decay: 0.01
```

CLI overrides:

```bash
python scripts/run_distill_stage.py \
  --config configs/distill_stage0_stub.yaml \
  --optimizer adamw \
  --learning-rate 0.0003 \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --adam-epsilon 1.0e-8 \
  --weight-decay 0.01
```

Validation rejects unknown optimizer types, non-positive learning rates,
invalid beta/epsilon values, negative weight decay, and `type: adam` with
non-zero weight decay. Use `adamw` for weight decay.

## Learning Rate Schedules

`optimizer.learning_rate` is the base learning rate. Distillation may apply a
local schedule per step and pass that scheduled value to the optimizer update.
The default schedule is `constant`, which preserves fixed learning-rate
behavior. See `docs/LR_SCHEDULES.md`.

## AdamW Decay

AdamW uses decoupled weight decay. Weight decay is applied directly to
parameters as part of the parameter update, not folded into gradients before
Adam moment updates.

For P16, AdamW weight decay applies to all parameter leaves. Bias/1D exclusion
or richer masks are deferred.

## State

Optimizer state is a JAX pytree:

- SGD: step plus empty slots
- Adam/AdamW: step plus `m` and `v` slot trees matching params

Distillation checkpoints persist optimizer config and optimizer state in the
existing JSON + NPZ format so resume continues Adam/AdamW moments. Older
checkpoints without optimizer state still load; resume initializes fresh
optimizer state at the checkpoint step and records a note.

## Gradient Clipping

When `distillation.gradients.max_grad_norm` is set, the distillation train step
clips by global norm before calling the optimizer. Adam and AdamW first and
second moments are therefore updated from clipped gradients. See
`docs/GRADIENT_CLIPPING.md`.
