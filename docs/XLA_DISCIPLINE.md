# XLA Discipline

## Why stable shapes matter

JAX/XLA compiles around shapes and dtypes. If training paths change shapes
between steps, the runtime can trigger unnecessary recompiles and make TPU smoke
results noisy or misleading.

## Recurrence discipline

The recurrent student path should keep token recurrence in XLA-safe forms such
as `jax.lax.scan` rather than Python-side token loops in hot paths.

## JIT boundaries

Keep the training step as the main JIT boundary. Avoid building separate ad hoc
trainer worlds for smoke paths when the existing distillation runner can be
wrapped instead.

## Static config discipline

The stage config should define the runtime path clearly enough that the same
shape discipline holds across CPU validation and later TPU smoke runs.

## Runtime inspection

Use `python scripts/xla_inspect.py` to inspect:

- JAX availability
- JAX version
- default backend
- visible devices
- visible platforms

## Current XLA-Safe Path

```text
fake target export
 -> target inspection
 -> run_distill_stage.py
 -> tpu_distill_smoke.py
```

## Avoiding duplicate trainer paths

`tpu_distill_smoke.py` is a small smoke wrapper around the existing distillation
runtime. It should not become a second independent trainer stack.

## Known limitations

Phase 6 does not add:

- TPU sharding
- `pjit`
- Pallas kernels
- production performance optimization
- real teacher loading
- PyTorch/HF extraction in the student runtime path
