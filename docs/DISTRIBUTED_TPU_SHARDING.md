# Distributed TPU Sharding

## Purpose

Phase 21 adds a minimum viable multi-device data-parallel smoke path for
QRWKV-XLA. It is meant to prove that the training loop can use more than one JAX
accelerator device without rewriting the whole stack.

## What P21 Supports

- device topology discovery
- batch sharding across the leading axis
- replicated params and optimizer state
- `lax.pmean` gradient and metric averaging
- skip-safe `pmap` smoke CLIs for Stage 1 distillation and Stage 3 CE
- checkpoint save from unreplicated first-device state

## What P21 Does Not Support

- model parallelism
- parameter sharding
- `pjit` / mesh partitioning
- multi-host orchestration
- Orbax sharded checkpointing
- performance tuning claims

## Strategy

P21 uses plain replicated-parameter data parallelism:

```text
params/state replicated on every device
batch reshaped to [device_count, per_device_batch, ...]
local loss/grads computed per device
grads and metrics averaged with pmean
each device applies the same optimizer update
checkpoint saved from first-device/unreplicated state
```

Active device count defaults to the largest divisor of batch size that is no
larger than the visible local device count, optionally capped by `--device-count`.

## CLI Usage

Stage 1 attention/mixer smoke:

```bash
python scripts/export_teacher_targets.py --config configs/teacher_export_stub_attention.yaml
python scripts/pmap_distill_smoke.py --config configs/distill_stage1_attention_pmap_smoke.yaml
```

Stage 3 CE smoke:

```bash
python scripts/pmap_lm_smoke.py --config configs/lm_stage3_pmap_smoke.yaml
```

Hard multi-device check:

```bash
python scripts/pmap_distill_smoke.py \
  --config configs/distill_stage1_attention_pmap_smoke.yaml \
  --require-multiple-devices \
  --min-device-count 2
```

## Skip Behavior

If not enough divisible devices are available, the smoke scripts print the JAX
backend/device topology and exit with:

- code `0` when multi-device is optional
- nonzero when `--require-multiple-devices` is passed

This keeps default CI and local validation CPU-safe.

## TPU Pod Runbook

1. Install `.[dev]`.
2. Export fake attention targets.
3. Run `scripts/pmap_distill_smoke.py --require-multiple-devices --min-device-count 2`.
4. Record backend, visible devices, active device count, per-device batch size,
   and final loss.
5. If checkpointing is enabled, confirm the saved checkpoint came from
   unreplicated first-device state.

## Limitations

This is only data-parallel smoke validation. It does not solve fitting models
larger than one device, sharded optimizer memory, or real multi-host TPU pod
training.
